#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
# 调试模式
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import re
import json
import csv
import argparse
from transformers.utils import logging
logging.disable_progress_bar()
from dataclasses import dataclass
from typing import List, Tuple

import torch
from PIL import Image

# 引入量化配置类
from transformers import AutoProcessor, MllamaForConditionalGeneration, BitsAndBytesConfig
from prompt_policy import build_group_prompt, infer_prompt_group

ANSWER_CONSTRAINT = (
    "Answer in <=25 words.\n"
    "Return exactly one line in this format:\n"
    "Final: PRESENT/ABSENT; Confidence: 0-100; Evidence: <location + visual cue>.\n"
    "Confidence reflects how certain you are of your answer."
)

FINAL_ANSWER_RE = re.compile(r"Final:\s*(PRESENT|ABSENT)", re.IGNORECASE)
PRESENT_ABSENT_RE = re.compile(r"\b(PRESENT|ABSENT)\b", re.IGNORECASE)

def normalize_yesno(text: str) -> str:
    if not text: return "OTHER"
    m_final = FINAL_ANSWER_RE.search(text)
    if m_final: return "YES" if m_final.group(1).upper() == "PRESENT" else "NO"
    matches = PRESENT_ABSENT_RE.findall(text)
    if matches: return "YES" if matches[-1].upper() == "PRESENT" else "NO"
    return "OTHER"

@dataclass
class Item:
    image_file: str
    image_num: int
    case_id: int
    absent_object: str

def load_annotation(ann_json: str) -> Tuple[str, str, List[int], List[Item]]:
    with open(ann_json, "r", encoding="utf-8") as f:
        ann = json.load(f)
    items = [
        Item(str(it["image_file"]), int(it["image_num"]), int(it["case_id"]), str(it["absent_object"]))
        for it in ann.get("items", [])
    ]
    return ann.get("dataset_name", "dataset"), ann.get("image_dir", ""), ann.get("tones", [1,2,3,4,5]), items

def ensure_image(path: str) -> Image.Image:
    if not os.path.exists(path): raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")

def build_human_prompt(image_file: str, obj: str, tone: int) -> str:
    base_prompt = build_group_prompt(image_file=image_file, absent_object=obj, tone=tone, prefer_yesno_tokens=False)
    if infer_prompt_group(image_file=image_file, absent_object=obj) in ("legacy_text", "legacy_time"):
        return base_prompt
    return f"{base_prompt}\n\n{ANSWER_CONSTRAINT}"

def load_model_and_processor(model_path: str, dtype: str, device_map: str, local_only: bool, use_4bit: bool):
    print(f"Loading processor from: {model_path}")
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=local_only)
    
    # 修复 Padding
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    print(f"Loading model from: {model_path}")
    print(f"  - Device Map: {device_map}")
    print(f"  - Quantization (4-bit NF4): {use_4bit}")

    # --- 核心修改：改为 4-bit (NF4) 量化，解决 view size 错误 ---
    quantization_config = None
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
    
    model = MllamaForConditionalGeneration.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        torch_dtype=getattr(torch, dtype) if not use_4bit else None,
        device_map=device_map, 
        local_files_only=local_only,
    )
    model.eval()
    return model, processor

@torch.no_grad()
def generate_answer_hf(model, processor, image: Image.Image, human_prompt: str, max_new_tokens: int):
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": human_prompt}
        ]}
    ]
    
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    
    inputs = processor(
        image,
        input_text,
        return_tensors="pt"
    )
    
    # 移动到 GPU
    inputs = inputs.to(model.device)
    
    # --- 修复 5: 确保输入张量连续 (解决 view size 错误的关键) ---
    # 对 input_ids 和 pixel_values 进行 contiguous 操作
    if hasattr(inputs, "input_ids"):
        inputs["input_ids"] = inputs["input_ids"].contiguous()
    if hasattr(inputs, "pixel_values"):
        inputs["pixel_values"] = inputs["pixel_values"].contiguous()
    
    # 兼容获取 Vocab Size
    if hasattr(model.config, "text_config"):
        vocab_size = model.config.text_config.vocab_size
    elif hasattr(model.config, "vocab_size"):
        vocab_size = model.config.vocab_size
    else:
        vocab_size = processor.tokenizer.vocab_size

    # 越界熔断保护
    input_ids = inputs.input_ids
    if input_ids.max().item() >= vocab_size:
        inputs.input_ids = torch.where(
            input_ids >= vocab_size,
            processor.tokenizer.pad_token_id, 
            input_ids
        )

    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
    )
    
    input_len = inputs.input_ids.shape[1]
    generated_ids = output[:, input_len:]
    return processor.decode(generated_ids[0], skip_special_tokens=True).strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_dir", type=str, default="/data/jianzhiy/Project/New_dataset")
    ap.add_argument("--ann_json", type=str, default="/data/jianzhiy/Project/annotations.json")
    ap.add_argument("--model_path", type=str, default="meta-llama/Llama-3.2-11B-Vision-Instruct") 
    ap.add_argument("--out_csv", type=str, default="/data/jianzhiy/Project/results_llama_success.csv")
    ap.add_argument("--tones", type=str, default="1,2,3,4,5")
    ap.add_argument("--max_new_tokens", type=int, default=256) 
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["float16","bfloat16"])
    
    # 保持单卡
    ap.add_argument("--device_map", type=str, default="cuda:0") 
    
    # --- 启用 4-bit 量化 (推荐) ---
    ap.add_argument("--no_quant", action="store_true", help="Disable quantization (Will OOM)")
    
    ap.add_argument("--local_only", action="store_true")
    
    args = ap.parse_args()

    # 默认开启量化，除非手动指定 --no_quant
    use_4bit = not args.no_quant

    if not os.path.exists(args.ann_json):
        print(f"Error: Annotation file not found: {args.ann_json}")
        return

    _, ann_image_dir, ann_tones, items = load_annotation(args.ann_json)
    image_dir = args.image_dir or ann_image_dir
    tones = [int(x) for x in args.tones.split(",")] if args.tones else list(map(int, ann_tones))

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)

    print(f"Loading model... (4-bit NF4 Quantization: {use_4bit})")
    try:
        model, processor = load_model_and_processor(
            model_path=args.model_path,
            dtype=args.dtype,
            device_map=args.device_map,
            local_only=args.local_only,
            use_4bit=use_4bit
        )
    except Exception as e:
        print(f"❌ Model Load Failed: {e}")
        return

    print(f"Start processing {len(items)} images...")

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Image_File","Image_Num","Case_ID","Absent_Object","Tone","Prompt","Response","Norm","Affirmed","Hallucinated"])

        for i, it in enumerate(items):
            img_path = os.path.join(image_dir, it.image_file)
            print(f"[{i+1}/{len(items)}] {it.image_file} | obj='{it.absent_object}'")

            try:
                img = ensure_image(img_path)
            except Exception as e:
                print(f"  ❌ skip: {e}")
                continue

            for tone in tones:
                prompt = build_human_prompt(it.image_file, it.absent_object, tone)

                try:
                    resp = generate_answer_hf(model, processor, img, prompt, args.max_new_tokens)
                except Exception as e:
                    resp = f"[GEN_ERROR] {e}"
                    print(f"  Error details: {e}")

                norm = normalize_yesno(resp)
                affirmed = 1 if norm == "YES" else 0
                
                print(f"    Tone {tone} -> Norm: {norm} | Resp: {resp[:50].replace(chr(10), ' ')}...")

                writer.writerow([it.image_file,it.image_num,it.case_id,it.absent_object,tone,prompt,resp,norm,affirmed,affirmed])
                f.flush()

    print(f"DONE! Results saved to {args.out_csv}")

if __name__ == "__main__":
    main()
