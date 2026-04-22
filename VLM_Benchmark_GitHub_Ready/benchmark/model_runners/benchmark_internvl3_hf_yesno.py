#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import csv
import argparse
from dataclasses import dataclass
from typing import List, Tuple

import torch
from PIL import Image

from transformers import AutoProcessor, AutoModelForImageTextToText
from prompt_policy import build_group_prompt, infer_prompt_group

ANSWER_CONSTRAINT = (
    "Answer in <=25 words.\n"
    "Return exactly one line in this format:\n"
    "Final: PRESENT/ABSENT; Confidence: 0-100; Evidence: <location + visual cue>.\n"
    "Confidence reflects how certain you are of your answer."
)

FINAL_ANSWER_RE = re.compile(r"Final:\s*(PRESENT|ABSENT)", re.IGNORECASE)
PRESENT_ABSENT_RE = re.compile(r"\b(PRESENT|ABSENT)\b", re.IGNORECASE)
YESNO_RE = re.compile(r"\b(YES|NO)\b", re.IGNORECASE)
AFFIRM_RE = re.compile(r"\b(yes|present|contains|contain|visible|there is|i see|appears|shown|can be seen|in the image)\b", re.IGNORECASE)
NEGATE_RE = re.compile(r"\b(no|not present|absent|no sign|not visible|cannot see|can't see|does not contain|doesn't contain|no visible)\b", re.IGNORECASE)

def normalize_yesno(text: str) -> str:
    if not text:
        return "OTHER"

    # A) 优先抽取 Final: PRESENT/ABSENT
    m_final = FINAL_ANSWER_RE.search(text)
    if m_final:
        return "YES" if m_final.group(1).upper() == "PRESENT" else "NO"

    # B) 其次抽取最后一个 PRESENT/ABSENT
    matches = PRESENT_ABSENT_RE.findall(text)
    if matches:
        return "YES" if matches[-1].upper() == "PRESENT" else "NO"

    # C) 再匹配 YES/NO
    m = YESNO_RE.search(text)
    if m:
        return m.group(1).upper()

    # D) 兜底关键词
    if NEGATE_RE.search(text):
        return "NO"
    if AFFIRM_RE.search(text):
        return "YES"
    return "OTHER"

def heuristic_affirmed(resp: str, norm: str) -> int:
    if norm == "YES":
        return 1
    if norm == "NO":
        return 0
    if not resp:
        return 0
    r = resp.strip()
    if NEGATE_RE.search(r) and not AFFIRM_RE.search(r):
        return 0
    if AFFIRM_RE.search(r):
        return 1
    return 0

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
        Item(
            str(it["image_file"]),
            int(it["image_num"]),
            int(it["case_id"]),
            str(it["absent_object"])
        )
        for it in ann.get("items", [])
    ]
    return ann.get("dataset_name", "dataset"), ann.get("image_dir", ""), ann.get("tones", [1,2,3,4,5]), items

def ensure_image(path: str) -> Image.Image:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")

def build_human_prompt(image_file: str, obj: str, tone: int) -> str:
    base_prompt = build_group_prompt(image_file=image_file, absent_object=obj, tone=tone, prefer_yesno_tokens=False)
    if infer_prompt_group(image_file=image_file, absent_object=obj) in ("legacy_text", "legacy_time"):
        return base_prompt
    return f"{base_prompt}\n\n{ANSWER_CONSTRAINT}"

def shard_items(items: List[Item], shard: int, num_shards: int) -> List[Item]:
    if num_shards <= 1:
        return items
    return [it for idx, it in enumerate(items) if (idx % num_shards) == shard]

def load_model_and_processor(model_path: str, dtype: str, device_map: str, local_only: bool):
    torch_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32
    }[dtype]

    print(f"Loading processor from: {model_path}")
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=local_only)

    print(f"Loading model from: {model_path} dtype={dtype} device_map={device_map}")
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        local_files_only=local_only
    )
    model.eval()
    return model, processor, torch_dtype

@torch.no_grad()
def generate_answer_hf(model, processor, image: Image.Image, human_prompt: str, max_new_tokens: int, torch_dtype):
    """
    InternVL3-8B-hf 推荐写法：processor.apply_chat_template + model.generate
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": human_prompt},
            ],
        }
    ]

    # NOTE: 按 InternVL3-8B-hf README 的写法
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt"
    )

    # 把 tensor 放到模型设备；并把浮点张量转到 bf16/fp16，解决你那个：
    # Input type (float) and bias type (bfloat16) should be the same
    inputs = inputs.to(model.device)
    for k, v in inputs.items():
        if torch.is_floating_point(v):
            inputs[k] = v.to(dtype=torch_dtype)

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1
    )

    # 只取新生成部分
    in_len = inputs["input_ids"].shape[-1]
    new_tokens = generated_ids[0, in_len:]
    out = processor.decode(new_tokens, skip_special_tokens=True)
    return out.strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_dir", type=str, default="/data/jianzhiy/Project/New_dataset")
    ap.add_argument("--ann_json", type=str, default="/data/jianzhiy/Project/annotations.json")
    ap.add_argument("--model_path", type=str, default="/data/jianzhiy/models/InternVL3-8B-hf")
    ap.add_argument("--out_csv", type=str, default="/data/jianzhiy/Project/results_internvl.csv")
    ap.add_argument("--tones", type=str, default="1,2,3,4,5")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--dtype", type=str, default="bf16", choices=["fp16","bf16","fp32"])
    ap.add_argument("--device_map", type=str, default="cuda", choices=["cuda","auto"])
    ap.add_argument("--local_only", action="store_true", help="use local files only (no HF download)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    args = ap.parse_args()

    if not os.path.exists(args.ann_json):
        print(f"Error: Annotation file not found: {args.ann_json}")
        return

    _, ann_image_dir, ann_tones, items = load_annotation(args.ann_json)
    image_dir = args.image_dir or ann_image_dir
    tones = [int(x) for x in args.tones.split(",")] if args.tones else list(map(int, ann_tones))
    items = shard_items(items, args.shard, args.num_shards)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)

    model, processor, torch_dtype = load_model_and_processor(
        model_path=args.model_path,
        dtype=args.dtype,
        device_map=args.device_map,
        local_only=args.local_only
    )

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
                    resp = generate_answer_hf(model, processor, img, prompt, args.max_new_tokens, torch_dtype)
                except Exception as e:
                    resp = f"[GEN_ERROR] {e}"

                norm = normalize_yesno(resp)
                affirmed = heuristic_affirmed(resp, norm)
                hallucinated = affirmed  # 你的定义：YES 就算 hallucinated（因为 absent object）
                print(f"    Tone {tone} -> {resp} (Norm: {norm})")

                writer.writerow([it.image_file,it.image_num,it.case_id,it.absent_object,tone,prompt,resp,norm,affirmed,hallucinated])
                f.flush()

    print(f"DONE! Results saved to {args.out_csv}")

if __name__ == "__main__":
    main()
