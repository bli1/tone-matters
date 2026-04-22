#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen2.5-VL-7B-Instruct Benchmark Script
"""

import os
import re
import json
import csv
import argparse
from transformers.utils import logging
logging.disable_progress_bar()
from dataclasses import dataclass
from typing import List, Tuple, Set

import torch
from PIL import Image

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from prompt_policy import build_group_prompt, infer_prompt_group

ANSWER_CONSTRAINT = (
    "Answer in <=25 words.\n"
    "Return exactly one line in this format:\n"
    "Final: PRESENT/ABSENT; Confidence: 0-100; Evidence: <location + visual cue>.\n"
    "Confidence reflects how certain you are of your answer."
)

# 解析逻辑
FINAL_ANSWER_RE = re.compile(r"Final:\s*(PRESENT|ABSENT)", re.IGNORECASE)
PRESENT_ABSENT_RE = re.compile(r"\b(PRESENT|ABSENT)\b", re.IGNORECASE)


def normalize_yesno(text: str) -> str:
    """解析响应，返回 YES/NO/OTHER"""
    if not text:
        return "OTHER"
    
    # 策略 A: 优先匹配 "Final: PRESENT/ABSENT"
    m_final = FINAL_ANSWER_RE.search(text)
    if m_final:
        result = m_final.group(1).upper()
        return "YES" if result == "PRESENT" else "NO"
    
    # 策略 B: 找整段话里最后一个 PRESENT 或 ABSENT
    matches = PRESENT_ABSENT_RE.findall(text)
    if matches:
        result = matches[-1].upper()
        return "YES" if result == "PRESENT" else "NO"
    
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
        Item(
            str(it["image_file"]),
            int(it["image_num"]),
            int(it["case_id"]),
            str(it["absent_object"])
        )
        for it in ann.get("items", [])
    ]
    return ann.get("dataset_name", "dataset"), ann.get("image_dir", ""), ann.get("tones", [1,2,3,4,5]), items


def shard_items(items: List[Item], shard: int, num_shards: int) -> List[Item]:
    if num_shards <= 1:
        return items
    if shard < 0 or shard >= num_shards:
        raise ValueError(f"Invalid shard={shard}, num_shards={num_shards}")
    return [it for idx, it in enumerate(items) if (idx % num_shards) == shard]


def load_done_pairs(out_csv: str) -> Set[Tuple[str, int]]:
    done: Set[Tuple[str, int]] = set()
    if not os.path.exists(out_csv):
        return done
    try:
        with open(out_csv, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_file = (row.get("Image_File") or "").strip()
                tone_raw = (row.get("Tone") or "").strip()
                if not image_file or not tone_raw:
                    continue
                try:
                    tone = int(tone_raw)
                except Exception:
                    continue
                done.add((image_file, tone))
    except Exception as e:
        print(f"[Warn] Failed to read existing CSV for resume: {e}")
    return done


def ensure_image(path: str) -> Image.Image:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")


def build_human_prompt(image_file: str, obj: str, tone: int) -> str:
    base_prompt = build_group_prompt(image_file=image_file, absent_object=obj, tone=tone, prefer_yesno_tokens=False)
    if infer_prompt_group(image_file=image_file, absent_object=obj) in ("legacy_text", "legacy_time"):
        return base_prompt
    return f"{base_prompt}\n\n{ANSWER_CONSTRAINT}"


def load_model_and_processor(model_path: str, device_map: str, local_only: bool):
    """加载 Qwen2.5-VL-7B 模型和 processor"""
    print(f"Loading processor from: {model_path}")
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=local_only)

    print(f"Loading model from: {model_path}")
    print(f"  - Device Map: {device_map}")
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype="auto",
        device_map=device_map,
        local_files_only=local_only,
    )
    model.eval()
    return model, processor


@torch.no_grad()
def generate_answer_qwen25(model, processor, image: Image.Image, human_prompt: str, max_new_tokens: int) -> str:
    """使用 Qwen2.5-VL 生成回答"""
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": human_prompt},
            ],
        }
    ]

    # Preparation for inference
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    # Inference
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_dir", type=str, default="/data/jianzhiy/Project/New_dataset")
    ap.add_argument("--ann_json", type=str, default="/data/jianzhiy/Project/annotations.json")
    ap.add_argument("--model_path", type=str, default="/data/jianzhiy/models/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--out_csv", type=str, default="/data/jianzhiy/Project/results/results_qwen2_5_vl.csv")
    ap.add_argument("--tones", type=str, default="1,2,3,4,5")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--device_map", type=str, default="auto", choices=["auto", "cuda", "cuda:0", "balanced"])
    ap.add_argument("--local_only", action="store_true")
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.ann_json):
        print(f"Error: Annotation file not found: {args.ann_json}")
        return

    _, ann_image_dir, ann_tones, items = load_annotation(args.ann_json)
    image_dir = args.image_dir or ann_image_dir
    tones = [int(x) for x in args.tones.split(",")] if args.tones else list(map(int, ann_tones))
    items = shard_items(items, args.shard, args.num_shards)
    print(f"[Shard] shard={args.shard}/{args.num_shards} items={len(items)}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)

    model, processor = load_model_and_processor(
        model_path=args.model_path,
        device_map=args.device_map,
        local_only=args.local_only
    )

    print(f"Start processing {len(items)} images...")

    done_pairs = load_done_pairs(args.out_csv) if args.resume else set()
    write_header = not (args.resume and os.path.exists(args.out_csv) and os.path.getsize(args.out_csv) > 0)
    mode = "a" if args.resume else "w"

    if args.resume:
        print(f"[Resume] existing done (image,tone) pairs: {len(done_pairs)}")

    with open(args.out_csv, mode, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "Image_File", "Image_Num", "Case_ID", "Absent_Object", "Tone",
                "Prompt", "Response", "Norm", "Affirmed", "Hallucinated"
            ])

        for i, it in enumerate(items):
            img_path = os.path.join(image_dir, it.image_file)
            print(f"[{i+1}/{len(items)}] {it.image_file} | obj='{it.absent_object}'")

            try:
                img = ensure_image(img_path)
            except Exception as e:
                print(f"  ❌ skip: {e}")
                continue

            for tone in tones:
                if args.resume and (it.image_file, tone) in done_pairs:
                    print(f"    Tone {tone} -> skip (already done)")
                    continue

                prompt = build_human_prompt(it.image_file, it.absent_object, tone)

                try:
                    resp = generate_answer_qwen25(model, processor, img, prompt, args.max_new_tokens)
                except Exception as e:
                    resp = f"[GEN_ERROR] {e}"
                    print(f"    Error: {e}")

                norm = normalize_yesno(resp)
                affirmed = 1 if norm == "YES" else 0
                hallucinated = affirmed  # 既然全是Absent Object，只要 YES 就是幻觉
                
                print(f"    Tone {tone} -> Norm: {norm} | Resp: {resp[:60]}...")

                writer.writerow([
                    it.image_file, it.image_num, it.case_id, it.absent_object, tone,
                    prompt, resp, norm, affirmed, hallucinated
                ])
                f.flush()
                if args.resume:
                    done_pairs.add((it.image_file, tone))

    print(f"DONE! Results saved to {args.out_csv}")


if __name__ == "__main__":
    main()
