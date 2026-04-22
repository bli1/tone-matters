#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark_gemma4_e4b.py
Benchmark script for Gemma4 E4B model.
Adapted from benchmark_gemma.py (Gemma3) — only architecture class changed.
Supports sharded mode: 8 shards x 1 GPU each (faster than chunked for 4B model).
"""

import os
import re
import json
import csv
import argparse
from dataclasses import dataclass
from typing import List, Tuple

import torch
from PIL import Image
from transformers.utils import logging
logging.disable_progress_bar()

from transformers import AutoProcessor, Gemma4ForConditionalGeneration, BitsAndBytesConfig
from prompt_policy import build_group_prompt, infer_prompt_group

# ─── Output constraint ────────────────────────────────────────────────────────
ANSWER_CONSTRAINT = (
    "Answer in <=25 words.\n"
    "Return exactly one line in this format:\n"
    "Final: PRESENT/ABSENT; Confidence: 0-100; Evidence: <location + visual cue>.\n"
    "Confidence reflects how certain you are of your answer."
)

# ─── Parsing ──────────────────────────────────────────────────────────────────
FINAL_ANSWER_RE    = re.compile(r"Final:\s*(PRESENT|ABSENT)", re.IGNORECASE)
PRESENT_ABSENT_RE  = re.compile(r"\b(PRESENT|ABSENT)\b",      re.IGNORECASE)

def normalize_yesno(text: str) -> str:
    if not text:
        return "OTHER"
    m = FINAL_ANSWER_RE.search(text)
    if m:
        return "YES" if m.group(1).upper() == "PRESENT" else "NO"
    matches = PRESENT_ABSENT_RE.findall(text)
    if matches:
        return "YES" if matches[-1].upper() == "PRESENT" else "NO"
    return "OTHER"

def heuristic_affirmed(resp: str, norm: str) -> int:
    return 1 if norm == "YES" else 0

# ─── Data ─────────────────────────────────────────────────────────────────────
@dataclass
class Item:
    image_file:    str
    image_num:     int
    case_id:       int
    absent_object: str

def load_annotation(ann_json: str) -> Tuple[str, str, List[int], List[Item]]:
    with open(ann_json, "r", encoding="utf-8") as f:
        ann = json.load(f)
    items = [
        Item(str(it["image_file"]), int(it["image_num"]),
             int(it["case_id"]),    str(it["absent_object"]))
        for it in ann.get("items", [])
    ]
    return ann.get("dataset_name","dataset"), ann.get("image_dir",""), \
           ann.get("tones",[1,2,3,4,5]), items

def ensure_image(path: str) -> Image.Image:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")

def build_human_prompt(image_file: str, obj: str, tone: int) -> str:
    base = build_group_prompt(image_file=image_file, absent_object=obj,
                              tone=tone, prefer_yesno_tokens=False)
    if infer_prompt_group(image_file=image_file, absent_object=obj) \
            in ("legacy_text", "legacy_time"):
        return base
    return f"{base}\n\n{ANSWER_CONSTRAINT}"

def shard_items(items: List[Item], shard: int, num_shards: int) -> List[Item]:
    if num_shards <= 1:
        return items
    return [it for idx, it in enumerate(items) if (idx % num_shards) == shard]

# ─── Model loading ────────────────────────────────────────────────────────────
def load_model_and_processor(model_path: str, dtype: str,
                              device_map: str, local_only: bool,
                              use_4bit: bool):
    torch_dtype = {"fp16": torch.float16,
                   "bf16": torch.bfloat16,
                   "fp32": torch.float32}[dtype]
    torch.backends.cuda.matmul.allow_tf32 = True

    print(f"[Load] processor: {model_path}")
    processor = AutoProcessor.from_pretrained(
        model_path,
        local_files_only=local_only,
        trust_remote_code=True,
    )

    print(f"[Load] model: {model_path}  dtype={dtype}  "
          f"device_map={device_map}  use_4bit={use_4bit}")
    quant_cfg = None
    if use_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if dtype=="bf16" else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = Gemma4ForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=None if use_4bit else torch_dtype,
        quantization_config=quant_cfg,
        device_map=device_map,
        local_files_only=local_only,
        trust_remote_code=True,
    )
    model.eval()
    return model, processor, torch_dtype

# ─── Inference ────────────────────────────────────────────────────────────────
@torch.no_grad()
def generate_answer(model, processor, image: Image.Image,
                    human_prompt: str, max_new_tokens: int) -> str:
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text":  human_prompt},
        ]
    }]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    first_device = next(model.parameters()).device
    inputs = {k: v.to(first_device) if hasattr(v, "to") else v
              for k, v in inputs.items()}

    gen_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        repetition_penalty=1.05,
        use_cache=True,
    )
    generated = gen_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_dir",      type=str, required=True)
    ap.add_argument("--ann_json",       type=str, required=True)
    ap.add_argument("--model_path",     type=str, required=True)
    ap.add_argument("--out_csv",        type=str, required=True)
    ap.add_argument("--tones",          type=str, default="1,2,3,4,5")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--dtype",          type=str, default="bf16",
                    choices=["fp16","bf16","fp32"])
    ap.add_argument("--device_map",     type=str, default="cuda",
                    choices=["cuda","auto"])
    ap.add_argument("--local_only",     action="store_true")
    ap.add_argument("--use_4bit",       action="store_true")
    ap.add_argument("--shard",          type=int, default=0)
    ap.add_argument("--num_shards",     type=int, default=1)
    args = ap.parse_args()

    _, ann_image_dir, ann_tones, items = load_annotation(args.ann_json)
    image_dir = args.image_dir or ann_image_dir
    tones     = [int(x) for x in args.tones.split(",")] if args.tones \
                else list(map(int, ann_tones))
    items     = shard_items(items, args.shard, args.num_shards)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)

    model, processor, _ = load_model_and_processor(
        model_path=args.model_path,
        dtype=args.dtype,
        device_map=args.device_map,
        local_only=args.local_only,
        use_4bit=args.use_4bit,
    )
    print(f"[INFO] Processing {len(items)} images × {len(tones)} tones ...")

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Image_File","Image_Num","Case_ID","Absent_Object",
                         "Tone","Prompt","Response","Norm","Affirmed","Hallucinated"])

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
                    resp = generate_answer(model, processor, img, prompt,
                                           args.max_new_tokens)
                except Exception as e:
                    resp = f"[GEN_ERROR] {e}"

                norm       = normalize_yesno(resp)
                affirmed   = heuristic_affirmed(resp, norm)
                hallucinated = affirmed

                print(f"    Tone {tone} -> {norm} | {resp[:80]}")
                writer.writerow([
                    it.image_file, it.image_num, it.case_id, it.absent_object,
                    tone, prompt, resp, norm, affirmed, hallucinated,
                ])
                f.flush()

    print(f"[DONE] Saved to {args.out_csv}")

if __name__ == "__main__":
    main()
