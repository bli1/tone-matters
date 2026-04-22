#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import csv
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Type

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

from transformers import AutoTokenizer, AutoModel, GenerationConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.generation.utils import GenerationMixin
from prompt_policy import build_group_prompt, infer_prompt_group


# ============================================================
# 0) Patch A: 修复 transformers <-> InternVL2.5 tied-weights 不兼容
#    (all_tied_weights_keys 缺失导致 AttributeError)
# ============================================================
_ORIG_MARK_TIED = PreTrainedModel.mark_tied_weights_as_initialized

def _patched_mark_tied_weights_as_initialized(self: PreTrainedModel):
    if not hasattr(self, "all_tied_weights_keys"):
        try:
            keys = getattr(self, "_tied_weights_keys", None)
            if isinstance(keys, dict):
                self.all_tied_weights_keys = keys
            elif isinstance(keys, (list, tuple, set)):
                self.all_tied_weights_keys = {k: None for k in keys}
            else:
                self.all_tied_weights_keys = {}
        except Exception:
            self.all_tied_weights_keys = {}
    return _ORIG_MARK_TIED(self)

PreTrainedModel.mark_tied_weights_as_initialized = _patched_mark_tied_weights_as_initialized


# ============================================================
# 0) Patch B: 修复 transformers v4.50+ 之后 remote code 模型没 generate()
#    方案：给所有具备 prepare_inputs_for_generation 但缺 generate 的 module 动态混入 GenerationMixin
# ============================================================
_PATCHED_CLASS_CACHE: Dict[Type, Type] = {}

def _ensure_generation_mixin(module: torch.nn.Module):
    """If module has prepare_inputs_for_generation but no generate(), dynamically mix in GenerationMixin."""
    if hasattr(module, "generate"):
        return
    if not hasattr(module, "prepare_inputs_for_generation"):
        return

    cls = module.__class__
    if cls in _PATCHED_CLASS_CACHE:
        module.__class__ = _PATCHED_CLASS_CACHE[cls]
        return

    Patched = type(f"{cls.__name__}WithGeneration", (cls, GenerationMixin), {})
    _PATCHED_CLASS_CACHE[cls] = Patched
    module.__class__ = Patched

def patch_generation_everywhere(root: torch.nn.Module):
    # patch root itself
    _ensure_generation_mixin(root)
    # patch all submodules (InternLM2ForCausalLM 通常在这里面)
    for m in root.modules():
        _ensure_generation_mixin(m)


def patch_generation_config_everywhere(root: torch.nn.Module):
    """Ensure modules used by chat/generate have a generation_config."""
    mods = [root] + list(root.modules())
    for m in mods:
        if not (hasattr(m, "generate") or hasattr(m, "prepare_inputs_for_generation")):
            continue
        if hasattr(m, "generation_config") and getattr(m, "generation_config", None) is not None:
            continue
        try:
            if hasattr(m, "config") and m.config is not None:
                m.generation_config = GenerationConfig.from_model_config(m.config)
            else:
                m.generation_config = GenerationConfig()
        except Exception:
            try:
                m.generation_config = GenerationConfig()
            except Exception:
                pass


# ============================================================
# 1) Patch C: 强制 default device = CPU（防 meta 污染）
# ============================================================
def force_default_device_cpu():
    if hasattr(torch, "set_default_device"):
        try:
            torch.set_default_device("cpu")
        except Exception:
            pass

def print_default_device(tag=""):
    try:
        if hasattr(torch, "get_default_device"):
            print(f"[Debug]{tag} torch.get_default_device() = {torch.get_default_device()}", flush=True)
    except Exception:
        pass

force_default_device_cpu()


# ============================================================
# 2) InternVL2.5 图像预处理（必须）
# ============================================================
def build_transform(input_size=448):
    MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
    ])

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = orig_width * orig_height

    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio

    target_width = image_size * best_ratio[0]
    target_height = image_size * best_ratio[1]
    blocks = best_ratio[0] * best_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    grid_w = target_width // image_size

    for i in range(blocks):
        box = (
            (i % grid_w) * image_size,
            (i // grid_w) * image_size,
            ((i % grid_w) + 1) * image_size,
            ((i // grid_w) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))

    if use_thumbnail and len(processed_images) > 1:
        processed_images.append(image.resize((image_size, image_size)))

    return processed_images

def process_image_to_pixel_values(image: Image.Image, max_num=12):
    transform = build_transform(input_size=448)
    images = dynamic_preprocess(image, image_size=448, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    return torch.stack(pixel_values)


# ============================================================
# 3) Prompt + 规则判定
# ============================================================
ANSWER_CONSTRAINT = (
    "Answer in <=25 words.\n"
    "Return exactly one line in this format:\n"
    "Final: PRESENT/ABSENT; Confidence: 0-100; Evidence: <location + visual cue>.\n"
    "Confidence reflects how certain you are of your answer."
)

FINAL_ANSWER_RE = re.compile(r"Final:\s*(PRESENT|ABSENT)", re.IGNORECASE)
PRESENT_ABSENT_RE = re.compile(r"\b(PRESENT|ABSENT)\b", re.IGNORECASE)
YESNO_RE = re.compile(r"\b(YES|NO)\b", re.IGNORECASE)
AFFIRM_RE = re.compile(r"\b(present|visible|there is|there's|i see|appears|shown|can be seen|in the image|yes)\b", re.IGNORECASE)
NEGATE_RE = re.compile(r"\b(absent|not present|no sign|not visible|cannot see|can't see|does not contain|doesn't contain|no visible|no)\b", re.IGNORECASE)
UNCERTAIN_RE = re.compile(r"\b(uncertain|not sure|hard to tell|maybe|possibly|cannot confirm|can't confirm|unclear)\b", re.IGNORECASE)

def truncate_words(text: str, max_words: int) -> str:
    if not text:
        return ""
    words = text.strip().split()
    if max_words <= 0 or len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]).strip()

def normalize_yesno(resp: str) -> str:
    if not resp:
        return "OTHER"
    t = resp.strip()

    m_final = FINAL_ANSWER_RE.search(t)
    if m_final:
        return "YES" if m_final.group(1).upper() == "PRESENT" else "NO"

    pa = PRESENT_ABSENT_RE.findall(t)
    if pa:
        return "YES" if pa[-1].upper() == "PRESENT" else "NO"

    m = YESNO_RE.search(t)
    if m:
        return m.group(1).upper()

    if UNCERTAIN_RE.search(t):
        return "OTHER"
    if NEGATE_RE.search(t):
        return "NO"
    if AFFIRM_RE.search(t):
        return "YES"
    return "OTHER"


def heuristic_affirmed(resp: str, norm: str) -> int:
    if norm == "YES":
        return 1
    if norm == "NO":
        return 0
    if not resp:
        return 0
    return 0


# ============================================================
# 4) 加载 InternVL2.5（含 linspace 强制 CPU 防 meta）
# ============================================================
def load_internvl25_chat(model_path: str, dtype: str, device: str, local_only: bool, trust_remote_code: bool):
    force_default_device_cpu()
    print_default_device(" before load")

    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype.lower()]
    print(f"[Load] InternVL2.5 from {model_path} dtype={dtype} device={device} local_only={local_only}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        local_files_only=local_only,
    )
    if not hasattr(tokenizer, "clean_up_tokenization"):
        tokenizer.clean_up_tokenization = lambda text: text

    # 防止 remote code 在构造 vision encoder 时走到 meta device
    _orig_linspace = torch.linspace
    def _linspace_force_cpu(*args, **kwargs):
        if "device" not in kwargs or kwargs["device"] is None:
            kwargs["device"] = "cpu"
        return _orig_linspace(*args, **kwargs)
    torch.linspace = _linspace_force_cpu

    try:
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            local_files_only=local_only,
            low_cpu_mem_usage=False,
            device_map=None,  # 先全 CPU/默认，再 .to(device)
        )
    finally:
        torch.linspace = _orig_linspace

    # ⭐关键：补上 generate（包括内部 language_model）
    patch_generation_everywhere(model)
    patch_generation_config_everywhere(model)

    model.eval()
    model.to(device)

    # 再 patch 一次，防止 .to 后有 lazy module
    patch_generation_everywhere(model)
    patch_generation_config_everywhere(model)

    print_default_device(" after load")
    return model, tokenizer, torch_dtype


# ============================================================
# 5) 推理（InternVL2.5：model.chat + pixel_values）
# ============================================================
@torch.no_grad()
def internvl25_generate(
    model,
    tokenizer,
    image: Image.Image,
    question: str,
    max_new_tokens: int,
    max_num_patches: int,
) -> str:
    candidates = []
    for n in (max_num_patches, 8, 6, 4, 2):
        if n > 0 and n not in candidates:
            candidates.append(n)

    last_err = None
    for patch_num in candidates:
        try:
            pixel_values = process_image_to_pixel_values(image, max_num=patch_num)
            pixel_values = pixel_values.to(device=model.device, dtype=model.dtype)

            generation_config = {
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "num_beams": 1,
            }

            resp = model.chat(
                tokenizer=tokenizer,
                pixel_values=pixel_values,
                question=question,
                generation_config=generation_config,
            )
            return str(resp).strip()
        except Exception as e:
            last_err = e
            if "out of memory" in str(e).lower():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            return f"[GEN_ERROR] {type(e).__name__}: {e}"

    return f"[GEN_ERROR] OutOfMemoryError: {last_err}"


# ============================================================
# 6) 数据读取
# ============================================================
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
            int(it.get("image_num", 0)),
            int(it.get("case_id", 0)),
            str(it["absent_object"]),
        )
        for it in ann.get("items", [])
    ]
    return ann.get("dataset_name", "dataset"), ann.get("image_dir", ""), ann.get("tones", [1,2,3,4,5]), items

def ensure_image(path: str) -> Image.Image:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")


# ============================================================
# 7) Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_dir", type=str, required=True)
    ap.add_argument("--ann_json", type=str, required=True)
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--out_csv", type=str, required=True)

    ap.add_argument("--tones", type=str, default="1,2,3,4,5")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--max_words", type=int, default=20)
    ap.add_argument("--max_num_patches", type=int, default=6)
    ap.add_argument("--dtype", type=str, default="bf16")
    ap.add_argument("--device", type=str, default="cuda")

    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)

    ap.add_argument("--local_only", action="store_true")
    ap.add_argument("--trust_remote_code", action="store_true")

    args = ap.parse_args()

    _, ann_image_dir, ann_tones, items = load_annotation(args.ann_json)
    image_dir = args.image_dir or ann_image_dir
    tones = [int(x) for x in args.tones.split(",")] if args.tones else list(map(int, ann_tones))

    if args.num_shards > 1:
        items = [it for idx, it in enumerate(items) if (idx % args.num_shards) == args.shard]

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)

    model, tokenizer, _dtype = load_internvl25_chat(
        model_path=args.model_path,
        dtype=args.dtype,
        device=args.device,
        local_only=args.local_only,
        trust_remote_code=args.trust_remote_code,
    )

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "Image_File", "Image_Num", "Case_ID", "Absent_Object", "Tone",
            "Prompt", "Response", "Norm", "Affirmed", "Hallucinated",
            "Raw_Answer", "Trimmed_Answer", "Rule", "Final", "Error"
        ])

        print(f"[Run] items={len(items)} tones={tones}", flush=True)

        for i, it in enumerate(items):
            img_path = os.path.join(image_dir, it.image_file)
            print(f"[{i+1}/{len(items)}] {it.image_file} | obj='{it.absent_object}'", flush=True)

            try:
                img = ensure_image(img_path)
            except Exception as e:
                print(f"  [IMG_ERROR] {e}", flush=True)
                continue

            for tone in tones:
                base_q = build_group_prompt(
                    image_file=it.image_file,
                    absent_object=it.absent_object,
                    tone=tone,
                    prefer_yesno_tokens=False,
                )
                if infer_prompt_group(image_file=it.image_file, absent_object=it.absent_object) in ("legacy_text", "legacy_time"):
                    prompt = base_q
                else:
                    prompt = f"{base_q}\n\n{ANSWER_CONSTRAINT}"

                raw = internvl25_generate(
                    model=model,
                    tokenizer=tokenizer,
                    image=img,
                    question=prompt,
                    max_new_tokens=args.max_new_tokens,
                    max_num_patches=args.max_num_patches,
                )
                trimmed = truncate_words(raw, args.max_words)
                response = trimmed if trimmed else raw

                if raw.startswith("[GEN_ERROR]"):
                    norm = "OTHER"
                    affirmed = 0
                    hallucinated = 0
                    rule = "OTHER"
                    final = "ERROR"
                    err = raw
                else:
                    norm = normalize_yesno(response)
                    affirmed = heuristic_affirmed(response, norm)
                    hallucinated = affirmed
                    rule = norm
                    final = norm
                    err = ""

                print(f"    Tone {tone}: final={final} norm={norm}", flush=True)

                w.writerow([
                    it.image_file, it.image_num, it.case_id, it.absent_object, tone,
                    prompt, response, norm, affirmed, hallucinated,
                    raw, trimmed, rule, final, err
                ])
                f.flush()

    print(f"[Done] Saved: {args.out_csv}", flush=True)

if __name__ == "__main__":
    main()
