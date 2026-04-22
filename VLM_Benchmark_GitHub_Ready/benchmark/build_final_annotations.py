#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# Default mapping for your 8 final categories.
DEFAULT_ABSENT_OBJECT_MAP = {
    "01_text_blur": "legible badge text",
    "02_text_gibberish": "legible badge text",
    "03_text_blank": "legible badge text",
    "04_time_fix": "determinable clock time",
    "05_analog_decoy": "determinable clock time",
    "06_digital_decoy": "determinable clock time",
    "07_scene_schema": "target object",
    "08_human_intent": "target object",
}

EVAL_PROMPT_RE = re.compile(r"^Is there a (.+) visible in this image\?$", re.IGNORECASE)


def load_folder_absent_object_map(image_dir: Path):
    """Load per-image absent object from folder-level annotations.json when available."""
    out = {}
    for folder in ("07_scene_schema", "08_human_intent"):
        ann_path = image_dir / folder / "annotations.json"
        if not ann_path.exists():
            continue
        try:
            arr = json.loads(ann_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        by_image = {}
        for it in arr:
            image_id = str(it.get("image_id", "")).strip()
            eval_prompt = str(it.get("eval_prompt", "")).strip()
            if not image_id:
                continue
            m = EVAL_PROMPT_RE.match(eval_prompt)
            if m:
                by_image[image_id] = m.group(1).strip()
        out[folder] = by_image
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_dir", type=str, default="/data/jianzhiy/Final/Final Dataset")
    ap.add_argument("--out_json", type=str, default="/data/jianzhiy/Final/final_annotations_800.json")
    ap.add_argument("--dataset_name", type=str, default="Final_dataset_800")
    ap.add_argument("--tones", type=str, default="1,2,3,4,5")
    args = ap.parse_args()

    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    folder_absent_object_map = load_folder_absent_object_map(image_dir)

    items = []
    idx = 1

    paths = sorted(
        [p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
        key=lambda p: str(p.relative_to(image_dir))
    )

    for p in paths:
        rel = p.relative_to(image_dir).as_posix()
        folder = rel.split("/", 1)[0]
        absent_object = DEFAULT_ABSENT_OBJECT_MAP.get(folder, "target object")
        per_folder = folder_absent_object_map.get(folder, {})
        absent_object = per_folder.get(p.name, absent_object)

        items.append({
            "image_file": rel,
            "image_num": idx,
            "case_id": idx,
            "absent_object": absent_object,
        })
        idx += 1

    tones = [int(x) for x in args.tones.split(",") if x.strip()]

    out = {
        "dataset_name": args.dataset_name,
        "image_dir": str(image_dir),
        "tones": tones,
        "items": items,
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] wrote {out_path}")
    print(f"[OK] total images: {len(items)}")


if __name__ == "__main__":
    main()
