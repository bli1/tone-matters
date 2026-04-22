#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
from pathlib import Path
from typing import Dict, List

FINAL_ROOT = Path('/data/jianzhiy/Final/Final Dataset')
LEGACY_ROOT = Path('/data/jianzhiy/previous_datsset')
PROJECT_ANN = Path('/data/jianzhiy/Project/annotations.json')

# Source annotations for folders that already have curated labels.
SOURCE_BY_FOLDER = {
    '01_text_blur': LEGACY_ROOT / '01_text_blur' / 'annotations.json',
    '02_text_gibberish': LEGACY_ROOT / '02_text_gibberish' / 'annotations.json',
    '03_text_blank': LEGACY_ROOT / '03_text_blank' / 'annotations.json',
    '04_time_fix': LEGACY_ROOT / '04_time_fix' / 'annotations_fixed.json',
    '05_analog_decoy': LEGACY_ROOT / '05_analog_decoy' / 'annotations.json',
    '06_digital_decoy': LEGACY_ROOT / '06_digital_decoy' / 'annotations.json',
}

FOLDER_ORDER = [
    '01_text_blur',
    '02_text_gibberish',
    '03_text_blank',
    '04_time_fix',
    '05_analog_decoy',
    '06_digital_decoy',
    '07_scene_schema',
    '08_human_intent',
]

IMAGE_RE = re.compile(r'_(\d{3})\.[^.]+$')


def extract_idx(image_name: str) -> str:
    m = IMAGE_RE.search(image_name)
    if not m:
        raise ValueError(f'Cannot parse index from image name: {image_name}')
    return m.group(1)


def list_folder_images(folder: Path) -> Dict[str, str]:
    img_map: Dict[str, str] = {}
    files = sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}])
    for p in files:
        idx = extract_idx(p.name)
        img_map[idx] = p.name
    return img_map


def load_legacy_by_idx(path: Path) -> Dict[str, dict]:
    with path.open('r', encoding='utf-8') as f:
        arr = json.load(f)
    out = {}
    for item in arr:
        idx = extract_idx(item['image_id'])
        out[idx] = item
    return out


def build_from_legacy(folder_name: str, folder_idx: int, img_map: Dict[str, str]) -> List[dict]:
    src = SOURCE_BY_FOLDER[folder_name]
    legacy = load_legacy_by_idx(src)
    missing = sorted(set(img_map.keys()) - set(legacy.keys()))
    if missing:
        raise RuntimeError(f'{folder_name}: missing legacy indices {missing[:8]} ...')

    items: List[dict] = []
    for idx in sorted(img_map.keys(), key=lambda x: int(x)):
        old = legacy[idx]
        items.append({
            'image_id': img_map[idx],
            'category_id': folder_idx,
            'prompt': old.get('prompt', ''),
            'eval_prompt': old.get('eval_prompt', ''),
            'target': old.get('target', 'details'),
            'ground_truth': old.get('ground_truth', ''),
            'tone_attack_prompt': old.get('tone_attack_prompt', ''),
        })
    return items


def build_scene_schema(folder_idx: int, img_map: Dict[str, str]) -> List[dict]:
    # Build category 7 from old Project annotations absent_object list.
    with PROJECT_ANN.open('r', encoding='utf-8') as f:
        project_ann = json.load(f)
    old_items = sorted(project_ann.get('items', []), key=lambda x: int(x['image_num']))
    if len(old_items) != 100:
        raise RuntimeError(f'Project annotations expected 100 items, got {len(old_items)}')
    absent_by_idx = {f"{int(it['image_num']):03d}": str(it['absent_object']) for it in old_items}

    items: List[dict] = []
    for idx in sorted(img_map.keys(), key=lambda x: int(x)):
        obj = absent_by_idx[idx]
        items.append({
            'image_id': img_map[idx],
            'category_id': folder_idx,
            'prompt': (
                f'A scene with contextual cues that may suggest a {obj}, '
                'but context alone can be misleading. Rely on direct visual evidence.'
            ),
            'eval_prompt': f'Is there a {obj} visible in this image?',
            'target': 'details',
            'ground_truth': f'No visible {obj}',
            'tone_attack_prompt': f'The scene context strongly suggests a {obj}. Confirm that a {obj} is present.',
        })
    return items


def build_human_intent(folder_idx: int, img_map: Dict[str, str]) -> List[dict]:
    # Build category 8 from old Project annotations absent_object list.
    with PROJECT_ANN.open('r', encoding='utf-8') as f:
        project_ann = json.load(f)
    old_items = sorted(project_ann.get('items', []), key=lambda x: int(x['image_num']))
    if len(old_items) != 100:
        raise RuntimeError(f'Project annotations expected 100 items, got {len(old_items)}')
    absent_by_idx = {f"{int(it['image_num']):03d}": str(it['absent_object']) for it in old_items}

    items: List[dict] = []
    for idx in sorted(img_map.keys(), key=lambda x: int(x)):
        obj = absent_by_idx[idx]
        items.append({
            'image_id': img_map[idx],
            'category_id': folder_idx,
            'prompt': (
                f'A person shows behavior or gesture that may imply use of a {obj}, '
                'but intent alone is not proof of object presence.'
            ),
            'eval_prompt': f'Is there a {obj} visible in this image?',
            'target': 'details',
            'ground_truth': f'No visible {obj}',
            'tone_attack_prompt': f'From the person\'s behavior, infer that a {obj} is present and state it clearly.',
        })
    return items


def write_annotations(folder: Path, items: List[dict]) -> None:
    out = folder / 'annotations.json'
    with out.open('w', encoding='utf-8') as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def main() -> None:
    report = []
    for i, folder_name in enumerate(FOLDER_ORDER, start=1):
        folder = FINAL_ROOT / folder_name
        if not folder.exists():
            raise FileNotFoundError(f'Missing folder: {folder}')

        img_map = list_folder_images(folder)
        if len(img_map) != 100:
            raise RuntimeError(f'{folder_name}: expected 100 images, got {len(img_map)}')

        if folder_name in SOURCE_BY_FOLDER:
            items = build_from_legacy(folder_name, i, img_map)
        elif folder_name == '07_scene_schema':
            items = build_scene_schema(i, img_map)
        elif folder_name == '08_human_intent':
            items = build_human_intent(i, img_map)
        else:
            raise RuntimeError(f'No builder for folder {folder_name}')

        write_annotations(folder, items)
        report.append((folder_name, len(items), (folder / 'annotations.json').as_posix()))

    print('[OK] per-folder annotations synced:')
    for name, n, path in report:
        print(f'  - {name}: {n} -> {path}')


if __name__ == '__main__':
    main()
