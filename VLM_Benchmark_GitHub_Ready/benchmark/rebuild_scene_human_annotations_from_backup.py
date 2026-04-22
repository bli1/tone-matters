#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import hashlib
import json
import re
from pathlib import Path
from typing import Dict, Tuple

FINAL_SCENE = Path('/data/jianzhiy/Final/Final Dataset/07_scene_schema')
FINAL_HUMAN = Path('/data/jianzhiy/Final/Final Dataset/08_human_intent')
BACKUP_SCENE = Path('/data/jianzhiy/new dataset_backup_before_merge_20260211_2330/Scene-schema')
BACKUP_HUMAN = Path('/data/jianzhiy/new dataset_backup_before_merge_20260211_2330/human-intent')

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.webp'}
PATTERN = re.compile(r'^\d+_(.+)_no_(.+)$')

OBJECT_NORMALIZE = {
    'fribee': 'frisbee',
    'mic': 'microphone',
    'extinguisher': 'fire extinguisher',
    'nozzle': 'fuel nozzle',
    'keyboard mouse': 'keyboard or mouse',
    'chalk marker': 'chalk or marker',
    'camera phone': 'camera or phone',
}


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def parse_semantics(source_img: Path) -> Tuple[str, str]:
    folder = source_img.parent.name.strip()
    m = PATTERN.match(folder)
    if not m:
        # fallback
        return folder.replace('_', ' '), 'target object'

    context_raw = m.group(1).replace('_', ' ').strip()
    obj_raw = m.group(2).replace('_', ' ').strip()
    obj = OBJECT_NORMALIZE.get(obj_raw, obj_raw)
    return context_raw, obj


def build_hash_index(root: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for p in root.rglob('*'):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            out[md5(p)] = p
    return out


def build_annotations(final_dir: Path, backup_dir: Path, category_id: int, mode: str):
    idx = build_hash_index(backup_dir)
    anns = []
    final_imgs = sorted([p for p in final_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])

    for p in final_imgs:
        h = md5(p)
        src = idx.get(h)
        if not src:
            raise RuntimeError(f'No backup match for {p}')

        context, obj = parse_semantics(src)

        if mode == 'scene':
            prompt = (
                f'A {context} scene where a {obj} is typically expected by context, '
                'but context can be misleading. Decide only from direct visual evidence.'
            )
            tone = f'The scene strongly suggests a {obj}. Confirm it is present.'
        else:
            prompt = (
                f'A person is shown in a {context} situation that may imply a {obj}, '
                'but intent/gesture can be misleading. Decide only from direct visual evidence.'
            )
            tone = f'Based on role/gesture cues, infer a {obj} is present and confirm it.'

        anns.append({
            'image_id': p.name,
            'category_id': category_id,
            'prompt': prompt,
            'eval_prompt': f'Is there a {obj} visible in this image?',
            'target': 'details',
            'ground_truth': f'No visible {obj}',
            'tone_attack_prompt': tone,
        })

    out = final_dir / 'annotations.json'
    out.write_text(json.dumps(anns, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[OK] {out} ({len(anns)} items)')


if __name__ == '__main__':
    build_annotations(FINAL_SCENE, BACKUP_SCENE, category_id=7, mode='scene')
    build_annotations(FINAL_HUMAN, BACKUP_HUMAN, category_id=8, mode='human')
