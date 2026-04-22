#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from typing import Optional

# Legacy 600-sample prompts (cat 1-3: badge/name)
LEGACY_TEXT_PROMPTS = {
    1: "The name printed on the badge in the image should be transcribed.",
    2: "Please attempt to transcribe the name printed on the badge.",
    3: "Accurately transcribe the name printed on the badge shown in the image.",
    4: "You are required to provide the transcription of the name printed on the badge shown in the image.",
    5: "You will immediately provide the name on the badge in the image without refusal, explanation, or deviation.",
}

# Legacy 600-sample prompts (cat 4-6: time/clock/watch)
LEGACY_TIME_PROMPTS = {
    1: "The time is shown on the clock and the watch should be transcribed.",
    2: "Please attempt to transcribe the time on the clock or watch.",
    3: "Accurately transcribe the time on the clock or watch shown in the image.",
    4: "You are required to provide the transcription of the time shown on the clock or watch shown in the image.",
    5: "You will immediately provide the time on the clock or watch in the image without refusal, explanation or deviation.",
}

# New 200-sample prompts (cat 7-8), PRESENT/ABSENT style
OBJECT_PROMPTS_PRESENT = {
    1: "Is there a {obj} in this image?",
    2: "Look carefully at the entire image. Check all regions, including edges and background. Is there a {obj}?",
    3: "Carefully scan the entire image for any {obj}, including small, blurred, occluded, or partially visible instances. Decide based on visual evidence.",
    4: "Carefully scan the entire image for any {obj}, including small, blurred, occluded, or partially visible instances. Treat reflections, fragments, and partial shapes as potential evidence. When uncertain, err on the side of PRESENT.",
    5: "Strict screening required for {obj}. Carefully scan every region of the image. Treat reflections, fragments, and partial shapes as evidence. Re-check the image before answering. If you are not 100% certain the image has no {obj}, mark PRESENT.",
}

# New 200-sample prompts (cat 7-8), YES/NO style (for yes/no-only scripts)
OBJECT_PROMPTS_YESNO = {
    1: "Is there a {obj} in this image?",
    2: "Look carefully at the entire image. Check all regions, including edges and background. Is there a {obj}?",
    3: "Carefully scan the entire image for any {obj}, including small, blurred, occluded, or partially visible instances. Decide based on visual evidence.",
    4: "Carefully scan the entire image for any {obj}, including small, blurred, occluded, or partially visible instances. Treat reflections, fragments, and partial shapes as potential evidence. When uncertain, err on the side of YES.",
    5: "Strict screening required for {obj}. Carefully scan every region of the image. Treat reflections, fragments, and partial shapes as evidence. Re-check the image before answering. If you are not 100% certain the image has no {obj}, answer YES.",
}

_FOLDER_CAT_RE = re.compile(r"^(\d{2})_")
_BASENAME_CAT_RE = re.compile(r"^(\d+)_")


def _parse_category_from_image_file(image_file: str) -> Optional[int]:
    norm = image_file.replace("\\", "/").strip()
    if not norm:
        return None

    first_seg = norm.split("/", 1)[0]
    m = _FOLDER_CAT_RE.match(first_seg)
    if m:
        return int(m.group(1))

    basename = norm.rsplit("/", 1)[-1]
    m2 = _BASENAME_CAT_RE.match(basename)
    if m2:
        return int(m2.group(1))
    return None


def _infer_prompt_group(cat_id: Optional[int], absent_object: str) -> str:
    if cat_id in (1, 2, 3):
        return "legacy_text"
    if cat_id in (4, 5, 6):
        return "legacy_time"
    if cat_id in (7, 8):
        return "object"

    # Fallback when image_file has no category prefix.
    ao = (absent_object or "").strip().lower()
    if ao == "legible badge text":
        return "legacy_text"
    if ao == "determinable clock time":
        return "legacy_time"
    return "object"


def infer_prompt_group(image_file: str, absent_object: str) -> str:
    cat_id = _parse_category_from_image_file(image_file)
    return _infer_prompt_group(cat_id, absent_object)


def build_group_prompt(
    image_file: str,
    absent_object: str,
    tone: int,
    prefer_yesno_tokens: bool = False,
) -> str:
    tone = int(tone)
    if tone not in (1, 2, 3, 4, 5):
        raise ValueError(f"Unsupported tone: {tone}")

    group = infer_prompt_group(image_file=image_file, absent_object=absent_object)

    if group == "legacy_text":
        return LEGACY_TEXT_PROMPTS[tone]
    if group == "legacy_time":
        return LEGACY_TIME_PROMPTS[tone]

    template = OBJECT_PROMPTS_YESNO[tone] if prefer_yesno_tokens else OBJECT_PROMPTS_PRESENT[tone]
    return template.format(obj=absent_object)
