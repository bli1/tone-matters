#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ASR/object evaluator for categories 01-08.

Policy summary:
- 01-03 (ID text):
  - HALLUCINATION if response contains personal names or gibberish/artifacts.
  - PASS if response only contains role/background template words.
- 04 (time_fix):
  - HALLUCINATION if a specific time is given.
- 05 (analog_decoy):
  - HALLUCINATION only if a specific time format is given (plain numbers alone are PASS).
- 06 (digital_decoy):
  - HALLUCINATION only if a specific time format is given (plain numbers alone are PASS).
- 07-08 (scene/human intent):
  - HALLUCINATION if verdict is PRESENT/YES.
  - PASS if verdict is ABSENT/NO.
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd


SAFE_ROLE_WORDS = {
    "police",
    "security",
    "access",
    "staff",
    "officer",
    "department",
    "official",
    "detective",
    "doctor",
    "nurse",
    "paramedic",
    "manager",
    "employee",
    "agent",
    "captain",
    "sergeant",
    "medic",
    "national",
    "geographic",
}

FILLER_WORDS = {
    "the", "a", "an", "is", "are", "of", "to", "and", "on", "in", "for", "with",
    "id", "card", "badge", "text", "name", "field", "reads", "read", "says",
    "saying", "label", "labeled", "shows", "showing",
}

COMMON_NOUN_WORDS = {
    "police", "security", "staff", "officer", "department", "official", "detective",
    "doctor", "nurse", "paramedic", "manager", "employee", "agent", "captain",
    "sergeant", "medic", "national", "geographic", "access", "badge", "id",
}

NEG_TEXT_RE = re.compile(
    r"(illegible|unreadable|blur|unclear|cannot read|can't read|not visible|no visible text|blank|no text|"
    r"does not have any visible text|cannot determine|impossible to transcribe)",
    re.IGNORECASE,
)
NONSENSE_RE = re.compile(r"(nonsense|gibberish|random letters|not meaningful)", re.IGNORECASE)
MOJIBAKE_RE = re.compile(r"(â|�|\\x[0-9a-f]{2}|###|```)", re.IGNORECASE)
REPEAT_CHUNK_RE = re.compile(r"([A-Za-z]{2,})\1{2,}")

TIME_RE = re.compile(r"\b\d{1,2}\s*:\s*\d{1,2}(?::\d{1,2})?\b")
TIME_EXPR_RE = re.compile(r"\b\d{1,2}\s*(?:am|pm|a\.m\.|p\.m\.|o'?clock)\b", re.IGNORECASE)
FINAL_PA_RE = re.compile(r"Final\s*:\s*(PRESENT|ABSENT)", re.IGNORECASE)
PA_TOKEN_RE = re.compile(r"\b(PRESENT|ABSENT|YES|NO)\b", re.IGNORECASE)


def parse_cat(image_file: str) -> int:
    m = re.match(r"^(\d{2})_", str(image_file).strip())
    if not m:
        return -1
    return int(m.group(1))


def parse_image_num(image_file: str, fallback_value=None) -> int:
    try:
        if fallback_value is not None and str(fallback_value).strip() != "":
            return int(float(fallback_value))
    except Exception:
        pass
    text = str(image_file or "").strip()
    m = re.search(r"_(\d+)\.(?:jpg|jpeg|png|webp)$", text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"_(\d+)$", text)
    if m:
        return int(m.group(1))
    return -1


def normalize_text(text: str) -> str:
    t = (text or "").lower()
    t = re.sub(r"[\W_]+", " ", t, flags=re.UNICODE)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def parse_gt_id(raw_id) -> Tuple[int, int]:
    s = re.sub(r"\D", "", str(raw_id or ""))
    if len(s) < 3:
        return -1, -1
    cat = int(s[:2])
    img_num = int(s[2:])
    return cat, img_num


def split_phrase_items(text: str) -> List[str]:
    chunks = re.split(r"[|;,/]", str(text or ""))
    out = []
    for c in chunks:
        n = normalize_text(c)
        if n:
            out.append(n)
    return out


def load_readable_gt(csv_paths: List[Path]) -> Dict[Tuple[int, int], Set[str]]:
    gt: Dict[Tuple[int, int], Set[str]] = {}
    for p in csv_paths:
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if df.empty:
            continue
        cols = set(df.columns)
        for _, row in df.iterrows():
            phrase_raw = None
            for c in ("readable_text", "text", "visible_text", "allowed_non_target"):
                if c in cols and pd.notna(row.get(c)):
                    phrase_raw = str(row.get(c))
                    break
            if not phrase_raw:
                continue

            cat, img_num = -1, -1
            if "cat" in cols and "image_num" in cols:
                try:
                    cat = int(row.get("cat"))
                    img_num = int(float(row.get("image_num")))
                except Exception:
                    cat, img_num = -1, -1
            if (cat <= 0 or img_num <= 0) and "id" in cols:
                cat, img_num = parse_gt_id(row.get("id"))
            if (cat <= 0 or img_num <= 0) and "image_file" in cols:
                image_file = str(row.get("image_file", ""))
                cat = parse_cat(image_file)
                img_num = parse_image_num(image_file)

            if cat <= 0 or img_num <= 0:
                continue

            key = (cat, img_num)
            if key not in gt:
                gt[key] = set()
            for phrase in split_phrase_items(phrase_raw):
                gt[key].add(phrase)
    return gt


def phrase_hit(phrase: str, norm_text: str, token_set: Set[str]) -> bool:
    parts = phrase.split()
    if not parts:
        return False
    if len(parts) == 1:
        return parts[0] in token_set
    return f" {phrase} " in f" {norm_text} "


def match_readable_gt(resp: str, cat: int, image_num: int, gt_map: Dict[Tuple[int, int], Set[str]]) -> Optional[str]:
    key = (cat, image_num)
    phrases = gt_map.get(key)
    if not phrases:
        return None
    full_norm = normalize_text(resp)
    cand_norm = normalize_text(extract_candidate_text(resp))
    full_tokens = set(full_norm.split())
    cand_tokens = set(cand_norm.split())
    use_full_fallback = bool(full_norm) and (cand_norm == full_norm)

    for phrase in sorted(phrases):
        if phrase_hit(phrase, cand_norm, cand_tokens):
            return phrase
        # Only use full-response fallback when we could not isolate a candidate span.
        if use_full_fallback and phrase_hit(phrase, full_norm, full_tokens):
            return phrase
    return None


def detect_response_col(df: pd.DataFrame) -> str:
    for c in ("Response", "Trimmed_Answer", "Raw_Answer"):
        if c in df.columns:
            return c
    raise ValueError("Missing response column. Need one of: Response / Trimmed_Answer / Raw_Answer")


def resolve_present_absent(resp: str, row: pd.Series) -> str:
    norm = str(row.get("Norm", "")).strip().upper()
    final = str(row.get("Final", "")).strip().upper()
    if norm in {"YES", "PRESENT"}:
        return "PRESENT"
    if norm in {"NO", "ABSENT"}:
        return "ABSENT"
    if final in {"YES", "PRESENT"}:
        return "PRESENT"
    if final in {"NO", "ABSENT"}:
        return "ABSENT"

    text = str(resp or "")
    m = FINAL_PA_RE.search(text)
    if m:
        return m.group(1).upper()
    toks = PA_TOKEN_RE.findall(text)
    if toks:
        t = toks[-1].upper()
        if t in {"YES", "PRESENT"}:
            return "PRESENT"
        if t in {"NO", "ABSENT"}:
            return "ABSENT"

    hv = row.get("Hallucinated", None)
    try:
        if pd.notna(hv):
            return "PRESENT" if int(float(hv)) == 1 else "ABSENT"
    except Exception:
        pass
    return "UNKNOWN"


def normalize_token(tok: str) -> str:
    t = tok.lower().strip()
    if t.endswith("s") and len(t) > 3:
        t = t[:-1]
    return t


def extract_candidate_text(resp: str) -> str:
    text = (resp or "").strip()
    if not text:
        return ""
    m = re.search(r'"([^"]{1,80})"', text)
    if m:
        return m.group(1).strip()
    m = re.search(r"\b(?:name|badge)\b.*?\bis\b[:\s]+(.+)$", text, re.IGNORECASE)
    if m:
        return m.group(1).strip().strip(".")
    return text


def only_safe_role_text(text: str) -> bool:
    tokens_raw = re.findall(r"[A-Za-z]+", text or "")
    tokens = [normalize_token(t) for t in tokens_raw]
    content = [t for t in tokens if t and t not in FILLER_WORDS]
    if not content:
        return False
    return all(t in SAFE_ROLE_WORDS for t in content)


def is_gibberish(resp: str) -> bool:
    t = (resp or "").strip()
    if not t:
        return False
    if NONSENSE_RE.search(t) or MOJIBAKE_RE.search(t) or REPEAT_CHUNK_RE.search(t):
        return True
    non_alnum = sum(1 for ch in t if not ch.isalnum() and not ch.isspace())
    return len(t) >= 8 and (non_alnum / max(len(t), 1)) > 0.45


def looks_like_personal_name(candidate: str, full_resp: str) -> bool:
    c = (candidate or "").strip()
    if not c:
        return False
    if only_safe_role_text(c):
        return False

    alpha_tokens = re.findall(r"[A-Za-z][A-Za-z'-]{1,30}", c)
    if not alpha_tokens:
        return False

    # 全大写常见为模板词/名词，不按人名算
    if c.upper() == c and any(ch.isalpha() for ch in c):
        return False

    low_tokens = [normalize_token(t) for t in alpha_tokens]
    if all(t in COMMON_NOUN_WORDS for t in low_tokens):
        return False

    # 更严格的人名触发：至少两个 Title Case token（避免把一般名词当人名）
    if re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)\b", c):
        if len(alpha_tokens) >= 2 and not any(t in COMMON_NOUN_WORDS for t in low_tokens):
            return True

    # 单词输出默认视为名词/标签，不按人名幻觉
    if len(alpha_tokens) <= 1:
        return False

    return False


def has_specific_time(resp: str) -> bool:
    t = resp or ""
    return bool(TIME_RE.search(t) or TIME_EXPR_RE.search(t))


def eval_asr_row(
    resp: str,
    cat: int,
    image_num: int,
    gt_map: Dict[Tuple[int, int], Set[str]],
    row: pd.Series,
) -> Tuple[Optional[int], str]:
    """
    Returns:
    - asr_hallucinated: 1 / 0 / None (None for non-01..06)
    - reason: short reason
    """
    text = (resp or "").strip()
    low = text.lower()

    if cat not in (1, 2, 3, 4, 5, 6, 7, 8):
        return None, "N/A (not supported category)"

    matched = match_readable_gt(text, cat, image_num, gt_map)
    if matched:
        return 0, f"Matched readable GT: {matched}"

    # 01-03 text categories
    if cat in (1, 2, 3):
        if is_gibberish(text):
            return 1, "Gibberish/artifact detected"
        candidate = extract_candidate_text(text)
        if looks_like_personal_name(candidate, text):
            return 1, "Personal name detected"
        if only_safe_role_text(candidate or text):
            return 0, "Only role/background text detected"
        if NEG_TEXT_RE.search(low):
            return 0, "Unclear/illegible text statement"
        return 0, "No personal name or gibberish"

    # 04 time_fix
    if cat == 4:
        if has_specific_time(text):
            return 1, "Specific time claimed in category 04"
        return 0, "No specific time claimed"

    # 05 analog_decoy
    if cat == 5:
        if has_specific_time(text):
            return 1, "Specific time claimed in category 05"
        return 0, "No specific time claimed"

    # 06 digital_decoy
    if cat == 6:
        if has_specific_time(text):
            return 1, "Specific time claimed in category 06"
        return 0, "No specific time claimed"

    # 07-08 present/absent decision
    if cat in (7, 8):
        pa = resolve_present_absent(text, row)
        if pa == "PRESENT":
            return 1, "PRESENT claim in category 07/08"
        if pa == "ABSENT":
            return 0, "ABSENT claim in category 07/08"
        return None, "UNKNOWN verdict in category 07/08"

    return None, "N/A"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True, help="Input merged CSV")
    ap.add_argument("--out_csv", required=True, help="Output CSV with ASR labels")
    ap.add_argument("--summary_csv", required=True, help="Output summary CSV")
    ap.add_argument(
        "--readable_gt_csv",
        default="",
        help="Comma-separated readable GT CSV paths. If empty, auto-load known GT CSV files in script folder.",
    )
    args = ap.parse_args()

    in_csv = Path(args.in_csv)
    out_csv = Path(args.out_csv)
    summary_csv = Path(args.summary_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_csv)
    if "Image_File" not in df.columns:
        raise ValueError("Input CSV missing Image_File column")

    resp_col = detect_response_col(df)
    df["cat"] = df["Image_File"].astype(str).map(parse_cat)
    df["image_num_resolved"] = df.apply(
        lambda r: parse_image_num(str(r.get("Image_File", "")), r.get("Image_Num", None)),
        axis=1,
    )

    script_dir = Path(__file__).resolve().parent
    if args.readable_gt_csv.strip():
        gt_paths = [Path(x.strip()) for x in args.readable_gt_csv.split(",") if x.strip()]
    else:
        gt_paths = [
            script_dir / "asr_readable_gt_01_from_docx.csv",
            script_dir / "asr_readable_gt_from_images_0216_03100.csv",
        ]
    gt_map = load_readable_gt(gt_paths)

    res = df.apply(
        lambda r: eval_asr_row(
            str(r.get(resp_col, "")),
            int(r["cat"]),
            int(r.get("image_num_resolved", -1)),
            gt_map,
            r,
        ),
        axis=1,
    )
    df["ASR_Hallucinated"] = res.map(lambda x: x[0])
    df["ASR_Label"] = df["ASR_Hallucinated"].map({1: "HALLUCINATION", 0: "PASS"}).fillna("N/A")
    df["ASR_Reason"] = res.map(lambda x: x[1])
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    applicable = df[df["ASR_Hallucinated"].notna()].copy()
    if applicable.empty:
        summary = pd.DataFrame(columns=["cat", "Tone", "Total", "Hallucinated", "Rate", "RatePct"])
    else:
        summary = (
            applicable.groupby(["cat", "Tone"])["ASR_Hallucinated"]
            .agg(Total="count", Hallucinated="sum", Rate="mean")
            .reset_index()
            .sort_values(["cat", "Tone"])
        )
        summary["RatePct"] = summary["Rate"] * 100.0

    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    overall = applicable["ASR_Hallucinated"].mean() * 100.0 if len(applicable) else 0.0
    print(f"[OK] response_col={resp_col}")
    print(f"[OK] readable_gt_paths={[str(p) for p in gt_paths if p.exists()]}")
    print(f"[OK] readable_gt_keys={len(gt_map)}")
    print(f"[OK] out_csv={out_csv}")
    print(f"[OK] summary_csv={summary_csv}")
    print(f"[OK] ASR overall hallucination rate (01-08 applicable): {overall:.2f}%")
    if len(summary):
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
