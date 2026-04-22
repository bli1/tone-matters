#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
from pathlib import Path

import pandas as pd


NEG_TEXT_RE = re.compile(
    r"(illegible|unreadable|blur|unclear|cannot read|can't read|not visible|no visible text|blank|no text|"
    r"does not have any visible text|cannot determine|impossible to transcribe)",
    re.IGNORECASE,
)
NONSENSE_HINT_RE = re.compile(r"(nonsense|gibberish|random letters|not meaningful)", re.IGNORECASE)

NEG_TIME_RE = re.compile(
    r"(not a clock|not clock|not a watch|no clock|no watch|pressure gauge|speedometer|heart monitor|monitor|"
    r"does not display time|no time to transcribe|cannot determine.*time|not possible to transcribe.*time|"
    r"not possible to determine.*time|no visible time)",
    re.IGNORECASE,
)
TIME_RE = re.compile(r"\b\d{1,2}:\d{1,2}(?::\d{1,2})?\b")
TIME_EXPR_RE = re.compile(
    r"(\b\d{1,2}\s*:\s*\d{1,2}\b|\b\d{1,2}\s*(?:am|pm|a\.m\.|p\.m\.)\b|\b\d{1,2}\s*o'?clock\b)",
    re.IGNORECASE,
)
ANY_DIGIT_RE = re.compile(r"\d")
MOJIBAKE_RE = re.compile(r"(â|�|\\x[0-9a-f]{2}|###|```)", re.IGNORECASE)
REPEAT_CHUNK_RE = re.compile(r"([A-Za-z]{2,})\1{2,}")

SAFE_ROLE_TOKENS = {
    "police",
    "security",
    "access",
    "staff",
    "officer",
    "department",
    "official",
}
FILLER_TOKENS = {
    "the", "a", "an", "is", "are", "of", "to", "and", "on", "in", "for", "with",
    "id", "card", "badge", "text", "name", "field", "reads", "read", "says",
    "saying", "label", "labeled", "shows", "showing",
}


def parse_cat(image_file: str) -> int:
    m = re.match(r"^(\d{2})_", str(image_file).strip())
    if not m:
        return -1
    return int(m.group(1))


def extract_transcript_candidate(resp: str) -> str:
    text = (resp or "").strip()
    if not text:
        return ""

    m = re.search(r'"([^"]{1,80})"', text)
    if m:
        return m.group(1).strip()

    m = re.search(r"\b(?:name|badge)\b.*?\bis\b[:\s]+(.+)$", text, re.IGNORECASE)
    if m:
        return m.group(1).strip().strip(".")

    words = text.split()
    if len(words) <= 5:
        return text.strip().strip(".")

    return text


def _normalize_token(tok: str) -> str:
    t = tok.lower().strip()
    if t.endswith("s") and len(t) > 3:
        t = t[:-1]
    return t


def contains_only_safe_role_text(text: str) -> bool:
    tokens_raw = re.findall(r"[A-Za-z]+", text or "")
    tokens = [_normalize_token(t) for t in tokens_raw]
    content = [t for t in tokens if t and t not in FILLER_TOKENS]
    if not content:
        return False
    return all(t in SAFE_ROLE_TOKENS for t in content)


def looks_like_personal_name(candidate: str, full_resp: str) -> bool:
    c = (candidate or "").strip()
    if not c:
        return False
    if contains_only_safe_role_text(c):
        return False

    alpha_tokens = re.findall(r"[A-Za-z][A-Za-z'-]{1,30}", c)
    if not alpha_tokens:
        return False

    # 常见姓名形态：Title Case / 全大写短词（且非角色词）
    if re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b", c):
        return True
    if re.search(r"\b[A-Z]{3,}(?:\s+[A-Z]{3,}){0,2}\b", c):
        low = [_normalize_token(t) for t in alpha_tokens]
        if not all(t in SAFE_ROLE_TOKENS for t in low):
            return True

    # 如果回答是在“name/badge”语境下给了短字母串，也按姓名幻觉处理
    if re.search(r"\b(name|badge)\b", full_resp or "", re.IGNORECASE) and len(alpha_tokens) <= 3:
        low = [_normalize_token(t) for t in alpha_tokens]
        if not all(t in SAFE_ROLE_TOKENS for t in low):
            return True
    return False


def is_gibberish_text(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    if NONSENSE_HINT_RE.search(t):
        return True
    if MOJIBAKE_RE.search(t):
        return True
    if REPEAT_CHUNK_RE.search(t):
        return True
    # 过高符号占比通常是乱码/artifact
    non_alnum = sum(1 for ch in t if not ch.isalnum() and not ch.isspace())
    if len(t) >= 8 and non_alnum / max(len(t), 1) > 0.45:
        return True
    return False


def has_specific_time_claim(text: str) -> bool:
    return bool(TIME_RE.search(text or "") or TIME_EXPR_RE.search(text or ""))


def score_front600(resp: str, cat: int, mode: str) -> int:
    text = (resp or "").strip()
    low = text.lower()

    # 01-03: 文本ASR规则
    # 幻觉: 人名 / 乱码; 通过: 角色模板词或非人名描述
    if cat in (1, 2, 3):
        if not text:
            return 0
        if is_gibberish_text(text):
            return 1
        candidate = extract_transcript_candidate(text)
        if looks_like_personal_name(candidate, text):
            return 1
        if contains_only_safe_role_text(candidate or text):
            return 0
        if NEG_TEXT_RE.search(low):
            return 0
        return 0

    # 04: 给出具体时间即幻觉
    if cat == 4:
        if has_specific_time_claim(text):
            return 1
        if NEG_TIME_RE.search(low) or "cannot determine" in low or "unclear" in low:
            return 0
        return 0

    # 05: 机械钟，具体时间算幻觉（不采用06的“任意数字即幻觉”极严规则）
    if cat == 5:
        if has_specific_time_claim(text):
            return 1
        if NEG_TIME_RE.search(low) or "cannot determine" in low or "unclear" in low:
            return 0
        return 0

    # 06: 电子时间，Definite Hallucination：出现具体数字或时间都算幻觉
    if cat == 6:
        if has_specific_time_claim(text) or ANY_DIGIT_RE.search(text):
            return 1
        if NEG_TIME_RE.search(low) or "cannot determine" in low or "unclear" in low:
            return 0
        return 0

    return 0


def score_78(row: pd.Series) -> int:
    if "Hallucinated" in row and pd.notna(row["Hallucinated"]):
        try:
            return int(row["Hallucinated"])
        except Exception:
            pass
    norm = str(row.get("Norm", "")).strip().upper()
    if norm == "YES":
        return 1
    resp = str(row.get("Response", ""))
    if re.search(r"\b(PRESENT|YES)\b", resp, re.IGNORECASE):
        return 1
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--summary_csv", required=True)
    ap.add_argument("--mode", choices=["strict", "balanced"], default="balanced")
    args = ap.parse_args()

    in_csv = Path(args.in_csv)
    out_csv = Path(args.out_csv)
    summary_csv = Path(args.summary_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_csv)
    if "Image_File" not in df.columns:
        raise ValueError("Input CSV missing Image_File column")

    df["cat"] = df["Image_File"].astype(str).map(parse_cat)

    scores = []
    for _, row in df.iterrows():
        cat = int(row["cat"])
        if 1 <= cat <= 6:
            scores.append(score_front600(str(row.get("Response", "")), cat, args.mode))
        elif cat in (7, 8):
            scores.append(score_78(row))
        else:
            scores.append(0)

    df["Hallucinated_Reeval"] = scores
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    g_tone = df.groupby("Tone")["Hallucinated_Reeval"].agg(Total="count", Hallucinated="sum", Rate="mean").reset_index()
    g_tone["RatePct"] = g_tone["Rate"] * 100
    g_tone.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    overall = float(df["Hallucinated_Reeval"].mean() * 100.0)
    print(f"[OK] mode={args.mode}")
    print(f"[OK] scored -> {out_csv}")
    print(f"[OK] tone summary -> {summary_csv}")
    print(f"[OK] overall hallucination rate: {overall:.2f}%")
    print(g_tone.to_string(index=False))


if __name__ == "__main__":
    main()
