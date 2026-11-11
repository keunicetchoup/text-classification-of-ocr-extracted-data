"""OCR quality metrics against a ground-truth transcription.

CER / WER follow the usual definition (S + D + I) / N, computed with `jiwer`.
Two normalisation levels are reported, because they answer different questions:
  * "strict"  : only whitespace normalised  -> what an archive user sees
  * "relaxed" : lower-case, no punctuation, accents kept -> pure recognition quality
Layout is approximated by the line-count ratio and a reading-order proxy.
"""
from __future__ import annotations

import re
import unicodedata

import jiwer


def norm_strict(t: str) -> str:
    t = unicodedata.normalize("NFC", t or "")
    t = re.sub(r"[ \t]+", " ", t)
    return re.sub(r"\s*\n\s*", "\n", t).strip()


def norm_relaxed(t: str) -> str:
    t = unicodedata.normalize("NFC", t or "").lower()
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _flat(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip()


def cer(ref: str, hyp: str) -> float:
    ref, hyp = _flat(ref), _flat(hyp)
    return jiwer.cer(ref, hyp) if ref else float("nan")


def wer(ref: str, hyp: str) -> float:
    ref, hyp = _flat(ref), _flat(hyp)
    return jiwer.wer(ref, hyp) if ref else float("nan")


def word_error_breakdown(ref: str, hyp: str) -> dict:
    o = jiwer.process_words(_flat(norm_relaxed(ref)), _flat(norm_relaxed(hyp)))
    n = max(1, o.hits + o.substitutions + o.deletions)
    return {"substitutions": o.substitutions, "deletions": o.deletions,
            "insertions": o.insertions, "missing_text_rate": o.deletions / n}


def layout_scores(ref: str, hyp: str) -> dict:
    """Line-structure preservation (1.0 = same number of non-empty lines)."""
    rl = [l for l in norm_strict(ref).split("\n") if l.strip()]
    hl = [l for l in norm_strict(hyp).split("\n") if l.strip()]
    ratio = min(len(rl), len(hl)) / max(1, max(len(rl), len(hl)))
    return {"ref_lines": len(rl), "hyp_lines": len(hl), "line_ratio": ratio}


def all_scores(ref: str, hyp: str) -> dict:
    d = {
        "cer_strict": cer(norm_strict(ref), norm_strict(hyp)),
        "wer_strict": wer(norm_strict(ref), norm_strict(hyp)),
        "cer_relaxed": cer(norm_relaxed(ref), norm_relaxed(hyp)),
        "wer_relaxed": wer(norm_relaxed(ref), norm_relaxed(hyp)),
    }
    d.update(word_error_breakdown(ref, hyp))
    d.update(layout_scores(ref, hyp))
    return d


def top_confusions(ref: str, hyp: str, k: int = 20) -> list[tuple[str, str]]:
    """Most frequent word substitutions (for the error analysis section)."""
    from collections import Counter
    o = jiwer.process_words(norm_relaxed(ref), norm_relaxed(hyp))
    r, h = o.references[0], o.hypotheses[0]
    c = Counter()
    for a in o.alignments[0]:
        if a.type == "substitute":
            for i, j in zip(range(a.ref_start_idx, a.ref_end_idx), range(a.hyp_start_idx, a.hyp_end_idx)):
                c[(r[i], h[j])] += 1
    return c.most_common(k)
