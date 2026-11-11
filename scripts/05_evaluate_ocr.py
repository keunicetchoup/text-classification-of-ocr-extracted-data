"""Step 5 - CER / WER / layout evaluation against verified ground truth.

Usage:
    python scripts/05_evaluate_ocr.py
Reads  ground_truth/verified/*.txt and outputs/ocr/<engine>/<variant>/*.txt
Writes results/ocr_eval_pages.csv, results/ocr_eval_summary.csv, results/ocr_eval_by_class.csv,
       results/ocr_top_confusions.csv, results/cer_by_engine.png
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from ocrbench.metrics import all_scores, top_confusions

gt_dir = ROOT / "ground_truth" / "verified"
meta = pd.read_csv(ROOT / "ground_truth" / "evaluation_set.csv")
refs = {p.stem: p.read_text(encoding="utf-8") for p in gt_dir.glob("*.txt")}
if not refs:
    sys.exit("No verified transcription yet in ground_truth/verified/")

rows, conf = [], []
for vdir in sorted((ROOT / "outputs" / "ocr").glob("*/*")):
    engine, variant = vdir.parent.name, vdir.name
    for pid, ref in refs.items():
        f = vdir / f"{pid}.txt"
        if not f.exists():
            continue
        hyp = f.read_text(encoding="utf-8")
        rows.append({"engine": engine, "variant": variant, "page_id": pid, **all_scores(ref, hyp)})
        for (r, h), n in top_confusions(ref, hyp):
            conf.append({"engine": engine, "variant": variant, "ref": r, "ocr": h, "count": n})

P = pd.DataFrame(rows).merge(meta, on="page_id", how="left")
runs = ROOT / "results" / "ocr_runs.csv"
if runs.exists():
    P = P.merge(pd.read_csv(runs), on=["engine", "variant", "page_id"], how="left")
P.to_csv(ROOT / "results" / "ocr_eval_pages.csv", index=False)

cols = [c for c in ["cer_strict", "wer_strict", "cer_relaxed", "wer_relaxed", "missing_text_rate", "line_ratio", "seconds", "confidence"] if c in P]
S = P.groupby(["engine", "variant"])[cols].agg(["mean", "median"]).round(4)
S.to_csv(ROOT / "results" / "ocr_eval_summary.csv")
P.groupby(["engine", "variant", "label"])["cer_relaxed"].mean().unstack(0).round(4).to_csv(ROOT / "results" / "ocr_eval_by_class.csv")
(pd.DataFrame(conf).groupby(["engine", "variant", "ref", "ocr"])["count"].sum()
 .reset_index().sort_values("count", ascending=False).to_csv(ROOT / "results" / "ocr_top_confusions.csv", index=False))
print(S.to_string())

m = P.groupby(["engine", "variant"])["cer_relaxed"].mean().unstack()
m.plot.bar(figsize=(9, 5)); plt.ylabel("mean CER (relaxed)"); plt.title(f"CER on {len(refs)} ground-truth pages")
plt.xticks(rotation=0); plt.tight_layout(); plt.savefig(ROOT / "results" / "cer_by_engine.png", dpi=150)
