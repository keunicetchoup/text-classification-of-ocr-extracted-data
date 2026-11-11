"""Step 2 - downstream classification benchmark (document-level CV vs. original random split).

Usage:
    python scripts/02_classify.py                 # uses results/manifest_balanced.csv
    python scripts/02_classify.py --manifest results/manifest_full.csv --tag full
Outputs (results/):
    classification_<tag>.csv, classification_<tag>_summary.md, confusion_<engine>_<model>.png, f1_macro_<tag>.png
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

from ocrbench.classify import evaluate

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", default=str(ROOT / "results" / "manifest_balanced.csv"))
ap.add_argument("--tag", default="balanced")
ap.add_argument("--engines", default="tesseract,easyocr,kerasocr")
ap.add_argument("--splits", default="group,random")
a = ap.parse_args()
out = ROOT / "results"; out.mkdir(exist_ok=True)

df = pd.read_csv(a.manifest)
allres, best_preds = [], {}
for engine in a.engines.split(","):
    for split in a.splits.split(","):
        res, preds = evaluate(df, engine, split=split)
        print(res.round(3).to_string(index=False), flush=True)
        allres.append(res)
        if split == "group":
            bm = res.sort_values("f1_macro").iloc[-1]["model"]
            best_preds[engine] = (bm, preds[bm])

R = pd.concat(allres)
R.to_csv(out / f"classification_{a.tag}.csv", index=False)

piv = R.pivot_table(index="model", columns=["split", "engine"], values="f1_macro").round(3)
(out / f"classification_{a.tag}_summary.md").write_text(
    "# Macro-F1 by model, OCR engine and split protocol\n\n"
    "`group` = document-level 5-fold CV (correct). `random` = page-level random split (original thesis, leaky).\n\n"
    + piv.to_markdown() + "\n", encoding="utf-8")

g = R[R["split"] == "group"].pivot(index="model", columns="engine", values="f1_macro")
g.plot.bar(figsize=(10, 5), color=["#1f4e79", "#4f8fc0", "#a9c6e3"][: g.shape[1]])
plt.ylabel("Macro-F1 (document-level 5-fold CV)"); plt.ylim(0, 1); plt.xticks(rotation=30, ha="right")
plt.title("Subject classification of OCR text - by OCR engine"); plt.tight_layout()
plt.savefig(out / f"f1_macro_{a.tag}.png", dpi=150); plt.close()

for engine, (model, (yt, yp)) in best_preds.items():
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay.from_predictions(yt, yp, ax=ax, xticks_rotation=60, colorbar=False, cmap="Blues")
    ax.set_title(f"{engine} + {model} (document-level CV)")
    plt.tight_layout(); plt.savefig(out / f"confusion_{a.tag}_{engine}_{model}.png", dpi=150); plt.close()
print("\n", piv.to_string())
