"""Step 1 - build the page manifest and the balanced dataset.

Usage:
    python scripts/01_build_dataset.py --dataset "C:/Users/LENOVO/Documents/Master2/RECHERCHES/Soutenance/Projet/Dataset"
Outputs (results/):
    manifest_full.csv, manifest_balanced.csv, dataset_stats.csv, class_distribution.png
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

from ocrbench.dataset import balance, build_manifest

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True)
ap.add_argument("--max-per-class", type=int, default=40)
ap.add_argument("--out", default=str(ROOT / "results"))
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

full = build_manifest(a.dataset)
bal = balance(full, a.max_per_class)
full.to_csv(out / "manifest_full.csv", index=False)
bal.to_csv(out / "manifest_balanced.csv", index=False)

stats = pd.DataFrame({
    "pages_original": full.groupby("subject_raw").size(),
}).reset_index()
s2 = pd.DataFrame({
    "pages_balanced": bal.groupby("label").size(),
    "documents_balanced": bal.groupby("label")["source_doc"].nunique(),
}).reset_index()
s2.to_csv(out / "dataset_stats.csv", index=False)
print(f"Original : {len(full)} pages, {full['source_doc'].nunique()} documents, {full['subject_raw'].nunique()} subjects")
print(f"Balanced : {len(bal)} pages, {bal['source_doc'].nunique()} documents, {bal['label'].nunique()} classes")
print(s2.to_string(index=False))

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
full.groupby("subject_raw").size().sort_values().plot.barh(ax=ax[0], color="#9aa9b8")
ax[0].set_title(f"Original: {full['subject_raw'].nunique()} subjects ({len(full)} pages)")
bal.groupby("label").size().sort_values().plot.barh(ax=ax[1], color="#1f4e79")
ax[1].set_title(f"Balanced: {bal['label'].nunique()} classes ({len(bal)} pages)")
for x in ax: x.set_xlabel("pages"); x.set_ylabel("")
plt.tight_layout(); plt.savefig(out / "class_distribution.png", dpi=150)
