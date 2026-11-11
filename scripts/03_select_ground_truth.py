"""Step 3 - select a stratified evaluation subset and pre-fill transcriptions to correct by hand.

Selects N pages (default 30) spread over classes, countries and source documents,
copies the page images and pre-fills each transcription with the Tesseract output.
A human then corrects every file -> ground_truth/verified/<page_id>.txt

Usage:
    python scripts/03_select_ground_truth.py --dataset "<...>/Dataset" --n 30
"""
import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True)
ap.add_argument("--n", type=int, default=30)
ap.add_argument("--seed", type=int, default=7)
a = ap.parse_args()

df = pd.read_csv(ROOT / "results" / "manifest_balanced.csv")
df = df.sample(frac=1, random_state=a.seed)
per_class = max(1, a.n // df["label"].nunique())
picked, used_docs = [], set()
for label, g in df.groupby("label"):
    k = 0
    for _, r in g.iterrows():
        if r["source_doc"] in used_docs:
            continue
        picked.append(r); used_docs.add(r["source_doc"]); k += 1
        if k >= per_class:
            break
rest = df[~df["source_doc"].isin(used_docs)].drop_duplicates("source_doc")
# top-up: favour Cameroon documents (fewer of them) to keep both sources represented
rest = pd.concat([rest[rest.country == "Cameroun"], rest[rest.country != "Cameroun"]])
for _, r in rest.iterrows():
    if len(picked) >= a.n:
        break
    picked.append(r)
sel = pd.DataFrame(picked).sort_values(["label", "page_id"])

gt = ROOT / "ground_truth"
todo, ver, img = gt / "to_correct", gt / "verified", gt / "images"
for p in (todo, ver, img):
    p.mkdir(parents=True, exist_ok=True)
for _, r in sel.iterrows():
    src = Path(a.dataset) / "Images" / f"{r.page_id}.jpg"
    if src.exists():
        shutil.copy2(src, img / src.name)
    (todo / f"{r.page_id}.txt").write_text(str(r.text_tesseract), encoding="utf-8")
sel[["page_id", "source_doc", "country", "exam", "label"]].to_csv(gt / "evaluation_set.csv", index=False)
print(sel.groupby(["label", "country"]).size().to_string())
print(f"\n{len(sel)} pages -> {todo}")
