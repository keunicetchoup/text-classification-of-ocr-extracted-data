"""Step 4 - run OCR engines x preprocessing variants on the evaluation set (or any page list).

Usage:
    python scripts/04_run_ocr.py --images ground_truth/images --engines tesseract,easyocr --variants raw,thesis,deskew_otsu
Outputs:
    outputs/ocr/<engine>/<variant>/<page_id>.txt
    results/ocr_runs.csv   (seconds per page, mean confidence)
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2
import pandas as pd

from ocrbench.engines import get_engine
from ocrbench.preprocessing import preprocess

ap = argparse.ArgumentParser()
ap.add_argument("--images", default=str(ROOT / "ground_truth" / "images"))
ap.add_argument("--engines", default="tesseract")
ap.add_argument("--variants", default="raw,thesis")
ap.add_argument("--tess-lang", default="fra+eng+deu+spa")
a = ap.parse_args()

imgs = sorted(Path(a.images).glob("*.jpg"))
log_path = ROOT / "results" / "ocr_runs.csv"
rows = []
for ename in a.engines.split(","):
    eng = get_engine(ename, lang=a.tess_lang) if ename == "tesseract" else get_engine(ename)
    for variant in a.variants.split(","):
        od = ROOT / "outputs" / "ocr" / ename / variant
        od.mkdir(parents=True, exist_ok=True)
        for p in imgs:
            im = preprocess(cv2.imread(str(p)), variant)
            r = eng(im)
            (od / f"{p.stem}.txt").write_text(r.text, encoding="utf-8")
            rows.append({"engine": ename, "variant": variant, "page_id": p.stem,
                         "seconds": round(r.seconds, 2), "confidence": r.confidence})
            print(ename, variant, p.stem, f"{r.seconds:.1f}s", flush=True)
new = pd.DataFrame(rows)
if log_path.exists():
    old = pd.read_csv(log_path)
    old = old.merge(new[["engine", "variant", "page_id"]], how="left", indicator=True)
    new = pd.concat([old[old["_merge"] == "left_only"].drop(columns="_merge"), new])
new.to_csv(log_path, index=False)
