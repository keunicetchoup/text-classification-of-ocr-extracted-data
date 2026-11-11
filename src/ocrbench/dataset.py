"""Build a page-level manifest from the labelled OCR text folders.

Folder layout produced during the Master's thesis:
    Classes/   -> Tesseract output, organised  <country>/<exam>/[<zone|year>/]<subject>/<page>.txt
    Classes2/  -> EasyOCR output   (same layout, Côte d'Ivoire stored under "BEPC/")
    Classes3/  -> Keras-OCR output (same layout as Classes2)

Each row of the manifest = one page image, with
    page_id      file stem (identical across engines)
    source_doc   original PDF (page suffix removed) -> used as the GROUP for splitting
    subject_raw  folder name
    label        harmonised class (see LABEL_MAP)
    text_<engine>
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ENGINE_DIRS = {"tesseract": "Classes", "easyocr": "Classes2", "kerasocr": "Classes3"}

# Harmonisation of the 16 original subject folders into 12 balanced classes.
LABEL_MAP = {
    "Mathématiques": "Mathematiques",
    "Anglais": "Anglais",
    "Allemand": "Allemand",
    "Espagnol": "Espagnol",
    "Histoire Géographie": "Histoire-Geographie",
    "Education Musicale": "Education-Musicale",
    "Education aux Droits de l'Homme et à la Citoyenneté": "EDHC",
    "Composition Française": "Composition-Francaise",
    "Arts Plastiques": "Arts-Plastiques",
    "Orthographe": "Orthographe",
    # merged classes
    "Physique Chimie": "Physique-Chimie",
    "Physique": "Physique-Chimie",
    "Chimie": "Physique-Chimie",
    "Science de la Vie et de la Terre": "SVT",
    "Biologie": "SVT",
    # dropped (only 4 pages, 2 documents)
    "Culture Générale": None,
}

PAGE_SUFFIX = re.compile(r"_page_\d+$", re.IGNORECASE)


def source_doc_of(page_id: str) -> str:
    return PAGE_SUFFIX.sub("", page_id)


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def build_manifest(dataset_dir: str | Path) -> pd.DataFrame:
    dataset_dir = Path(dataset_dir)
    rows: dict[str, dict] = {}
    for engine, sub in ENGINE_DIRS.items():
        root = dataset_dir / sub
        if not root.exists():
            continue
        for f in sorted(root.rglob("*.txt")):
            rel = f.relative_to(root).parts
            subject = rel[-2]
            country = "Cameroun" if rel[0] == "Cameroun" else "Cote d'Ivoire"
            exam = rel[1] if rel[0] == "Cameroun" else "BEPC"
            pid = f.stem
            r = rows.setdefault(pid, {
                "page_id": pid,
                "source_doc": source_doc_of(pid),
                "country": country,
                "exam": exam,
                "subject_raw": subject,
                "label": LABEL_MAP.get(subject, subject),
            })
            r[f"text_{engine}"] = _read(f)
    df = pd.DataFrame(rows.values()).sort_values("page_id").reset_index(drop=True)
    return df


def balance(df: pd.DataFrame, max_per_class: int = 40, seed: int = 42) -> pd.DataFrame:
    """Drop unmapped classes and cap over-represented classes.

    Capping is done by sampling whole *documents* (not pages) so that a class is
    not represented by a handful of near-duplicate pages of a single PDF.
    """
    df = df[df["label"].notna()].copy()
    out = []
    for label, g in df.groupby("label"):
        if len(g) <= max_per_class:
            out.append(g)
            continue
        docs = g.drop_duplicates("source_doc")["source_doc"].sample(frac=1, random_state=seed)
        keep, n = [], 0
        for d in docs:
            k = (g["source_doc"] == d).sum()
            if n + k > max_per_class and n > 0:
                continue
            keep.append(d)
            n += k
        out.append(g[g["source_doc"].isin(keep)])
    return pd.concat(out).sort_values("page_id").reset_index(drop=True)
