"""Downstream evaluation: subject classification of OCR text.

Fixes compared with the original thesis notebooks:
  * document-level splitting (StratifiedGroupKFold on `source_doc`) -> no page of the
    same PDF in both train and test (removes the leakage that made KNN best at k=1);
  * hyper-parameters tuned *inside* each training fold (nested CV);
  * macro-F1 reported next to accuracy and weighted-F1 (classes are imbalanced);
  * one script produces every table and figure -> no inconsistent numbers.
"""
from __future__ import annotations

import re
import time
import unicodedata

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, StratifiedKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
except ImportError:  # optional
    XGBClassifier = None

# Small multilingual stop-word list (FR/EN/DE/ES) - avoids an NLTK download.
STOP = set("""
le la les un une des de du et en à au aux ce ces cette pour par sur dans avec est sont que qui
ne pas plus ou il elle on nous vous ils se sa son ses leur leurs y a été être
the a an of and to in is are for on with that this be as by it at from or which
der die das und ist zu den mit von ein eine nicht sich auf für im dem
el la los las y de en que un una por con para es del al se lo
""".split())


def clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = re.sub(r"[^a-zàâäçéèêëîïôöùûüÿñœæß\s]", " ", text)
    toks = [t for t in text.split() if len(t) > 2 and t not in STOP]
    return " ".join(toks)


def models(seed: int = 42) -> dict:
    m = {
        "NaiveBayes": (ComplementNB(), {"clf__alpha": [0.1, 0.5, 1.0]}),
        "SVM": (LinearSVC(class_weight="balanced", random_state=seed), {"clf__C": [0.1, 1, 10]}),
        "KNN": (KNeighborsClassifier(metric="cosine"), {"clf__n_neighbors": [1, 3, 5, 9]}),
        "LogisticRegression": (LogisticRegression(max_iter=3000, class_weight="balanced"), {"clf__C": [1, 10, 100]}),
        "RandomForest": (RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=seed, n_jobs=-1), {"clf__max_depth": [None, 30]}),
        "DecisionTree": (DecisionTreeClassifier(class_weight="balanced", random_state=seed), {"clf__max_depth": [None, 20]}),
        "MLP": (MLPClassifier(hidden_layer_sizes=(128,), max_iter=200, random_state=seed), {"clf__alpha": [1e-3]}),
    }
    if XGBClassifier is not None:
        m["XGBoost"] = (XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.2, tree_method="hist", random_state=seed, n_jobs=2, verbosity=0), {"clf__max_depth": [4]})
    return m


def _pipe(clf):
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True, max_features=10000)),
        ("clf", clf),
    ])


def evaluate(df: pd.DataFrame, engine: str, split: str = "group", n_splits: int = 5, seed: int = 42) -> tuple[pd.DataFrame, dict]:
    """Return per-model scores (mean over outer folds) and pooled predictions."""
    col = f"text_{engine}"
    d = df[df[col].notna()].reset_index(drop=True)
    X = d[col].map(clean).to_numpy(dtype=object)
    le = LabelEncoder()
    y = le.fit_transform(d["label"].to_numpy(dtype=object))
    groups = d["source_doc"].to_numpy(dtype=object)

    if split == "group":
        outer = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(outer.split(X, y, groups))
    else:  # "random" = the original (leaky) protocol, kept only for comparison
        outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(outer.split(X, y))

    rows, preds = [], {}
    for name, (clf, grid) in models(seed).items():
        y_pred = np.empty_like(y)
        fold_scores = []
        t0 = time.time()
        for tr, te in splits:
            inner = (StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
                     if split == "group" else StratifiedKFold(n_splits=3, shuffle=True, random_state=seed))
            gs = GridSearchCV(_pipe(clf), grid, scoring="f1_macro", cv=inner, n_jobs=1)
            fit_kw = {"groups": groups[tr]} if split == "group" else {}
            gs.fit(X[tr], y[tr], **fit_kw)
            p = gs.predict(X[te])
            y_pred[te] = p
            fold_scores.append(f1_score(y[te], p, average="macro", zero_division=0))
        rows.append({
            "engine": engine, "split": split, "model": name,
            "accuracy": accuracy_score(y, y_pred),
            "precision_macro": precision_score(y, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y, y_pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y, y_pred, average="macro", zero_division=0),
            "f1_weighted": f1_score(y, y_pred, average="weighted", zero_division=0),
            "f1_macro_std_folds": float(np.std(fold_scores)),
            "seconds": round(time.time() - t0, 1),
        })
        print(f"  {engine:9s} {split:6s} {name:18s} f1_macro={rows[-1]['f1_macro']:.3f} ({rows[-1]['seconds']}s)", flush=True)
        preds[name] = (le.inverse_transform(y), le.inverse_transform(y_pred))
    return pd.DataFrame(rows), preds
