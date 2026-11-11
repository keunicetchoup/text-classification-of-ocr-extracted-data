"""Uniform wrappers around OCR engines.

Every engine returns an OcrResult(text, confidence, seconds, lines) so that accuracy,
speed and confidence can be compared on the same pages. Heavy libraries are imported
lazily: install only the engines you want to benchmark.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class OcrResult:
    text: str
    confidence: float | None  # mean confidence in [0, 1], None if unavailable
    seconds: float
    lines: list = field(default_factory=list)  # [(text, conf)] for the prototype


def _lines_to_text(boxes):
    """Rebuild reading order (top->bottom, left->right) from (bbox, text, conf) tuples."""
    items = []
    for bbox, txt, conf in boxes:
        ys = [p[1] for p in bbox]; xs = [p[0] for p in bbox]
        items.append((float(np.mean(ys)), float(min(xs)), float(max(ys) - min(ys)), txt, conf))
    items.sort(key=lambda t: (t[0], t[1]))
    lines, cur, cur_y, cur_h = [], [], None, None
    for y, x, h, txt, conf in items:
        if cur and abs(y - cur_y) > 0.6 * max(cur_h, h, 1):
            lines.append(sorted(cur, key=lambda t: t[0])); cur = []
        if not cur:
            cur_y, cur_h = y, h
        cur.append((x, txt, conf))
    if cur:
        lines.append(sorted(cur, key=lambda t: t[0]))
    out = [(" ".join(t for _, t, _ in l), float(np.mean([c for *_, c in l]))) for l in lines]
    return "\n".join(t for t, _ in out), out


class Tesseract:
    name = "tesseract"

    def __init__(self, lang: str = "fra+eng", psm: int = 3):
        import pytesseract
        self.pt, self.lang, self.cfg = pytesseract, lang, f"--psm {psm}"

    def __call__(self, img) -> OcrResult:
        t0 = time.time()
        # One Tesseract pass only: the text is rebuilt from image_to_data (words come in
        # reading order), so timing is not doubled by a second image_to_string call.
        d = self.pt.image_to_data(img, lang=self.lang, config=self.cfg, output_type=self.pt.Output.DICT)
        conf = [float(c) for c, w in zip(d["conf"], d["text"]) if w.strip() and float(c) >= 0]
        lines = {}  # insertion order = reading order
        for i, w in enumerate(d["text"]):
            if w.strip() and float(d["conf"][i]) >= 0:
                k = (d["block_num"][i], d["par_num"][i], d["line_num"][i])
                lines.setdefault(k, []).append((w, float(d["conf"][i]) / 100))
        ln = [(" ".join(w for w, _ in v), float(np.mean([c for _, c in v]))) for v in lines.values()]
        out, prev = [], None
        for (blk, par, _), (txt, _) in zip(lines.keys(), ln):
            if prev is not None and (blk, par) != prev:
                out.append("")  # blank line between paragraphs/blocks, like image_to_string
            out.append(txt)
            prev = (blk, par)
        text = "\n".join(out)
        return OcrResult(text.strip(), float(np.mean(conf)) / 100 if conf else None, time.time() - t0, ln)


class EasyOCR:
    name = "easyocr"

    def __init__(self, langs=("fr", "en"), gpu=False):
        import easyocr
        self.r = easyocr.Reader(list(langs), gpu=gpu, verbose=False)

    def __call__(self, img) -> OcrResult:
        t0 = time.time()
        res = self.r.readtext(img)
        text, lines = _lines_to_text(res)
        conf = float(np.mean([c for *_, c in res])) if res else None
        return OcrResult(text, conf, time.time() - t0, lines)


class PaddleOCR:
    name = "paddleocr"

    def __init__(self, lang="fr"):
        from paddleocr import PaddleOCR as P
        self.p = P(lang=lang, use_textline_orientation=False)

    def __call__(self, img) -> OcrResult:
        t0 = time.time()
        if img.ndim == 2:
            import cv2
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        res = self.p.predict(img)
        boxes = []
        for r in res:
            for poly, txt, sc in zip(r["rec_polys"], r["rec_texts"], r["rec_scores"]):
                boxes.append((np.asarray(poly).tolist(), txt, float(sc)))
        text, lines = _lines_to_text(boxes)
        conf = float(np.mean([b[2] for b in boxes])) if boxes else None
        return OcrResult(text, conf, time.time() - t0, lines)


class DocTR:
    name = "doctr"

    def __init__(self):
        from doctr.models import ocr_predictor
        self.m = ocr_predictor(pretrained=True)

    def __call__(self, img) -> OcrResult:
        t0 = time.time()
        if img.ndim == 2:
            img = np.stack([img] * 3, -1)
        doc = self.m([img])
        lines, confs = [], []
        for blk in doc.pages[0].blocks:
            for ln in blk.lines:
                ws = [(w.value, w.confidence) for w in ln.words]
                lines.append((" ".join(w for w, _ in ws), float(np.mean([c for _, c in ws]))))
                confs += [c for _, c in ws]
        return OcrResult("\n".join(t for t, _ in lines), float(np.mean(confs)) if confs else None, time.time() - t0, lines)


class KerasOCR:
    """Keras-OCR (CRAFT detector + CRNN recognizer, trained on English / Latin without accents).

    No confidence score is available: `confidence` is None and each line confidence is NaN
    (so the prototype never flags a Keras-OCR line as low-confidence). It needs TensorFlow 2.x and
    `keras-ocr`, which do not build on Python 3.13: use a separate Python 3.10 environment, e.g.
        py -3.10 -m venv .venv-keras && .venv-keras\\Scripts\\pip install keras-ocr tensorflow-cpu==2.13.1 "numpy==1.24.3"
    """
    name = "kerasocr"

    def __init__(self):
        import keras_ocr
        self.p = keras_ocr.pipeline.Pipeline()  # downloads CRAFT + recognizer weights on first use

    def __call__(self, img) -> OcrResult:
        import cv2
        t0 = time.time()
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        words = self.p.recognize([rgb])[0]  # [(word, 4x2 box)]
        boxes = [(np.asarray(box).tolist(), txt, float("nan")) for txt, box in words]
        text, lines = _lines_to_text(boxes)
        return OcrResult(text, None, time.time() - t0, lines)


ENGINES = {"tesseract": Tesseract, "easyocr": EasyOCR, "paddleocr": PaddleOCR, "doctr": DocTR, "kerasocr": KerasOCR}


def get_engine(name: str, **kw):
    return ENGINES[name](**kw)
