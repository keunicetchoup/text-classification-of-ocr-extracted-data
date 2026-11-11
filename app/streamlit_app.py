"""Digitisation prototype: upload a page -> extracted text, confidence, side-by-side view, exports.

Run:  streamlit run app/streamlit_app.py
"""
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2
import numpy as np
import streamlit as st

from ocrbench.engines import ENGINES, get_engine
from ocrbench.preprocessing import VARIANTS, preprocess

st.set_page_config(page_title="OCR digitisation prototype", layout="wide")
st.title("OCR digitisation prototype")

with st.sidebar:
    engine = st.selectbox("OCR engine", list(ENGINES), index=0)
    variant = st.selectbox("Preprocessing", list(VARIANTS), index=1)
    low = st.slider("Flag lines below confidence", 0.0, 1.0, 0.6, 0.05)

up = st.file_uploader("Page image (JPG/PNG) or PDF", type=["jpg", "jpeg", "png", "pdf"])


@st.cache_resource
def load(name):
    return get_engine(name)


if up:
    data = up.read()
    if up.name.lower().endswith(".pdf"):
        from pdf2image import convert_from_bytes
        pages = convert_from_bytes(data, dpi=200)
        i = st.number_input("Page", 1, len(pages), 1) - 1
        img = cv2.cvtColor(np.array(pages[i]), cv2.COLOR_RGB2BGR)
    else:
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    proc = preprocess(img, variant)
    with st.spinner("Running OCR..."):
        r = load(engine)(proc)

    m1, m2, m3 = st.columns(3)
    m1.metric("Mean confidence", f"{r.confidence:.0%}" if r.confidence is not None else "n/a")
    m2.metric("Processing time", f"{r.seconds:.1f} s")
    m3.metric("Lines to review", sum(c < low for _, c in r.lines))

    left, right = st.columns(2)
    left.image(proc, caption="Preprocessed source", use_container_width=True)
    with right:
        st.subheader("Extracted text")
        for t, c in r.lines:
            st.markdown(f":red[{t}]" if c < low else t)

    q = st.text_input("Search in extracted text")
    if q:
        hits = [t for t, _ in r.lines if q.lower() in t.lower()]
        st.write(f"{len(hits)} line(s) found", hits)

    payload = {"file": up.name, "engine": engine, "preprocessing": variant,
               "mean_confidence": r.confidence, "seconds": r.seconds,
               "lines": [{"text": t, "confidence": c, "needs_review": c < low} for t, c in r.lines]}
    st.download_button("Download text (.txt)", r.text, file_name=f"{Path(up.name).stem}.txt")
    st.download_button("Download structured output (.json)", json.dumps(payload, ensure_ascii=False, indent=2),
                       file_name=f"{Path(up.name).stem}.json")
