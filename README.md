# OCR Benchmark on African Exam Documents

**Evaluating OCR engines, preprocessing and post-correction on scanned multilingual exam papers from Cameroon and Côte d'Ivoire — and measuring the impact of OCR quality on a downstream NLP task.**

This repository extends my Master's thesis *Text Classification of OCR-Extracted Data* (University of Ngaoundéré, 2024, supervisor Pr. Dr.-Ing. Jean Louis Fendji) into a reproducible **OCR evaluation framework**.

## Questions addressed

1. How accurate are open-source OCR engines (Tesseract, EasyOCR, Keras-OCR, PaddleOCR, docTR) on real African exam documents, measured with **CER / WER against human-verified ground truth**?
2. Which **preprocessing** steps (deskewing, binarization, denoising, contrast) give measurable gains?
3. Do engines **preserve document structure** (line breaks, reading order)?
4. What is the **operational cost** (seconds per page, CPU only)?
5. How much does OCR quality matter for a **downstream task** (subject classification), when evaluated without data leakage?

## Corpus

| | Original thesis | This version |
|---|---|---|
| Documents | 248 PDFs | 214 PDFs |
| Pages | 470 | 384 |
| Classes | 16 subjects (4 to 80 pages) | **12 classes (21 to 40 pages)** |
| Languages | French, English, German, Spanish | same |
| Split | random by page (**leaky**) | **by document** (StratifiedGroupKFold) |

Balancing: Physique + Chimie + Physique-Chimie → *Physique-Chimie*; Biologie + SVT → *SVT*; *Culture Générale* (4 pages) dropped; classes capped at 40 pages by sampling whole documents.

## Pipeline

```
PDF → images (pdf2image) → preprocessing variants → OCR engines → post-correction
    → evaluation (CER, WER, missing text, line structure, time, confidence)
    → downstream classification (TF-IDF + 8 models, document-level CV)
    → prototype (Streamlit: side-by-side, confidence flags, TXT/JSON export)
```

## Repository layout

```
src/ocrbench/      dataset.py · preprocessing.py · engines.py · metrics.py · classify.py
scripts/           01_build_dataset → 02_classify → 03_select_ground_truth → 04_run_ocr → 05_evaluate_ocr
app/               streamlit_app.py  (digitisation prototype)
ground_truth/      evaluation_set.csv · images/ · to_correct/ · verified/
results/           CSV tables and figures produced by the scripts
```

## Reproduce

```bash
pip install -r requirements.txt          # + Tesseract with fra, eng, deu, spa language data
python scripts/01_build_dataset.py --dataset "<path>/Dataset"
python scripts/02_classify.py
python scripts/03_select_ground_truth.py --dataset "<path>/Dataset" --n 30
#   -> correct each file of ground_truth/to_correct/ and save it in ground_truth/verified/
python scripts/04_run_ocr.py --engines tesseract,easyocr --variants raw,thesis,deskew_otsu,clahe_adaptive
python scripts/05_evaluate_ocr.py
streamlit run app/streamlit_app.py
```

## Ground-truth protocol

Each transcription in `ground_truth/verified/` is a faithful copy of the page: same spelling, accents and punctuation as printed, one printed line per text line, tables read row by row, mathematical formulas typed linearly (e.g. `x^2 + 3x = 0`), figures ignored. Pre-filled from Tesseract output and corrected by hand against the image.

## Results

See [`REPORT.md`](REPORT.md). Every table and figure there is produced by the scripts above.

## Roadmap

- [x] Document-level evaluation of downstream classification (leakage fix)
- [x] Unified preprocessing module and OCR engine wrappers
- [ ] CER / WER on 30 human-verified pages
- [ ] PaddleOCR, docTR and a vision-language model
- [ ] Post-correction ablation (dictionary, LanguageTool, LLM)
- [ ] Extension to African-language documents (perspective)

## Author

Eunice Priscille KEMAYOU TANTCHOU — eunicekemayou@gmail.com
