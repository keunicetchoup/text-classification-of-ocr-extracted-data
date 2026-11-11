# Macro-F1 by model, OCR engine and split protocol

`group` = document-level 5-fold CV (correct). `random` = page-level random split (original thesis, leaky).

| model              |   ('group', 'easyocr') |   ('group', 'kerasocr') |   ('group', 'tesseract') |   ('random', 'easyocr') |   ('random', 'kerasocr') |   ('random', 'tesseract') |
|:-------------------|-----------------------:|------------------------:|-------------------------:|------------------------:|-------------------------:|--------------------------:|
| DecisionTree       |                  0.876 |                   0.877 |                    0.883 |                   0.875 |                    0.848 |                     0.886 |
| KNN                |                  0.968 |                   0.971 |                    0.974 |                   0.983 |                    0.983 |                     0.984 |
| LogisticRegression |                  0.976 |                   0.975 |                    0.975 |                   0.995 |                    0.991 |                     0.989 |
| MLP                |                  0.982 |                   0.98  |                    0.983 |                   0.993 |                    0.993 |                     0.986 |
| NaiveBayes         |                  0.982 |                   0.976 |                    0.972 |                   0.995 |                    0.991 |                     0.99  |
| RandomForest       |                  0.96  |                   0.956 |                    0.966 |                   0.969 |                    0.98  |                     0.978 |
| SVM                |                  0.978 |                   0.978 |                    0.98  |                   0.993 |                    0.989 |                     0.989 |
| XGBoost            |                  0.92  |                   0.904 |                    0.922 |                   0.953 |                    0.942 |                     0.936 |
