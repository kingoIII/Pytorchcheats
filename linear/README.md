Understood. Here’s a compact, company-ready **Linear Models Notes Pack** (binary + multiclass), with clean templates you can drop into projects.

# Linear Classification — Practical Notes

## 1) What “linear” means

* Decision function: ( f(x)= w^\top x + b )
* **Binary (logistic):** (p(y=1|x)=\sigma(f(x))); loss = log-loss; threshold at (\tau).
* **Multiclass (softmax):** (p(y=k|x)=\frac{e^{w_k^\top x}}{\sum_j e^{w_j^\top x}}); loss = cross-entropy.
* Regularization:

  * L2 (ridge): penalizes (|w|_2^2), stable, default.
  * L1 (lasso): sparsity, feature selection.
  * ElasticNet: mix of L1/L2.

## 2) When to use

* Tabular with many features; linear signal suspected.
* High-dim, sparse (text TF-IDF).
* You need calibrated probabilities + interpretability.

## 3) Preprocessing rules

* Numeric: impute → standardize.
* Categorical: one-hot (ignore unknowns).
* Text: TF-IDF (linear models shine here).
* **Fit transforms on train only** (pipeline takes care of it).

---

# SKLEARN — Binary & Multiclass Pipelines (with CV, calibration, thresholding)

```python
# linear_classification_pipeline.py
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, average_precision_score,
                             confusion_matrix, classification_report)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.calibration import CalibratedClassifierCV

# ---- Build preprocessing ----
def build_preprocessor(num_cols, cat_cols):
    num = Pipeline([("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler())])
    cat = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", num, num_cols), ("cat", cat, cat_cols)])

# ---- Choose a linear classifier ----
def make_logreg(C=1.0, penalty="l2", class_weight=None, multi_class="auto", max_iter=2000):
    return LogisticRegression(C=C, penalty=penalty, solver="lbfgs",
                              class_weight=class_weight, multi_class=multi_class, max_iter=max_iter)

def make_sgd(loss="log_loss", alpha=1e-4, class_weight=None):
    # SGD: supports hinge (SVM), log_loss (logistic), modified_huber, etc.
    return SGDClassifier(loss=loss, alpha=alpha, class_weight=class_weight, max_iter=2000, tol=1e-3)

# ---- End-to-end pipeline ----
def build_pipeline(num_cols, cat_cols, model="logreg", **kwargs):
    pre = build_preprocessor(num_cols, cat_cols)
    if model == "logreg":
        clf = make_logreg(**kwargs)
    elif model == "sgd_log":
        clf = make_sgd(loss="log_loss", **kwargs)
    elif model == "sgd_hinge":
        clf = make_sgd(loss="hinge", **kwargs)  # linear SVM (max-margin)
    else:
        raise ValueError("model ∈ {logreg, sgd_log, sgd_hinge}")
    return Pipeline([("pre", pre), ("clf", clf)])

# ---- Cross-validated eval (binary or multiclass) ----
def cv_eval(pipe, X, y, k=5, seed=42, is_binary=True):
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    if is_binary:
        # CV probabilities for metrics & thresholding
        proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
        pred = (proba >= 0.5).astype(int)
        acc = accuracy_score(y, pred)
        f1  = f1_score(y, pred)
        roc = roc_auc_score(y, proba)
        pr  = average_precision_score(y, proba)
        print(f"[Binary] ACC {acc:.3f} | F1 {f1:.3f} | ROC-AUC {roc:.3f} | PR-AUC {pr:.3f}")
        print(confusion_matrix(y, pred))
        return {"proba": proba}
    else:
        # Multiclass uses predicted labels for now
        pred = cross_val_predict(pipe, X, y, cv=cv, method="predict")
        acc = accuracy_score(y, pred)
        print(f"[Multiclass] ACC {acc:.3f}")
        print(classification_report(y, pred))
        return {}

# ---- Threshold selection (binary) ----
def choose_threshold(proba, y, target_precision=None, target_recall=None):
    from sklearn.metrics import precision_recall_curve
    P, R, T = precision_recall_curve(y, proba)
    if target_precision is not None:
        idx = np.argmax(P >= target_precision)
        return T[idx] if P[idx] >= target_precision else 0.5
    if target_recall is not None:
        # last threshold achieving recall target
        idxs = np.where(R >= target_recall)[0]
        return T[idxs[-1]] if len(idxs) else 0.5
    # optional: return F1-optimal
    f1 = 2 * P * R / (P + R + 1e-12)
    return T[np.argmax(f1)]

# ---- Train final with calibration (optional) ----
def fit_final_with_calibration(pipe, X, y, method="isotonic", cv=3):
    # Wrap the whole pipeline with calibration
    final = CalibratedClassifierCV(pipe, method=method, cv=cv)
    final.fit(X, y)
    return final

"""
Usage (binary):
num_cols = [...]; cat_cols=[...]
pipe = build_pipeline(num_cols, cat_cols, model="logreg", C=1.0, class_weight="balanced")
res = cv_eval(pipe, X, y, k=5, is_binary=True)
tau = choose_threshold(res["proba"], y, target_precision=0.9)  # or target_recall=0.95
final = fit_final_with_calibration(pipe, X, y, method="isotonic", cv=3)
# save final with joblib; persist num_cols/cat_cols and tau

Usage (multiclass):
pipe = build_pipeline(num_cols, cat_cols, model="logreg", multi_class="auto")
cv_eval(pipe, X, y, k=5, is_binary=False)
final = pipe.fit(X, y)
"""
```

**Notes**

* For imbalanced data, set `class_weight="balanced"` or provide weights.
* For text, replace preprocessor with `TfidfVectorizer` → `LogisticRegression`.
* For linear SVM probabilities, use `CalibratedClassifierCV` wrapper.

---

# PYTORCH — Minimal Linear Classifiers

## A) Binary Logistic (dense/tabular)

```python
# torch_linear_binary.py
import torch, torch.nn as nn, torch.nn.functional as F

class LogisticBinary(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, 1)

    def forward(self, x):
        logits = self.lin(x).squeeze(1)   # (N,)
        return logits                     # use with BCEWithLogitsLoss

# Training sketch:
# model = LogisticBinary(in_dim)
# loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([w]).to(device))  # optional imbalance
# logits = model(x)                     # (N,)
# loss = loss_fn(logits, y.float())     # y ∈ {0,1}
# prob = torch.sigmoid(logits)          # threshold later
```

## B) Multiclass Softmax

```python
# torch_linear_multiclass.py
import torch, torch.nn as nn

class SoftmaxLinear(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.lin = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.lin(x)  # logits for CrossEntropyLoss

# Training sketch:
# model = SoftmaxLinear(in_dim, K)
# loss_fn = nn.CrossEntropyLoss(weight=class_weights)  # optional
# logits = model(x)              # (N, K)
# loss = loss_fn(logits, y)      # y ∈ {0..K-1}
# pred  = logits.argmax(1)
```

## C) Linear on top of embeddings (text/image)

* Text: `nn.EmbeddingBag` or HF encoder → freeze/finetune → **linear head**.
* Vision: Precomputed features (e.g., from a CNN or ViT) → **linear head**.

---

# Interpreting Linear Models

* **Coefficients** (LogReg/SVM): sign = direction; magnitude = strength (after scaling).
* For one-hot, compare weights per category (watch dummy variable trap).
* SHAP/Permutation importance can validate feature influence.

---

# Calibration

* Logistic Regression usually well-calibrated; still verify **ECE**, **Brier**.
* SVM/SGD → wrap with `CalibratedClassifierCV` (isotonic > platt when data is enough).

---

# Thresholding (binary)

* Pick (\tau) by business target: precision floor, recall floor, cost matrix, or F1-max.
* Re-tune (\tau) when class distribution drifts.

---

# Common Failure Modes

* Leakage: imputation/scaling fitted on full data. **Always pipeline.**
* Categorical high-cardinality: one-hot blowup → use target/WOE encoding (careful with leakage).
* Multicollinearity: hurts interpretability; consider PCA or drop redundant cols (not required for predictive perf with L2).
* Class imbalance: don’t rely on accuracy; use PR-AUC/F1, weights, or resampling.
* Nonlinearity: add interactions/polynomials, or switch to tree/NN.

---

# Quick Recipes

**Text classification (fast & strong baseline)**

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
  ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.9)),
  ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
])
# CV, calibration, and thresholding same as above
```

**Tabular with many cats (balanced)**

```python
pipe = build_pipeline(num_cols, cat_cols, model="logreg",
                      C=1.0, class_weight="balanced", multi_class="auto")
```

**Linear SVM (robust margins, then calibrate)**

```python
pipe = build_pipeline(num_cols, cat_cols, model="sgd_hinge", alpha=1e-4, class_weight="balanced")
final = fit_final_with_calibration(pipe, X, y, method="isotonic", cv=3)
```

---

# Deployment Checklist

* Persist: pipeline + model + version + feature schema + thresholds.
* Log: input schema validation, predictions, latency.
* Monitor: data drift, label drift, calibration drift (ECE), key business metrics.
* Refit cadence: when drift or performance decay crosses threshold.

---

If you want, I can add a **ready-to-run notebook** that compares `LogReg` vs `SGD (hinge/log)` with CV, calibration, and threshold tuning on a sample dataset (tabular or text).
