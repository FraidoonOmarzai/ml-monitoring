"""
train.py — Train a RandomForest classifier on the UCI Heart Disease dataset.

Output
------
app/model.pkl            serialised sklearn Pipeline (scaler + classifier)
app/model_metadata.json  version, metrics, feature importances, timestamps

Usage
-----
    python train.py

Environment
-----------
The script first tries to download the real UCI dataset. If the URL is
unreachable (CI, air-gapped environments) it falls back to synthetic data
generated from the same statistical distribution as the Cleveland split.
"""

from __future__ import annotations

import json
import logging
import pathlib
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)
FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]
TARGET = "target"
MODEL_VERSION = "1.0.0"
OUTPUT_DIR = pathlib.Path("app")

# ── Data ──────────────────────────────────────────────────────────────────────


def _synthetic(n: int = 303) -> pd.DataFrame:
    """
    Synthetic data matching UCI Cleveland distribution.
    Used only as fallback when UCI URL is unavailable.
    """
    log.warning(
        "Using SYNTHETIC data (UCI URL unreachable). Not for production training."
    )
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "age": rng.normal(54, 9, n).clip(29, 77).astype(int),
            "sex": rng.choice([0, 1], n, p=[0.32, 0.68]),
            "cp": rng.choice([0, 1, 2, 3], n, p=[0.47, 0.17, 0.28, 0.08]),
            "trestbps": rng.normal(131, 17, n).clip(94, 200).astype(int),
            "chol": rng.normal(246, 52, n).clip(126, 564).astype(int),
            "fbs": rng.choice([0, 1], n, p=[0.85, 0.15]),
            "restecg": rng.choice([0, 1, 2], n, p=[0.48, 0.50, 0.02]),
            "thalach": rng.normal(149, 23, n).clip(71, 202).astype(int),
            "exang": rng.choice([0, 1], n, p=[0.67, 0.33]),
            "oldpeak": rng.exponential(1.0, n).clip(0, 6.2).round(1),
            "slope": rng.choice([0, 1, 2], n, p=[0.21, 0.46, 0.33]),
            "ca": rng.choice([0, 1, 2, 3], n, p=[0.59, 0.23, 0.13, 0.05]),
            "thal": rng.choice([1, 2, 3], n, p=[0.06, 0.55, 0.39]),
            "target": rng.choice([0, 1], n, p=[0.46, 0.54]),
        }
    )


def load_data() -> pd.DataFrame:
    try:
        log.info("Downloading UCI Heart Disease dataset …")
        df = pd.read_csv(UCI_URL, names=FEATURES + [TARGET], na_values="?")
        n_before = len(df)
        df = df.dropna()
        log.info(
            "Rows: %d  (dropped %d with missing values)", len(df), n_before - len(df)
        )
        df[TARGET] = (df[TARGET] > 0).astype(int)
    except Exception as exc:
        log.warning("UCI download failed (%s) — using synthetic fallback", exc)
        df = _synthetic()

    counts = df[TARGET].value_counts()
    log.info(
        "Class distribution:  no-disease=%d  disease=%d",
        counts.get(0, 0),
        counts.get(1, 0),
    )
    return df


# ── Training ──────────────────────────────────────────────────────────────────


def train(df: pd.DataFrame) -> tuple[Pipeline, dict]:
    X, y = df[FEATURES], df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=8,
                    min_samples_leaf=5,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc")
    log.info("CV AUC: %.4f ± %.4f", cv_auc.mean(), cv_auc.std())

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_prob)
    test_acc = accuracy_score(y_test, y_pred)
    log.info("Test AUC: %.4f   Test Accuracy: %.4f", test_auc, test_acc)
    log.info(
        "\n%s",
        classification_report(y_test, y_pred, target_names=["No Disease", "Disease"]),
    )

    importances = dict(
        zip(
            FEATURES,
            pipeline.named_steps["clf"].feature_importances_.round(4).tolist(),
        )
    )

    metadata = {
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "feature_names": FEATURES,
        "metrics": {
            "cv_auc_mean": round(float(cv_auc.mean()), 4),
            "cv_auc_std": round(float(cv_auc.std()), 4),
            "test_auc": round(float(test_auc), 4),
            "test_accuracy": round(float(test_acc), 4),
        },
        "feature_importances": importances,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    return pipeline, metadata


# ── Saving ────────────────────────────────────────────────────────────────────


def save(pipeline: Pipeline, metadata: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = OUTPUT_DIR / "model.pkl"
    meta_path = OUTPUT_DIR / "model_metadata.json"
    joblib.dump(pipeline, model_path)
    meta_path.write_text(json.dumps(metadata, indent=2))
    log.info("Saved model    → %s", model_path)
    log.info("Saved metadata → %s", meta_path)


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    df = load_data()
    pipeline, metadata = train(df)
    save(pipeline, metadata)
    log.info("✅  Training complete — v%s", MODEL_VERSION)


if __name__ == "__main__":
    main()
