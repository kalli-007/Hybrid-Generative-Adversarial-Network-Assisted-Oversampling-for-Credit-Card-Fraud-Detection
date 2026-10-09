import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .advisor import risk_level


TARGET_NAMES = ("Class", "class", "is_fraud", "fraud", "target", "label")
DEFAULT_FEATURES = [
    "Time", "Amount", "MerchantRisk", "DistanceFromHome",
    "DistanceFromLastTransaction", "RatioToMedianPurchase", "RepeatRetailer",
    "UsedChip", "UsedPinNumber", "OnlineOrder", "ForeignTransaction",
    "TransactionsLast24h",
]


class ThresholdClassifier:
    """Serializable probability-threshold wrapper selected on training validation data."""
    def __init__(self, estimator, threshold=0.5):
        self.estimator = estimator
        self.threshold = float(threshold)

    def fit(self, X, y):
        self.estimator.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)


def _target_column(df):
    for name in TARGET_NAMES:
        if name in df.columns:
            return name
    raise ValueError("Target column not found. Use one of: Class, is_fraud, fraud, target or label.")


def load_dataset(path):
    df = pd.read_csv(path)
    if len(df) < 50:
        raise ValueError("The dataset must contain at least 50 records.")
    target = _target_column(df)
    numeric = df.select_dtypes(include=[np.number]).copy()
    if target not in numeric.columns:
        numeric[target] = pd.to_numeric(df[target], errors="coerce")
    numeric = numeric.dropna(subset=[target]).drop_duplicates()
    numeric[target] = numeric[target].astype(int)
    numeric[target] = (numeric[target] != 0).astype(int)
    if numeric[target].nunique() != 2:
        raise ValueError("The target column must contain both legitimate (0) and fraud (1) records.")
    return numeric, target


def inspect_dataset(path):
    df, target = load_dataset(path)
    counts = df[target].value_counts().to_dict()
    missing = int(df.isna().sum().sum())
    return {
        "records": len(df), "columns": len(df.columns), "target": target,
        "fraud": int(counts.get(1, 0)), "legitimate": int(counts.get(0, 0)),
        "fraud_rate": round(counts.get(1, 0) / len(df) * 100, 3),
        "missing": missing, "duplicates": int(pd.read_csv(path).duplicated().sum()),
        "features": [c for c in df.columns if c != target],
    }


def preprocessing_summary(path):
    df, target = load_dataset(path)
    X = df.drop(columns=[target]).replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    y = df[target]
    train_count = int(len(df) * 0.8)
    return {
        "records_after": len(df), "features": X.shape[1],
        "train_count": train_count, "test_count": len(df) - train_count,
        "train_percent": 80, "test_percent": 20,
        "memory_kb": round(df.memory_usage(deep=True).sum() / 1024, 2),
        "steps": ["Duplicate removal", "Missing-value median imputation", "Numeric feature validation",
                  "Target normalization", "80/20 stratified division", "Standard scaling"],
    }


def feature_ranking(path):
    df, target = load_dataset(path)
    X = df.drop(columns=[target]).replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    scores = mutual_info_classif(X, df[target], random_state=42)
    ranked = sorted(zip(X.columns, scores), key=lambda x: x[1], reverse=True)
    selected = min(12, max(5, int(np.ceil(X.shape[1] * 0.75))))
    return {
        "total": X.shape[1], "selected": selected,
        "ranking": [{"name": n, "score": round(float(s), 5), "chosen": i < selected}
                    for i, (n, s) in enumerate(ranked)],
    }


def _simple_smote(X, y, random_state=42):
    rng = np.random.default_rng(random_state)
    minority = X[y == 1]
    majority_count = int((y == 0).sum())
    needed = max(0, majority_count - len(minority))
    if needed == 0 or len(minority) < 2:
        return X, y
    a = rng.integers(0, len(minority), needed)
    b = rng.integers(0, len(minority), needed)
    lam = rng.random((needed, 1))
    synthetic = minority[a] + lam * (minority[b] - minority[a])
    return np.vstack([X, synthetic]), np.concatenate([y, np.ones(needed, dtype=int)])


def _gan_samples(minority, count, epochs=45, random_state=42):
    if count <= 0:
        return np.empty((0, minority.shape[1])), "No generation required"
    np.random.seed(random_state)
    try:
        import tensorflow as tf
        tf.random.set_seed(random_state)
        dim = minority.shape[1]
        latent = max(8, min(32, dim * 2))
        generator = tf.keras.Sequential([
            tf.keras.layers.Input((latent,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(dim),
        ])
        discriminator = tf.keras.Sequential([
            tf.keras.layers.Input((dim,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ])
        discriminator.compile(optimizer=tf.keras.optimizers.Adam(0.0003), loss="binary_crossentropy")
        discriminator.trainable = False
        noise = tf.keras.layers.Input((latent,))
        validity = discriminator(generator(noise))
        combined = tf.keras.Model(noise, validity)
        combined.compile(optimizer=tf.keras.optimizers.Adam(0.0003), loss="binary_crossentropy")
        discriminator.trainable = True
        batch = min(64, max(8, len(minority)))
        rng = np.random.default_rng(random_state)
        for _ in range(epochs):
            idx = rng.integers(0, len(minority), batch)
            real = minority[idx].astype("float32")
            z = rng.normal(size=(batch, latent)).astype("float32")
            fake = generator.predict(z, verbose=0)
            discriminator.train_on_batch(real, np.ones((batch, 1)) * 0.9)
            discriminator.train_on_batch(fake, np.zeros((batch, 1)))
            z = rng.normal(size=(batch, latent)).astype("float32")
            combined.train_on_batch(z, np.ones((batch, 1)))
        generated = generator.predict(rng.normal(size=(count, latent)), verbose=0)
        return np.asarray(generated), "TensorFlow Dense GAN"
    except Exception:
        # Keeps the demo runnable on low-resource systems; installing TensorFlow enables the true GAN above.
        rng = np.random.default_rng(random_state)
        base = minority[rng.integers(0, len(minority), count)]
        spread = np.std(minority, axis=0) + 1e-4
        generated = base + rng.normal(0, 0.08, size=base.shape) * spread
        return generated, "Lightweight GAN-compatible fallback"


def _metrics(model, X_test, y_test):
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "confusion": confusion_matrix(y_test, pred).tolist(),
    }


def _save_confusion(matrix, label, output_dir):
    filename = label.lower().replace(" ", "_").replace("+", "plus") + ".png"
    path = Path(output_dir) / filename
    fig, ax = plt.subplots(figsize=(4.2, 3.5))
    ax.imshow(matrix, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, matrix[i][j], ha="center", va="center", fontsize=13,
                    color="white" if matrix[i][j] > np.max(matrix) / 2 else "#10213c")
    ax.set(xticks=[0, 1], yticks=[0, 1], xlabel="Predicted class", ylabel="Actual class",
           title=label, xticklabels=["Legitimate", "Fraud"], yticklabels=["Legitimate", "Fraud"])
    fig.tight_layout()
    fig.savefig(path, dpi=130, transparent=False)
    plt.close(fig)
    return filename


def train_all(path, model_dir, static_generated_dir):
    df, target = load_dataset(path)
    X = df.drop(columns=[target]).replace([np.inf, -np.inf], np.nan)
    medians = X.median(numeric_only=True).to_dict()
    X = X.fillna(medians).fillna(0)
    y = df[target].to_numpy()

    rank = feature_ranking(path)
    selected = [r["name"] for r in rank["ranking"] if r["chosen"]]
    X = X[selected]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)
    X_smote, y_smote = _simple_smote(X_train_s, y_train)

    majority = int((y_train == 0).sum())
    minority = X_train_s[y_train == 1]
    generated, backend = _gan_samples(minority, majority - len(minority))
    X_gan = np.vstack([X_train_s, generated])
    y_gan = np.concatenate([y_train, np.ones(len(generated), dtype=int)])

    # Select the proposed model threshold using only a validation subset of training data.
    core_x, val_x, core_y, val_y = train_test_split(
        X_train_s, y_train, test_size=0.22, stratify=y_train, random_state=17
    )
    core_minority = core_x[core_y == 1]
    core_needed = int((core_y == 0).sum()) - len(core_minority)
    core_generated, _ = _gan_samples(core_minority, core_needed, epochs=30, random_state=17)
    core_gan = np.vstack([core_x, core_generated])
    core_labels = np.concatenate([core_y, np.ones(len(core_generated), dtype=int)])
    threshold_model = GradientBoostingClassifier(
        n_estimators=250, max_depth=2, learning_rate=0.04, random_state=21
    )
    threshold_model.fit(core_gan, core_labels)
    validation_probability = threshold_model.predict_proba(val_x)[:, 1]
    threshold = max(
        ((f1_score(val_y, validation_probability >= value, zero_division=0), value)
         for value in np.linspace(0.05, 0.60, 23)), key=lambda pair: pair[0]
    )[1]

    models = [
        ("Logistic Regression", "Existing", LogisticRegression(max_iter=1200, class_weight="balanced", random_state=42), X_train_s, y_train),
        ("Decision Tree", "Existing", DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=42), X_train_s, y_train),
        ("Random Forest + SMOTE", "Existing", RandomForestClassifier(n_estimators=180, class_weight="balanced", random_state=42, n_jobs=-1), X_smote, y_smote),
        ("Gradient Boosting + SMOTE", "Existing", GradientBoostingClassifier(n_estimators=130, random_state=42), X_smote, y_smote),
        ("Hybrid GAN + Gradient Boosting", "Proposed", ThresholdClassifier(
            GradientBoostingClassifier(n_estimators=250, max_depth=2, learning_rate=0.04, random_state=21), threshold
        ), X_gan, y_gan),
    ]
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    Path(static_generated_dir).mkdir(parents=True, exist_ok=True)
    results, fitted = [], {}
    for name, kind, model, train_x, train_y in models:
        model.fit(train_x, train_y)
        fitted[name] = model
        values = _metrics(model, X_test_s, y_test)
        image = _save_confusion(np.array(values["confusion"]), name, static_generated_dir)
        results.append({"algorithm": name, "method_type": kind, "backend": backend if kind == "Proposed" else "scikit-learn",
                        "confusion_image": image, **values})

    baselines = [r for r in results if r["method_type"] == "Existing"]
    best_existing = max(baselines, key=lambda r: r["f1"])["algorithm"]
    bundle = {
        "feature_names": selected, "all_features": list(df.drop(columns=[target]).columns),
        "medians": medians, "scaler": scaler, "existing_model": fitted[best_existing],
        "existing_name": best_existing, "proposed_model": fitted["Hybrid GAN + Gradient Boosting"],
        "backend": backend,
    }
    joblib.dump(bundle, Path(model_dir) / "fraud_model_bundle.joblib")
    with open(Path(model_dir) / "training_results.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    return results, {
        "train": len(X_train), "test": len(X_test), "selected": len(selected),
        "total_features": len(df.columns) - 1, "synthetic": len(generated), "backend": backend,
        "best_existing": best_existing, "threshold": round(float(threshold), 3),
    }


def model_ready(model_dir):
    return (Path(model_dir) / "fraud_model_bundle.joblib").exists()


def _frame_for_bundle(frame, bundle):
    data = frame.copy()
    for name in bundle["feature_names"]:
        if name not in data.columns:
            data[name] = bundle["medians"].get(name, 0)
        data[name] = pd.to_numeric(data[name], errors="coerce").fillna(bundle["medians"].get(name, 0))
    return data[bundle["feature_names"]]


def predict_frame(frame, model_dir):
    bundle = joblib.load(Path(model_dir) / "fraud_model_bundle.joblib")
    prepared = _frame_for_bundle(frame, bundle)
    scaled = bundle["scaler"].transform(prepared)
    existing_prob = bundle["existing_model"].predict_proba(scaled)[:, 1]
    proposed_prob = bundle["proposed_model"].predict_proba(scaled)[:, 1]
    proposed_prediction = bundle["proposed_model"].predict(scaled)
    output = frame.copy()
    output["Existing_Probability"] = np.round(existing_prob, 5)
    output["Existing_Result"] = np.where(existing_prob >= 0.5, "Fraud", "Legitimate")
    output["Proposed_Probability"] = np.round(proposed_prob, 5)
    output["Proposed_Result"] = np.where(proposed_prediction == 1, "Fraud", "Legitimate")
    output["Risk_Level"] = [risk_level(float(p)) for p in proposed_prob]
    return output, bundle


def predict_single(values, model_dir):
    frame = pd.DataFrame([values])
    output, bundle = predict_frame(frame, model_dir)
    row = output.iloc[0]
    proposed_prob = float(row["Proposed_Probability"])
    return {
        "existing_result": row["Existing_Result"], "proposed_result": row["Proposed_Result"],
        "existing_probability": float(row["Existing_Probability"]),
        "probability": proposed_prob, "confidence": max(proposed_prob, 1 - proposed_prob),
        "risk": row["Risk_Level"], "existing_name": bundle["existing_name"],
        "backend": bundle["backend"],
    }
