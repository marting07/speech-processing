"""From-scratch ANN utilities for isolated-word classification."""

from __future__ import annotations

import gzip
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from dictionary_store import extract_entry_feature


FEATURE_TYPES = ("mfcc", "bark", "lpc", "wavelet")


def feature_matrix_to_vector(matrix: np.ndarray) -> np.ndarray:
    """Convert variable-length feature sequence to fixed-size vector."""
    if matrix.ndim != 2 or matrix.size == 0:
        return np.zeros(2, dtype=float)
    mean = np.mean(matrix, axis=0)
    std = np.std(matrix, axis=0)
    return np.concatenate([mean, std]).astype(float)


def dataset_from_dictionary(dictionary: List[Dict], feature_type: str) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
    """Build (X, y, idx_to_label) using entries of one feature type."""
    if feature_type not in FEATURE_TYPES:
        raise ValueError(f"Unsupported feature type: {feature_type}")

    vectors: List[np.ndarray] = []
    labels: List[str] = []
    for entry in dictionary:
        if entry.get("feature_type", "mfcc") != feature_type:
            continue
        label = entry.get("label")
        if not label:
            continue
        mat = extract_entry_feature(entry, feature_type)
        if mat.size == 0 or mat.ndim != 2:
            continue
        vectors.append(feature_matrix_to_vector(mat))
        labels.append(label)

    if not vectors:
        return np.empty((0, 0), dtype=float), np.array([], dtype=int), {}

    unique_labels = sorted(set(labels))
    label_to_idx = {lab: i for i, lab in enumerate(unique_labels)}
    idx_to_label = {i: lab for lab, i in label_to_idx.items()}
    y = np.array([label_to_idx[lab] for lab in labels], dtype=int)
    X = np.stack(vectors).astype(float)
    return X, y, idx_to_label


def train_test_split_indices(n: int, test_ratio: float = 0.2, seed: int = 13) -> Tuple[np.ndarray, np.ndarray]:
    """Random split indices."""
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = max(1, int(round(n * test_ratio))) if n >= 2 else 0
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    if train_idx.size == 0:
        train_idx = test_idx
    return train_idx, test_idx


def _one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    m = y.shape[0]
    out = np.zeros((m, n_classes), dtype=float)
    out[np.arange(m), y] = 1.0
    return out


def _softmax(z: np.ndarray) -> np.ndarray:
    z_shift = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shift)
    return exp_z / (np.sum(exp_z, axis=1, keepdims=True) + 1e-12)


@dataclass
class AnnModel:
    """Simple MLP model container."""

    feature_type: str
    input_dim: int
    hidden_dim: int
    n_classes: int
    W1: np.ndarray
    b1: np.ndarray
    W2: np.ndarray
    b2: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    idx_to_label: Dict[int, str]

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(0.0, z1)
        z2 = a1 @ self.W2 + self.b2
        p = _softmax(z2)
        return z1, a1, p

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xn = (X - self.mean) / (self.std + 1e-12)
        _, _, p = self.forward(Xn)
        return p

    def predict(self, X: np.ndarray) -> np.ndarray:
        p = self.predict_proba(X)
        return np.argmax(p, axis=1)


def train_ann(
    X_train: np.ndarray,
    y_train: np.ndarray,
    idx_to_label: Dict[int, str],
    feature_type: str,
    hidden_dim: int = 64,
    epochs: int = 80,
    lr: float = 0.01,
    batch_size: int = 16,
    seed: int = 13,
) -> Tuple[AnnModel, Dict[str, List[float]]]:
    """Train a 1-hidden-layer MLP from scratch using SGD."""
    if X_train.size == 0:
        raise ValueError("Empty training set")

    rng = np.random.default_rng(seed)
    n, d = X_train.shape
    n_classes = int(np.max(y_train)) + 1

    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    Xn = (X_train - mean) / (std + 1e-12)

    W1 = rng.normal(0.0, 0.05, size=(d, hidden_dim))
    b1 = np.zeros(hidden_dim, dtype=float)
    W2 = rng.normal(0.0, 0.05, size=(hidden_dim, n_classes))
    b2 = np.zeros(n_classes, dtype=float)

    history = {"loss": [], "acc": []}

    for _ in range(max(1, int(epochs))):
        perm = rng.permutation(n)
        X_epoch = Xn[perm]
        y_epoch = y_train[perm]

        for start in range(0, n, max(1, int(batch_size))):
            xb = X_epoch[start:start + batch_size]
            yb = y_epoch[start:start + batch_size]
            m = xb.shape[0]
            if m == 0:
                continue

            z1 = xb @ W1 + b1
            a1 = np.maximum(0.0, z1)
            z2 = a1 @ W2 + b2
            p = _softmax(z2)
            y_one = _one_hot(yb, n_classes)

            dz2 = (p - y_one) / m
            dW2 = a1.T @ dz2
            db2 = np.sum(dz2, axis=0)

            da1 = dz2 @ W2.T
            dz1 = da1 * (z1 > 0.0)
            dW1 = xb.T @ dz1
            db1 = np.sum(dz1, axis=0)

            W2 -= lr * dW2
            b2 -= lr * db2
            W1 -= lr * dW1
            b1 -= lr * db1

        z1_all = Xn @ W1 + b1
        a1_all = np.maximum(0.0, z1_all)
        p_all = _softmax(a1_all @ W2 + b2)
        loss = float(-np.mean(np.log(p_all[np.arange(n), y_train] + 1e-12)))
        pred = np.argmax(p_all, axis=1)
        acc = float(np.mean(pred == y_train))
        history["loss"].append(loss)
        history["acc"].append(acc)

    model = AnnModel(
        feature_type=feature_type,
        input_dim=d,
        hidden_dim=hidden_dim,
        n_classes=n_classes,
        W1=W1,
        b1=b1,
        W2=W2,
        b2=b2,
        mean=mean,
        std=std,
        idx_to_label=idx_to_label,
    )
    return model, history


def evaluate_ann(model: AnnModel, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Compute accuracy and cross-entropy."""
    if X.size == 0 or y.size == 0:
        return {"accuracy": 0.0, "loss": float("inf")}
    p = model.predict_proba(X)
    pred = np.argmax(p, axis=1)
    acc = float(np.mean(pred == y))
    loss = float(-np.mean(np.log(p[np.arange(y.shape[0]), y] + 1e-12)))
    return {"accuracy": acc, "loss": loss}


def classify_query_vector(model: AnnModel, vector: np.ndarray) -> Tuple[str, float]:
    """Classify one fixed-size vector and return label/confidence."""
    x = vector.reshape(1, -1).astype(float)
    p = model.predict_proba(x)[0]
    idx = int(np.argmax(p))
    return model.idx_to_label.get(idx, str(idx)), float(p[idx])


def save_ann_model(model: AnnModel, path: str) -> None:
    payload = {
        "feature_type": model.feature_type,
        "input_dim": model.input_dim,
        "hidden_dim": model.hidden_dim,
        "n_classes": model.n_classes,
        "W1": model.W1,
        "b1": model.b1,
        "W2": model.W2,
        "b2": model.b2,
        "mean": model.mean,
        "std": model.std,
        "idx_to_label": model.idx_to_label,
    }
    with gzip.open(path, "wb", compresslevel=5) as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_ann_model(path: str) -> AnnModel:
    with gzip.open(path, "rb") as f:
        p = pickle.load(f)
    return AnnModel(
        feature_type=p["feature_type"],
        input_dim=int(p["input_dim"]),
        hidden_dim=int(p["hidden_dim"]),
        n_classes=int(p["n_classes"]),
        W1=np.array(p["W1"], dtype=float),
        b1=np.array(p["b1"], dtype=float),
        W2=np.array(p["W2"], dtype=float),
        b2=np.array(p["b2"], dtype=float),
        mean=np.array(p["mean"], dtype=float),
        std=np.array(p["std"], dtype=float),
        idx_to_label={int(k): str(v) for k, v in p["idx_to_label"].items()},
    )
