"""Persisted HMM training/inference helpers for isolated-word recognition."""

from __future__ import annotations

import gzip
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from dictionary_store import group_feature_sequences_by_label
from speech_hmm import (
    _kmeans_codebook,
    _quantize_sequence,
    _score_continuous,
    _score_discrete,
    _train_continuous_hmm,
    _train_discrete_hmm,
)


def train_persisted_hmm_models(
    dictionary: List[Dict[str, Any]],
    feature_type: str,
    model_type: str = "discrete",
    n_states: int = 5,
    codebook_size: int = 32,
    n_iters: int = 10,
) -> Dict[str, Any]:
    """Train per-label HMM models and return serializable bundle."""
    grouped = group_feature_sequences_by_label(dictionary, feature_type)
    if not grouped:
        return {"models": {}, "feature_type": feature_type, "model_type": model_type}

    bundle: Dict[str, Any] = {
        "version": 1,
        "feature_type": feature_type,
        "model_type": model_type,
        "n_states": int(n_states),
        "n_iters": int(n_iters),
        "models": {},
    }

    if model_type == "discrete":
        all_vectors = [seq for seqs in grouped.values() for seq in seqs if seq.size > 0]
        if not all_vectors:
            return bundle
        data = np.concatenate(all_vectors, axis=0)
        if data.shape[0] < 2:
            return bundle
        codebook = _kmeans_codebook(data, k=int(codebook_size))
        n_symbols = codebook.shape[0]
        bundle["codebook"] = codebook
        bundle["codebook_size"] = int(codebook_size)

        for label, seqs in grouped.items():
            obs_seqs = [_quantize_sequence(s, codebook) for s in seqs if s.shape[0] >= 2]
            if not obs_seqs:
                continue
            s_states = min(max(2, n_states), max(2, min(len(s) for s in obs_seqs) - 1))
            model = _train_discrete_hmm(obs_seqs, n_states=s_states, n_symbols=n_symbols, n_iters=n_iters)
            bundle["models"][label] = model
    else:
        for label, seqs in grouped.items():
            valid = [s for s in seqs if s.shape[0] >= 2]
            if not valid:
                continue
            s_states = min(max(2, n_states), max(2, min(s.shape[0] for s in valid) - 1))
            model = _train_continuous_hmm(valid, n_states=s_states, n_iters=n_iters)
            bundle["models"][label] = model

    return bundle


def query_persisted_hmm(bundle: Dict[str, Any], query_features: np.ndarray) -> Tuple[Optional[str], Optional[float], List[Tuple[str, float]]]:
    """Score query against pre-trained models and return ranking."""
    if not bundle or query_features.size == 0:
        return None, None, []

    model_type = bundle.get("model_type", "discrete")
    models: Dict[str, Any] = bundle.get("models", {})
    if not models:
        return None, None, []

    ranking: List[Tuple[str, float]] = []
    if model_type == "discrete":
        codebook = np.array(bundle.get("codebook", []), dtype=float)
        if codebook.size == 0:
            return None, None, []
        q_obs = _quantize_sequence(query_features, codebook)
        for label, model in models.items():
            score = _score_discrete(q_obs, model)
            ranking.append((label, float(score)))
    else:
        for label, model in models.items():
            score = _score_continuous(query_features, model)
            ranking.append((label, float(score)))

    if not ranking:
        return None, None, []
    ranking.sort(key=lambda x: x[1], reverse=True)
    return ranking[0][0], ranking[0][1], ranking


def save_hmm_bundle(bundle: Dict[str, Any], path: str) -> None:
    """Save trained HMM bundle to compressed file."""
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with gzip.open(path, "wb", compresslevel=5) as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_hmm_bundle(path: str) -> Dict[str, Any]:
    """Load trained HMM bundle."""
    with gzip.open(path, "rb") as f:
        return pickle.load(f)
