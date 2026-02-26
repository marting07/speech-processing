"""Dictionary storage and schema helpers for speech recognition entries."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple, Literal

import numpy as np

from speech_dsp import (
    DEFAULT_FRAME_MS,
    DEFAULT_HOP_MS,
    DEFAULT_MEL_FILTERS,
    DEFAULT_MEL_FMAX_HZ,
    DEFAULT_MEL_FMIN_HZ,
    DEFAULT_PRE_EMPHASIS,
    DEFAULT_LPC_ORDER,
    DEFAULT_WAVELET_MIN_SCALE,
    DEFAULT_WAVELET_MAX_SCALE,
    dtw_distance,
    extract_entry_mfcc,
    matrix_to_indexed_rows,
)


def load_dictionary(path: str) -> List[Dict[str, Any]]:
    """Load dictionary entries from JSON file. Returns [] if missing/invalid."""
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
            return data if isinstance(data, list) else []
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_dictionary(path: str, entries: List[Dict[str, Any]]) -> None:
    """Persist dictionary entries to JSON file."""
    with open(path, "w", encoding="utf-8") as file:
        json.dump(entries, file, indent=2)


def append_entry(path: str, entry: Dict[str, Any]) -> None:
    """Append one entry to dictionary file, creating it if needed."""
    entries = load_dictionary(path)
    entries.append(entry)
    save_dictionary(path, entries)


def build_entry(
    *,
    label: str,
    fs: int,
    segment_seconds: Tuple[float, float],
    spectrogram: np.ndarray,
    feature_type: Literal["mfcc", "bark", "lpc", "wavelet"],
    feature_matrix: np.ndarray,
    frame_ms: float = DEFAULT_FRAME_MS,
    hop_ms: float = DEFAULT_HOP_MS,
    mel_filters: int = DEFAULT_MEL_FILTERS,
    mel_fmin_hz: float = DEFAULT_MEL_FMIN_HZ,
    mel_fmax_hz: float = DEFAULT_MEL_FMAX_HZ,
    pre_emphasis: float = DEFAULT_PRE_EMPHASIS,
    lpc_order: int = DEFAULT_LPC_ORDER,
    wavelet_min_scale: int = DEFAULT_WAVELET_MIN_SCALE,
    wavelet_max_scale: int = DEFAULT_WAVELET_MAX_SCALE,
) -> Dict[str, Any]:
    """Build one dictionary entry using the current JSON schema."""
    entry = {
        "schema_version": "2.0",
        "label": label,
        "feature_type": feature_type,
        "sampling_rate_hz": fs,
        "frame_ms": frame_ms,
        "hop_ms": hop_ms,
        "window": "hamming",
        "pre_emphasis": pre_emphasis,
        "segment_seconds": [segment_seconds[0], segment_seconds[1]],
        "spectrogram": {
            "num_frames": int(spectrogram.shape[0]),
            "num_bins": int(spectrogram.shape[1]) if spectrogram.size > 0 else 0,
            "matrix": spectrogram.tolist(),
        },
    }
    if feature_type == "mfcc":
        entry["mfcc"] = {
            "num_filters": mel_filters,
            "num_coefficients": mel_filters,
            "fmin_hz": mel_fmin_hz,
            "fmax_hz": mel_fmax_hz,
            "num_frames": int(feature_matrix.shape[0]),
            "frames": matrix_to_indexed_rows(feature_matrix, "coefficients"),
        }
    elif feature_type == "bark":
        entry["bark_energies"] = {
            "num_bands": 16,
            "num_frames": int(feature_matrix.shape[0]),
            "frames": matrix_to_indexed_rows(feature_matrix, "band_energies"),
        }
    elif feature_type == "lpc":
        entry["lpc"] = {
            "order": int(lpc_order),
            "num_frames": int(feature_matrix.shape[0]),
            "frames": matrix_to_indexed_rows(feature_matrix, "coefficients"),
        }
    else:
        entry["wavelet"] = {
            "min_scale": int(wavelet_min_scale),
            "max_scale": int(wavelet_max_scale),
            "num_frames": int(feature_matrix.shape[0]),
            "frames": matrix_to_indexed_rows(feature_matrix, "scale_energies"),
        }
    return entry


def extract_entry_bark(entry: Dict[str, Any]) -> np.ndarray:
    """Extract Bark-band matrix from dictionary schema."""
    bark_field = entry.get("bark_energies")
    if not isinstance(bark_field, dict):
        return np.empty((0, 0), dtype=float)
    rows = bark_field.get("frames", [])
    matrix = [row.get("band_energies", []) for row in rows if isinstance(row, dict)]
    return np.array(matrix, dtype=float)


def extract_entry_lpc(entry: Dict[str, Any]) -> np.ndarray:
    """Extract LPC matrix from dictionary schema."""
    field = entry.get("lpc")
    if not isinstance(field, dict):
        return np.empty((0, 0), dtype=float)
    rows = field.get("frames", [])
    matrix = [row.get("coefficients", []) for row in rows if isinstance(row, dict)]
    return np.array(matrix, dtype=float)


def extract_entry_wavelet(entry: Dict[str, Any]) -> np.ndarray:
    """Extract wavelet feature matrix from dictionary schema."""
    field = entry.get("wavelet")
    if not isinstance(field, dict):
        return np.empty((0, 0), dtype=float)
    rows = field.get("frames", [])
    matrix = [row.get("scale_energies", []) for row in rows if isinstance(row, dict)]
    return np.array(matrix, dtype=float)


def extract_entry_feature(
    entry: Dict[str, Any],
    feature_type: Literal["mfcc", "bark", "lpc", "wavelet"],
) -> np.ndarray:
    """Extract selected feature matrix from an entry."""
    if feature_type == "mfcc":
        return extract_entry_mfcc(entry)
    if feature_type == "bark":
        return extract_entry_bark(entry)
    if feature_type == "lpc":
        return extract_entry_lpc(entry)
    return extract_entry_wavelet(entry)


def find_best_match_label(
    dictionary: List[Dict[str, Any]],
    query_features: np.ndarray,
    feature_type: Literal["mfcc", "bark", "lpc", "wavelet"],
) -> Tuple[Optional[str], Optional[float]]:
    """Find closest dictionary label for query features using DTW."""
    best_label: Optional[str] = None
    best_distance: Optional[float] = None
    for entry in dictionary:
        entry_feature_type = entry.get("feature_type", "mfcc")
        if entry_feature_type != feature_type:
            continue
        entry_features = extract_entry_feature(entry, feature_type)
        if entry_features is None or entry_features.size == 0:
            continue
        try:
            distance = dtw_distance(query_features, entry_features)
        except Exception:
            continue
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_label = entry.get("label")
    return best_label, best_distance
