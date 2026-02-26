"""Dictionary storage and schema helpers for speech recognition entries."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from speech_dsp import (
    DEFAULT_FRAME_MS,
    DEFAULT_HOP_MS,
    DEFAULT_MEL_FILTERS,
    DEFAULT_MEL_FMAX_HZ,
    DEFAULT_MEL_FMIN_HZ,
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
    bark_energies: np.ndarray,
    mfcc: np.ndarray,
    frame_ms: float = DEFAULT_FRAME_MS,
    hop_ms: float = DEFAULT_HOP_MS,
    mel_filters: int = DEFAULT_MEL_FILTERS,
    mel_fmin_hz: float = DEFAULT_MEL_FMIN_HZ,
    mel_fmax_hz: float = DEFAULT_MEL_FMAX_HZ,
) -> Dict[str, Any]:
    """Build one dictionary entry using the current JSON schema."""
    return {
        "schema_version": "2.0",
        "label": label,
        "sampling_rate_hz": fs,
        "frame_ms": frame_ms,
        "hop_ms": hop_ms,
        "window": "hamming",
        "segment_seconds": [segment_seconds[0], segment_seconds[1]],
        "spectrogram": {
            "num_frames": int(spectrogram.shape[0]),
            "num_bins": int(spectrogram.shape[1]) if spectrogram.size > 0 else 0,
            "matrix": spectrogram.tolist(),
        },
        "bark_energies": {
            "num_bands": 16,
            "num_frames": int(bark_energies.shape[0]),
            "frames": matrix_to_indexed_rows(bark_energies, "band_energies"),
        },
        "mfcc": {
            "num_filters": mel_filters,
            "num_coefficients": mel_filters,
            "fmin_hz": mel_fmin_hz,
            "fmax_hz": mel_fmax_hz,
            "num_frames": int(mfcc.shape[0]),
            "frames": matrix_to_indexed_rows(mfcc, "coefficients"),
        },
    }


def find_best_match_label(
    dictionary: List[Dict[str, Any]],
    query_mfcc: np.ndarray,
) -> Tuple[Optional[str], Optional[float]]:
    """Find closest dictionary label for query MFCC using DTW."""
    best_label: Optional[str] = None
    best_distance: Optional[float] = None
    for entry in dictionary:
        entry_mfcc = extract_entry_mfcc(entry)
        if entry_mfcc is None or entry_mfcc.size == 0:
            continue
        try:
            distance = dtw_distance(query_mfcc, entry_mfcc)
        except Exception:
            continue
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_label = entry.get("label")
    return best_label, best_distance
