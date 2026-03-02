"""Dataset ingestion helpers for MSWC microset."""

from __future__ import annotations

import os
import io
import subprocess
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import signal
from scipy.io import wavfile
from scipy.io.wavfile import WavFileWarning

from dictionary_store import build_entry, load_dictionary, save_dictionary
from speech_dsp import (
    DEFAULT_MEL_FILTERS,
    DEFAULT_MEL_FMAX_HZ,
    DEFAULT_MEL_FMIN_HZ,
    DEFAULT_PRE_EMPHASIS,
    compute_mfcc,
    compute_spectrogram,
)


def _to_float_mono(audio: np.ndarray) -> np.ndarray:
    """Convert arbitrary PCM array to mono float in [-1, 1]."""
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    if np.issubdtype(audio.dtype, np.integer):
        max_val = max(1, np.iinfo(audio.dtype).max)
        return audio.astype(np.float32) / float(max_val)
    return audio.astype(np.float32)


def _resample_if_needed(samples: np.ndarray, fs_in: int, fs_out: int) -> np.ndarray:
    if fs_in == fs_out:
        return samples.astype(np.float32)
    g = np.gcd(fs_in, fs_out)
    up = fs_out // g
    down = fs_in // g
    y = signal.resample_poly(samples.astype(np.float32), up, down)
    return y.astype(np.float32)


def load_audio_for_app(path: str, target_fs: int) -> np.ndarray:
    """Load wav file into mono float32 array at app sampling rate."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".wav":
        fs_in, data = wavfile.read(path)
    elif ext == ".opus":
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            path,
            "-f",
            "wav",
            "-",
        ]
        raw = subprocess.check_output(cmd)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", WavFileWarning)
            fs_in, data = wavfile.read(io.BytesIO(raw))
    else:
        raise ValueError(f"Unsupported audio extension: {ext}")
    samples = _to_float_mono(data)
    return _resample_if_needed(samples, int(fs_in), int(target_fs))


def collect_mswc_files(
    dataset_root: str,
    languages: Sequence[str] = ("es", "en"),
) -> List[Tuple[str, str, str]]:
    """Collect (wav_path, language, label) from MSWC folder layout."""
    out: List[Tuple[str, str, str]] = []
    for lang in languages:
        clips_root = os.path.join(dataset_root, lang, "clips")
        if not os.path.isdir(clips_root):
            continue
        for label in sorted(os.listdir(clips_root)):
            label_dir = os.path.join(clips_root, label)
            if not os.path.isdir(label_dir):
                continue
            for fname in sorted(os.listdir(label_dir)):
                if not (fname.lower().endswith(".wav") or fname.lower().endswith(".opus")):
                    continue
                out.append((os.path.join(label_dir, fname), lang, label))
    return out


def import_mswc_to_dictionary(
    dictionary_path: str,
    dataset_root: str,
    fs: int,
    frame_ms: float,
    hop_ms: float,
    pre_emphasis: float = DEFAULT_PRE_EMPHASIS,
    languages: Sequence[str] = ("es", "en"),
    max_per_label: Optional[int] = None,
    max_total_per_language: Optional[int] = 2000,
) -> Dict[str, int]:
    """Import MSWC utterances into dictionary using MFCC feature type."""
    existing = load_dictionary(dictionary_path)
    entries = list(existing)
    existing_sources = set()
    for entry in existing:
        src = entry.get("source_utterance")
        if isinstance(src, str):
            existing_sources.add(src)

    files = collect_mswc_files(dataset_root, languages=languages)
    per_label_count: Dict[Tuple[str, str], int] = {}
    per_lang_total: Dict[str, int] = {}
    added = 0
    skipped_existing = 0
    skipped_invalid = 0

    for wav_path, lang, label in files:
        source_key = f"{lang}:{label}:{os.path.basename(wav_path)}"
        if source_key in existing_sources:
            skipped_existing += 1
            continue

        key = (lang, label)
        lang_total = per_lang_total.get(lang, 0)
        if max_total_per_language is not None and lang_total >= int(max_total_per_language):
            continue
        if max_per_label is not None:
            cur = per_label_count.get(key, 0)
            if cur >= int(max_per_label):
                continue

        try:
            samples = load_audio_for_app(wav_path, target_fs=fs)
        except Exception:
            skipped_invalid += 1
            continue

        if samples.size < int(fs * frame_ms / 1000.0):
            skipped_invalid += 1
            continue

        spect = compute_spectrogram(
            samples,
            fs,
            frame_ms=frame_ms,
            hop_ms=hop_ms,
            pre_emphasis=pre_emphasis,
        )
        mfcc = compute_mfcc(
            samples,
            fs,
            num_filters=DEFAULT_MEL_FILTERS,
            num_ceps=DEFAULT_MEL_FILTERS,
            frame_ms=frame_ms,
            hop_ms=hop_ms,
            fmin=DEFAULT_MEL_FMIN_HZ,
            fmax=DEFAULT_MEL_FMAX_HZ,
            pre_emphasis=pre_emphasis,
        )
        if mfcc.size == 0:
            skipped_invalid += 1
            continue

        entry = build_entry(
            label=f"{lang}:{label}",
            fs=fs,
            segment_seconds=(0.0, float(len(samples)) / fs),
            spectrogram=spect,
            feature_type="mfcc",
            feature_matrix=mfcc,
            frame_ms=frame_ms,
            hop_ms=hop_ms,
            mel_filters=DEFAULT_MEL_FILTERS,
            mel_fmin_hz=DEFAULT_MEL_FMIN_HZ,
            mel_fmax_hz=DEFAULT_MEL_FMAX_HZ,
            pre_emphasis=pre_emphasis,
        )
        entry["dataset"] = "mswc_microset"
        entry["dataset_language"] = lang
        entry["dataset_label"] = label
        entry["source_utterance"] = source_key

        entries.append(entry)
        existing_sources.add(source_key)
        per_label_count[key] = per_label_count.get(key, 0) + 1
        per_lang_total[lang] = per_lang_total.get(lang, 0) + 1
        added += 1

    if added > 0:
        save_dictionary(dictionary_path, entries)

    return {
        "added": added,
        "skipped_existing": skipped_existing,
        "skipped_invalid": skipped_invalid,
        "labels_touched": len(per_label_count),
        "total_candidates": len(files),
        "per_language_added": per_lang_total,
    }
