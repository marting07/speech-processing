"""Digital signal processing and feature utilities for the speech app."""

from typing import List, Tuple, Dict, Any

import numpy as np
from scipy import signal, fftpack

# Defaults aligned with Camarena's notes/presentation:
# 30 ms frames, 10 ms hop, Hamming window, Fs=8 kHz and MFCC with M=15.
DEFAULT_FS = 8000
DEFAULT_FRAME_MS = 30.0
DEFAULT_HOP_MS = 10.0
DEFAULT_MEL_FILTERS = 15
DEFAULT_MEL_FMIN_HZ = 80.0
DEFAULT_MEL_FMAX_HZ = 4000.0

# Bark critical bands 1..16 from the notes (band 0 is excluded).
BARK_BAND_LIMITS_HZ = [
    (100, 200),
    (200, 300),
    (300, 400),
    (400, 510),
    (510, 630),
    (630, 770),
    (770, 920),
    (920, 1080),
    (1080, 1270),
    (1270, 1480),
    (1480, 1720),
    (1720, 2000),
    (2000, 2320),
    (2320, 2700),
    (2700, 3150),
    (3150, 3700),
]


def compute_short_time_energy(
    signal_samples: np.ndarray, frame_size: int, hop_size: int
) -> np.ndarray:
    """Compute the short-time energy of a 1-D signal."""
    num_frames = 1 + (len(signal_samples) - frame_size) // hop_size
    energy = np.empty(num_frames)
    for i in range(num_frames):
        start = i * hop_size
        frame = signal_samples[start : start + frame_size]
        energy[i] = np.sum(frame.astype(float) ** 2)
    return energy


def compute_zero_crossing_rate(
    signal_samples: np.ndarray, frame_size: int, hop_size: int
) -> np.ndarray:
    """Compute zero crossing rate (ZCR) for each frame."""
    num_frames = 1 + (len(signal_samples) - frame_size) // hop_size
    zcr = np.empty(num_frames)
    signs = np.sign(signal_samples)
    signs[signs == 0] = 1
    for i in range(num_frames):
        start = i * hop_size
        frame_sign = signs[start : start + frame_size]
        zc = np.sum(np.abs(np.diff(frame_sign)) > 1e-10)
        zcr[i] = zc / frame_size
    return zcr


def compute_spectral_entropy(
    signal_samples: np.ndarray, frame_size: int, hop_size: int
) -> np.ndarray:
    """Compute spectral entropy for each frame."""
    num_frames = 1 + (len(signal_samples) - frame_size) // hop_size
    entropy = np.empty(num_frames)
    window = np.hamming(frame_size)
    for i in range(num_frames):
        start = i * hop_size
        frame = signal_samples[start : start + frame_size] * window
        spectrum = np.abs(np.fft.rfft(frame)) ** 2
        psd = spectrum / np.sum(spectrum + 1e-12)
        entropy[i] = -np.sum(psd * np.log2(psd + 1e-12))
    return entropy


def _adaptive_threshold(feature: np.ndarray, ratio: float) -> float:
    """Estimate threshold using initial-noise statistics and global ratio."""
    if feature.size == 0:
        return 0.0
    noise_frames = int(np.clip(np.ceil(0.1 * len(feature)), 5, 25))
    noise_slice = feature[:noise_frames]
    noise_mean = float(np.mean(noise_slice))
    noise_std = float(np.std(noise_slice))
    return max(ratio * float(np.max(feature)), noise_mean + 2.5 * noise_std)


def _smooth_mask(raw_mask: np.ndarray, hop_ms: float) -> np.ndarray:
    """Fill short gaps and suppress very short speech islands."""
    mask = raw_mask.astype(bool).copy()
    max_gap_frames = max(1, int(round(20.0 / hop_ms)))   # ~20 ms
    min_run_frames = max(1, int(round(50.0 / hop_ms)))   # ~50 ms

    idx = 0
    while idx < len(mask):
        if mask[idx]:
            idx += 1
            continue
        gap_start = idx
        while idx < len(mask) and not mask[idx]:
            idx += 1
        gap_end = idx
        gap_len = gap_end - gap_start
        left_speech = gap_start > 0 and mask[gap_start - 1]
        right_speech = gap_end < len(mask) and mask[gap_end]
        if left_speech and right_speech and gap_len <= max_gap_frames:
            mask[gap_start:gap_end] = True

    idx = 0
    while idx < len(mask):
        if not mask[idx]:
            idx += 1
            continue
        run_start = idx
        while idx < len(mask) and mask[idx]:
            idx += 1
        run_end = idx
        if (run_end - run_start) < min_run_frames:
            mask[run_start:run_end] = False

    return mask


def _mask_to_segments(
    mask: np.ndarray, hop_size: int, frame_size: int, signal_len: int
) -> List[Tuple[int, int]]:
    """Convert a boolean speech mask into sample-index segments."""
    segments: List[Tuple[int, int]] = []
    in_seg = False
    start_frame = 0
    for idx, is_speech in enumerate(mask):
        if is_speech and not in_seg:
            in_seg = True
            start_frame = idx
        elif not is_speech and in_seg:
            in_seg = False
            end_frame = idx
            start_sample = start_frame * hop_size
            end_sample = min(signal_len, end_frame * hop_size + frame_size)
            segments.append((start_sample, end_sample))
    if in_seg:
        start_sample = start_frame * hop_size
        end_sample = signal_len
        segments.append((start_sample, end_sample))
    return segments


def detect_segments_energy(
    signal_samples: np.ndarray,
    fs: int,
    frame_ms: float = DEFAULT_FRAME_MS,
    hop_ms: float = DEFAULT_HOP_MS,
    energy_threshold_ratio: float = 0.1,
) -> List[Tuple[int, int]]:
    """Detect speech segments using short-time energy."""
    frame_size = int(fs * frame_ms / 1000)
    hop_size = int(fs * hop_ms / 1000)
    if len(signal_samples) < frame_size:
        return []
    energy = compute_short_time_energy(signal_samples, frame_size, hop_size)
    threshold = _adaptive_threshold(energy, energy_threshold_ratio)
    mask = _smooth_mask(energy > threshold, hop_ms)
    return _mask_to_segments(mask, hop_size, frame_size, len(signal_samples))


def detect_segments_zcr(
    signal_samples: np.ndarray,
    fs: int,
    frame_ms: float = DEFAULT_FRAME_MS,
    hop_ms: float = DEFAULT_HOP_MS,
    zcr_threshold_ratio: float = 0.1,
) -> List[Tuple[int, int]]:
    """Detect speech segments using short-time zero crossing rate."""
    frame_size = int(fs * frame_ms / 1000)
    hop_size = int(fs * hop_ms / 1000)
    if len(signal_samples) < frame_size:
        return []
    zcr = compute_zero_crossing_rate(signal_samples, frame_size, hop_size)
    threshold = _adaptive_threshold(zcr, zcr_threshold_ratio)
    mask = _smooth_mask(zcr > threshold, hop_ms)
    return _mask_to_segments(mask, hop_size, frame_size, len(signal_samples))


def detect_segments_energy_zcr(
    signal_samples: np.ndarray,
    fs: int,
    frame_ms: float = DEFAULT_FRAME_MS,
    hop_ms: float = DEFAULT_HOP_MS,
    energy_threshold_ratio: float = 0.1,
    zcr_threshold_ratio: float = 0.1,
) -> List[Tuple[int, int]]:
    """Detect speech segments using combined energy/ZCR criterion."""
    frame_size = int(fs * frame_ms / 1000)
    hop_size = int(fs * hop_ms / 1000)
    if len(signal_samples) < frame_size:
        return []
    energy = compute_short_time_energy(signal_samples, frame_size, hop_size)
    zcr = compute_zero_crossing_rate(signal_samples, frame_size, hop_size)
    energy_th = _adaptive_threshold(energy, energy_threshold_ratio)
    zcr_th = _adaptive_threshold(zcr, zcr_threshold_ratio)
    noise_frames = int(np.clip(np.ceil(0.1 * len(energy)), 5, 25))
    noise_slice = energy[:noise_frames]
    noise_energy_floor = float(np.mean(noise_slice) + 2.0 * np.std(noise_slice))
    mask = (energy > energy_th) | ((zcr > zcr_th) & (energy > noise_energy_floor))
    mask = _smooth_mask(mask, hop_ms)
    return _mask_to_segments(mask, hop_size, frame_size, len(signal_samples))


def detect_segments_entropy(
    signal_samples: np.ndarray,
    fs: int,
    frame_ms: float = DEFAULT_FRAME_MS,
    hop_ms: float = DEFAULT_HOP_MS,
    entropy_threshold_ratio: float = 0.1,
) -> List[Tuple[int, int]]:
    """Detect speech segments using spectral entropy."""
    frame_size = int(fs * frame_ms / 1000)
    hop_size = int(fs * hop_ms / 1000)
    if len(signal_samples) < frame_size:
        return []
    entropy = compute_spectral_entropy(signal_samples, frame_size, hop_size)
    threshold = _adaptive_threshold(entropy, entropy_threshold_ratio)
    mask = _smooth_mask(entropy > threshold, hop_ms)
    return _mask_to_segments(mask, hop_size, frame_size, len(signal_samples))


def detect_segments(
    signal_samples: np.ndarray,
    fs: int,
    method: str = "energy",
    frame_ms: float = DEFAULT_FRAME_MS,
    hop_ms: float = DEFAULT_HOP_MS,
    energy_threshold_ratio: float = 0.1,
    zcr_threshold_ratio: float = 0.1,
    entropy_threshold_ratio: float = 0.1,
) -> List[Tuple[int, int]]:
    """Detect speech segments based on the chosen method."""
    if method == "energy":
        return detect_segments_energy(
            signal_samples,
            fs,
            frame_ms=frame_ms,
            hop_ms=hop_ms,
            energy_threshold_ratio=energy_threshold_ratio,
        )
    if method == "zcr":
        return detect_segments_zcr(
            signal_samples,
            fs,
            frame_ms=frame_ms,
            hop_ms=hop_ms,
            zcr_threshold_ratio=zcr_threshold_ratio,
        )
    if method == "energy_zcr":
        return detect_segments_energy_zcr(
            signal_samples,
            fs,
            frame_ms=frame_ms,
            hop_ms=hop_ms,
            energy_threshold_ratio=energy_threshold_ratio,
            zcr_threshold_ratio=zcr_threshold_ratio,
        )
    if method == "entropy":
        return detect_segments_entropy(
            signal_samples,
            fs,
            frame_ms=frame_ms,
            hop_ms=hop_ms,
            entropy_threshold_ratio=entropy_threshold_ratio,
        )
    raise ValueError(f"Unknown segmentation method: {method}")


def compute_mel_filterbank(
    n_filters: int, n_fft: int, fs: int, fmin: float = 0.0, fmax: float = None
) -> np.ndarray:
    """Create a Mel filter bank matrix."""
    if fmax is None:
        fmax = fs / 2

    def hz_to_mel(f: float) -> float:
        return 2595 * np.log10(1 + f / 700)

    def mel_to_hz(mel: float) -> float:
        return 700 * (10 ** (mel / 2595) - 1)

    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_filters + 2)
    hz_points = mel_to_hz(mel_points)
    bin_freqs = np.floor((n_fft + 1) * hz_points / fs).astype(int)

    filterbank = np.zeros((n_filters, n_fft // 2 + 1))
    for i in range(1, n_filters + 1):
        left = bin_freqs[i - 1]
        center = bin_freqs[i]
        right = bin_freqs[i + 1]
        for k in range(left, center):
            filterbank[i - 1, k] = (k - left) / (center - left + 1e-12)
        for k in range(center, right):
            filterbank[i - 1, k] = (right - k) / (right - center + 1e-12)
    return filterbank


def compute_mfcc(
    signal_samples: np.ndarray,
    fs: int,
    num_filters: int = DEFAULT_MEL_FILTERS,
    num_ceps: int = DEFAULT_MEL_FILTERS,
    frame_ms: float = DEFAULT_FRAME_MS,
    hop_ms: float = DEFAULT_HOP_MS,
    pre_emphasis: float = 0.0,
    n_fft: int = None,
    fmin: float = DEFAULT_MEL_FMIN_HZ,
    fmax: float = DEFAULT_MEL_FMAX_HZ,
) -> np.ndarray:
    """Compute MFCC feature matrix for a speech signal."""
    if signal_samples.size == 0:
        return np.empty((0, num_ceps))

    emphasized = signal_samples.astype(float)
    if pre_emphasis > 0:
        emphasized = np.append(
            emphasized[0], emphasized[1:] - pre_emphasis * emphasized[:-1]
        )

    frame_size = int(fs * frame_ms / 1000)
    hop_size = int(fs * hop_ms / 1000)
    if n_fft is None:
        n_fft = frame_size
    if len(emphasized) < frame_size:
        return np.empty((0, num_ceps))

    num_frames = 1 + (len(emphasized) - frame_size) // hop_size
    frames = np.stack([
        emphasized[i * hop_size : i * hop_size + frame_size] for i in range(num_frames)
    ])

    n = np.arange(frame_size)
    hamming = 0.54 - 0.46 * np.cos(np.pi * n / frame_size)
    frames *= hamming

    mag_frames = np.abs(np.fft.rfft(frames, n_fft))
    pow_frames = (1.0 / n_fft) * (mag_frames ** 2)
    filterbank = compute_mel_filterbank(num_filters, n_fft, fs, fmin=fmin, fmax=fmax)
    filterbank_energies = np.dot(pow_frames, filterbank.T)
    filterbank_energies[filterbank_energies == 0] = 1e-12
    log_energies = np.log(filterbank_energies)
    mfcc = fftpack.dct(log_energies, type=2, norm='ortho', axis=1)[:, :num_ceps]
    return mfcc


def compute_spectrogram(
    signal_samples: np.ndarray,
    fs: int,
    frame_ms: float = DEFAULT_FRAME_MS,
    hop_ms: float = DEFAULT_HOP_MS,
) -> np.ndarray:
    """Compute magnitude spectrogram of a signal."""
    frame_size = int(fs * frame_ms / 1000)
    hop_size = int(fs * hop_ms / 1000)
    if signal_samples.size < frame_size:
        return np.empty((0, frame_size // 2 + 1))

    n = np.arange(frame_size)
    hamming = 0.54 - 0.46 * np.cos(np.pi * n / frame_size)
    _, _, zxx = signal.stft(
        signal_samples,
        fs=fs,
        window=hamming,
        nperseg=frame_size,
        noverlap=frame_size - hop_size,
        nfft=frame_size,
        padded=False,
        boundary=None,
    )
    return np.abs(zxx).T


def compute_bark_band_energies(
    signal_samples: np.ndarray,
    fs: int,
    frame_ms: float = DEFAULT_FRAME_MS,
    hop_ms: float = DEFAULT_HOP_MS,
) -> np.ndarray:
    """Compute Bark-band energies (bands 1..16) per frame."""
    frame_size = int(fs * frame_ms / 1000)
    hop_size = int(fs * hop_ms / 1000)
    if signal_samples.size < frame_size:
        return np.empty((0, len(BARK_BAND_LIMITS_HZ)))

    num_frames = 1 + (len(signal_samples) - frame_size) // hop_size
    n = np.arange(frame_size)
    hamming = 0.54 - 0.46 * np.cos(np.pi * n / frame_size)
    fft_freqs = np.fft.rfftfreq(frame_size, d=1 / fs)
    band_bins = [
        np.where((fft_freqs >= lo) & (fft_freqs < hi))[0] for lo, hi in BARK_BAND_LIMITS_HZ
    ]

    energies = np.zeros((num_frames, len(BARK_BAND_LIMITS_HZ)))
    for i in range(num_frames):
        start = i * hop_size
        frame = signal_samples[start : start + frame_size].astype(float) * hamming
        mag = np.abs(np.fft.rfft(frame))
        for b, bins in enumerate(band_bins):
            if bins.size > 0:
                energies[i, b] = np.sum(mag[bins] ** 2)
    return energies


def dtw_distance(seq1: np.ndarray, seq2: np.ndarray) -> float:
    """Compute normalized DTW distance between two feature matrices."""
    n1, d1 = seq1.shape
    n2, d2 = seq2.shape
    if d1 != d2:
        raise ValueError("Feature dimensions do not match")

    cost = np.full((n1 + 1, n2 + 1), np.inf)
    cost[0, 0] = 0.0
    for i in range(1, n1 + 1):
        for j in range(1, n2 + 1):
            dist = np.linalg.norm(seq1[i - 1] - seq2[j - 1])
            cost[i, j] = min(
                cost[i - 1, j - 1] + 2.0 * dist,
                cost[i - 1, j] + dist,
                cost[i, j - 1] + dist,
            )
    return cost[n1, n2] / (n1 + n2)


def matrix_to_indexed_rows(matrix: np.ndarray, value_key: str) -> List[Dict[str, Any]]:
    """Serialize a 2-D feature matrix to dictionary row format."""
    rows = []
    for idx, row in enumerate(matrix.tolist(), start=1):
        rows.append({"frame_index": idx, value_key: row})
    return rows


def extract_entry_mfcc(entry: Dict[str, Any]) -> np.ndarray:
    """Extract MFCC matrix from either legacy or updated dictionary schema."""
    mfcc_field = entry.get("mfcc")
    if isinstance(mfcc_field, list):
        return np.array(mfcc_field, dtype=float)
    if isinstance(mfcc_field, dict):
        rows = mfcc_field.get("frames", [])
        matrix = [row.get("coefficients", []) for row in rows if isinstance(row, dict)]
        return np.array(matrix, dtype=float)
    return np.empty((0, 0), dtype=float)
