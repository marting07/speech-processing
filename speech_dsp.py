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
DEFAULT_PRE_EMPHASIS = 0.97
DEFAULT_LPC_ORDER = 12
DEFAULT_WAVELET_MIN_SCALE = 6
DEFAULT_WAVELET_MAX_SCALE = 24

# Segmentation tuning constants.
ADAPTIVE_NOISE_STD_FACTOR = 2.8
NOISE_HEAD_FRACTION = 0.10
NOISE_LOWEST_FRACTION = 0.20
ZCR_ENERGY_FLOOR_STD_FACTOR = 2.5
HYSTERESIS_MARGIN = 0.20
HANGOVER_MS = 120.0
SMOOTH_MS = 35.0
MIN_RUN_MS = 50.0
ONSET_PREROLL_MS = 50.0
FRICATIVE_BAND_LOW_HZ = 1800.0
POST_MERGE_GAP_MS = 150.0
POST_MIN_SEGMENT_MS = 120.0

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


def apply_pre_emphasis(signal_samples: np.ndarray, coeff: float = DEFAULT_PRE_EMPHASIS) -> np.ndarray:
    """Apply first-order pre-emphasis filter y[n]=x[n]-a*x[n-1]."""
    if signal_samples.size == 0:
        return signal_samples.astype(float)
    x = signal_samples.astype(float)
    if coeff <= 0.0:
        return x
    return np.append(x[0], x[1:] - coeff * x[:-1])


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


def compute_energy_trajectory(
    signal_samples: np.ndarray,
    fs: int,
    frame_ms: float = DEFAULT_FRAME_MS,
    hop_ms: float = DEFAULT_HOP_MS,
    pre_emphasis: float = DEFAULT_PRE_EMPHASIS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute normalized short-time energy trajectory and time axis."""
    frame_size = int(fs * frame_ms / 1000)
    hop_size = int(fs * hop_ms / 1000)
    emphasized = apply_pre_emphasis(signal_samples, pre_emphasis)
    if emphasized.size < frame_size:
        return np.array([]), np.array([])
    energy = compute_short_time_energy(emphasized, frame_size, hop_size)
    energy = energy / (np.max(energy) + 1e-12)
    times = (np.arange(len(energy)) * hop_size + frame_size / 2) / fs
    return times, energy


def kalman_filter_1d(
    measurements: np.ndarray,
    process_var: float = 1e-4,
    measurement_var: float = 1e-2,
    init_state: float = None,
    init_var: float = 1.0,
) -> np.ndarray:
    """Apply scalar Kalman filter to a 1-D measurement sequence."""
    if measurements.size == 0:
        return np.array([])
    q = max(process_var, 1e-12)
    r = max(measurement_var, 1e-12)
    x = float(measurements[0] if init_state is None else init_state)
    p = max(init_var, 1e-12)
    out = np.zeros_like(measurements, dtype=float)
    for i, z in enumerate(measurements):
        # Predict
        p = p + q
        # Update
        k = p / (p + r)
        x = x + k * (float(z) - x)
        p = (1.0 - k) * p
        out[i] = x
    return out


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


def compute_high_band_energy_ratio(
    signal_samples: np.ndarray,
    frame_size: int,
    hop_size: int,
    fs: int,
    low_hz: float = FRICATIVE_BAND_LOW_HZ,
) -> np.ndarray:
    """Compute high-frequency energy ratio per frame (useful for fricatives)."""
    num_frames = 1 + (len(signal_samples) - frame_size) // hop_size
    ratios = np.zeros(num_frames, dtype=float)
    window = np.hamming(frame_size)
    freqs = np.fft.rfftfreq(frame_size, d=1.0 / fs)
    high_bins = freqs >= low_hz
    for i in range(num_frames):
        start = i * hop_size
        frame = signal_samples[start : start + frame_size].astype(float) * window
        power = np.abs(np.fft.rfft(frame)) ** 2
        total = float(np.sum(power))
        if total <= 1e-12:
            ratios[i] = 0.0
        else:
            ratios[i] = float(np.sum(power[high_bins]) / total)
    return ratios


def _moving_average(feature: np.ndarray, window_frames: int) -> np.ndarray:
    """Apply simple moving-average smoothing to frame features."""
    if feature.size == 0 or window_frames <= 1:
        return feature.astype(float)
    kernel = np.ones(window_frames, dtype=float) / float(window_frames)
    return np.convolve(feature.astype(float), kernel, mode="same")


def _feature_noise_stats(feature: np.ndarray) -> Tuple[float, float]:
    """Robust noise-floor estimate from head and low-valued frames."""
    if feature.size == 0:
        return 0.0, 0.0
    head_frames = int(np.clip(np.ceil(NOISE_HEAD_FRACTION * len(feature)), 5, 25))
    head_slice = feature[:head_frames]
    low_frames = int(np.clip(np.ceil(NOISE_LOWEST_FRACTION * len(feature)), 5, len(feature)))
    low_slice = np.sort(feature)[:low_frames]
    noise_mean_head = float(np.mean(head_slice))
    noise_mean_low = float(np.mean(low_slice))
    noise_std_head = float(np.std(head_slice))
    noise_std_low = float(np.std(low_slice))
    noise_mean = min(noise_mean_head, noise_mean_low)
    noise_std = min(noise_std_head, noise_std_low)
    # MAD-based robustness for outlier frames.
    median_low = float(np.median(low_slice))
    mad_low = float(np.median(np.abs(low_slice - median_low)))
    robust_std = 1.4826 * mad_low
    noise_std = max(noise_std, robust_std)
    return noise_mean, noise_std


def _adaptive_threshold(feature: np.ndarray, ratio: float) -> float:
    """Estimate threshold using robust noise statistics and global ratio."""
    if feature.size == 0:
        return 0.0
    noise_mean, noise_std = _feature_noise_stats(feature)
    return max(
        ratio * float(np.max(feature)),
        noise_mean + ADAPTIVE_NOISE_STD_FACTOR * noise_std,
    )


def _hysteresis_mask(
    on_condition: np.ndarray,
    off_condition: np.ndarray,
    hangover_frames: int,
    min_run_frames: int,
) -> np.ndarray:
    """Create stable speech mask using hysteresis and hangover."""
    n = len(on_condition)
    mask = np.zeros(n, dtype=bool)
    active = False
    hold = 0
    run_start = -1
    for i in range(n):
        if not active:
            if on_condition[i]:
                active = True
                hold = hangover_frames
                run_start = i
                mask[i] = True
            continue

        mask[i] = True
        if off_condition[i]:
            hold -= 1
            if hold <= 0:
                active = False
                run_len = i - run_start + 1
                if run_len < min_run_frames:
                    mask[run_start : i + 1] = False
                run_start = -1
        else:
            hold = hangover_frames

    if active and run_start >= 0:
        run_len = n - run_start
        if run_len < min_run_frames:
            mask[run_start:] = False
    return mask


def _feature_hysteresis_mask(
    feature: np.ndarray,
    threshold_ratio: float,
    hop_ms: float,
    polarity: str = "high",
) -> np.ndarray:
    """Build speech mask from one feature with smoothing+hysteresis."""
    if feature.size == 0:
        return np.zeros(0, dtype=bool)
    smooth_frames = max(1, int(round(SMOOTH_MS / max(hop_ms, 1e-6))))
    feat_s = _moving_average(feature, smooth_frames)
    noise_mean, noise_std = _feature_noise_stats(feat_s)
    base_th = _adaptive_threshold(feat_s, threshold_ratio)
    hangover_frames = max(1, int(round(HANGOVER_MS / max(hop_ms, 1e-6))))
    min_run_frames = max(1, int(round(MIN_RUN_MS / max(hop_ms, 1e-6))))

    if polarity == "high":
        onset_th = max(base_th, noise_mean + ADAPTIVE_NOISE_STD_FACTOR * noise_std)
        offset_th = max(
            noise_mean + 0.6 * ADAPTIVE_NOISE_STD_FACTOR * noise_std,
            onset_th * (1.0 - HYSTERESIS_MARGIN),
        )
        on_condition = feat_s >= onset_th
        off_condition = feat_s < offset_th
    elif polarity == "low":
        # For features where speech reduces value (e.g., some entropy setups).
        onset_th = min(base_th, noise_mean - ADAPTIVE_NOISE_STD_FACTOR * noise_std)
        offset_th = min(
            noise_mean - 0.6 * ADAPTIVE_NOISE_STD_FACTOR * noise_std,
            onset_th * (1.0 + HYSTERESIS_MARGIN),
        )
        on_condition = feat_s <= onset_th
        off_condition = feat_s > offset_th
    else:
        raise ValueError(f"Unknown polarity: {polarity}")

    return _hysteresis_mask(on_condition, off_condition, hangover_frames, min_run_frames)


def _mask_to_segments(
    mask: np.ndarray,
    hop_size: int,
    frame_size: int,
    signal_len: int,
    onset_preroll_samples: int = 0,
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
            start_sample = max(0, start_frame * hop_size - onset_preroll_samples)
            end_sample = min(signal_len, end_frame * hop_size + frame_size)
            segments.append((start_sample, end_sample))
    if in_seg:
        start_sample = max(0, start_frame * hop_size - onset_preroll_samples)
        end_sample = signal_len
        segments.append((start_sample, end_sample))
    return segments


def _postprocess_segments(
    segments: List[Tuple[int, int]],
    fs: int,
    merge_gap_ms: float = POST_MERGE_GAP_MS,
    min_segment_ms: float = POST_MIN_SEGMENT_MS,
) -> List[Tuple[int, int]]:
    """Merge close segments and remove very short artifacts."""
    if not segments:
        return []

    merge_gap = int(fs * merge_gap_ms / 1000.0)
    min_segment = int(fs * min_segment_ms / 1000.0)

    merged: List[Tuple[int, int]] = []
    cur_start, cur_end = segments[0]
    for start, end in segments[1:]:
        if start - cur_end <= merge_gap:
            cur_end = end
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = start, end
    merged.append((cur_start, cur_end))

    filtered = [(s, e) for s, e in merged if (e - s) >= min_segment]
    if filtered:
        return filtered
    # If all got filtered out, keep the longest merged segment.
    return [max(merged, key=lambda se: se[1] - se[0])]


def detect_segments_energy(
    signal_samples: np.ndarray,
    fs: int,
    frame_ms: float = DEFAULT_FRAME_MS,
    hop_ms: float = DEFAULT_HOP_MS,
    energy_threshold_ratio: float = 0.1,
    onset_preroll_ms: float = ONSET_PREROLL_MS,
) -> List[Tuple[int, int]]:
    """Detect speech segments using short-time energy."""
    frame_size = int(fs * frame_ms / 1000)
    hop_size = int(fs * hop_ms / 1000)
    onset_preroll_samples = int(fs * onset_preroll_ms / 1000.0)
    if len(signal_samples) < frame_size:
        return []
    energy = compute_short_time_energy(signal_samples, frame_size, hop_size)
    mask = _feature_hysteresis_mask(energy, energy_threshold_ratio, hop_ms, polarity="high")
    segments = _mask_to_segments(
        mask, hop_size, frame_size, len(signal_samples), onset_preroll_samples=onset_preroll_samples
    )
    return _postprocess_segments(segments, fs)


def detect_segments_zcr(
    signal_samples: np.ndarray,
    fs: int,
    frame_ms: float = DEFAULT_FRAME_MS,
    hop_ms: float = DEFAULT_HOP_MS,
    zcr_threshold_ratio: float = 0.1,
    onset_preroll_ms: float = ONSET_PREROLL_MS,
    fricative_band_low_hz: float = FRICATIVE_BAND_LOW_HZ,
) -> List[Tuple[int, int]]:
    """Detect speech segments using short-time zero crossing rate."""
    frame_size = int(fs * frame_ms / 1000)
    hop_size = int(fs * hop_ms / 1000)
    onset_preroll_samples = int(fs * onset_preroll_ms / 1000.0)
    if len(signal_samples) < frame_size:
        return []
    zcr = compute_zero_crossing_rate(signal_samples, frame_size, hop_size)
    energy = compute_short_time_energy(signal_samples, frame_size, hop_size)
    hf_ratio = compute_high_band_energy_ratio(
        signal_samples, frame_size, hop_size, fs, low_hz=fricative_band_low_hz
    )
    smooth_frames = max(1, int(round(SMOOTH_MS / max(hop_ms, 1e-6))))
    energy_s = _moving_average(energy, smooth_frames)
    noise_mean_e, noise_std_e = _feature_noise_stats(energy_s)
    energy_gate_th = max(
        noise_mean_e + 0.5 * ADAPTIVE_NOISE_STD_FACTOR * noise_std_e,
        0.04 * float(np.max(energy_s) + 1e-12),
    )
    energy_gate = energy_s >= energy_gate_th
    mask_zcr = _feature_hysteresis_mask(zcr, zcr_threshold_ratio, hop_ms, polarity="high")
    mask_hf = _feature_hysteresis_mask(
        hf_ratio, max(0.05, 0.8 * zcr_threshold_ratio), hop_ms, polarity="high"
    )
    mask = (mask_zcr | mask_hf) & energy_gate
    segments = _mask_to_segments(
        mask, hop_size, frame_size, len(signal_samples), onset_preroll_samples=onset_preroll_samples
    )
    return _postprocess_segments(segments, fs)


def detect_segments_energy_zcr(
    signal_samples: np.ndarray,
    fs: int,
    frame_ms: float = DEFAULT_FRAME_MS,
    hop_ms: float = DEFAULT_HOP_MS,
    energy_threshold_ratio: float = 0.1,
    zcr_threshold_ratio: float = 0.1,
    onset_preroll_ms: float = ONSET_PREROLL_MS,
    fricative_band_low_hz: float = FRICATIVE_BAND_LOW_HZ,
) -> List[Tuple[int, int]]:
    """Detect speech segments using combined energy/ZCR criterion."""
    frame_size = int(fs * frame_ms / 1000)
    hop_size = int(fs * hop_ms / 1000)
    onset_preroll_samples = int(fs * onset_preroll_ms / 1000.0)
    if len(signal_samples) < frame_size:
        return []
    energy = compute_short_time_energy(signal_samples, frame_size, hop_size)
    zcr = compute_zero_crossing_rate(signal_samples, frame_size, hop_size)
    hf_ratio = compute_high_band_energy_ratio(
        signal_samples, frame_size, hop_size, fs, low_hz=fricative_band_low_hz
    )
    smooth_frames = max(1, int(round(SMOOTH_MS / max(hop_ms, 1e-6))))
    energy_s = _moving_average(energy, smooth_frames)
    zcr_s = _moving_average(zcr, smooth_frames)
    hf_s = _moving_average(hf_ratio, smooth_frames)
    energy_th = _adaptive_threshold(energy_s, energy_threshold_ratio)
    zcr_th = _adaptive_threshold(zcr_s, zcr_threshold_ratio)
    hf_th = _adaptive_threshold(hf_s, max(0.05, 0.8 * zcr_threshold_ratio))
    noise_frames = int(np.clip(np.ceil(NOISE_HEAD_FRACTION * len(energy)), 5, 25))
    noise_slice = energy_s[:noise_frames]
    noise_energy_floor = float(
        np.mean(noise_slice) + ZCR_ENERGY_FLOOR_STD_FACTOR * np.std(noise_slice)
    )
    energy_on = energy_s >= energy_th
    energy_off = energy_s < max(noise_energy_floor, energy_th * (1.0 - HYSTERESIS_MARGIN))
    zcr_on = zcr_s >= zcr_th
    zcr_off = zcr_s < zcr_th * (1.0 - HYSTERESIS_MARGIN)
    hf_on = hf_s >= hf_th
    fricative_on = zcr_on & hf_on & (energy_s >= 0.5 * noise_energy_floor)
    on_condition = energy_on | (zcr_on & (energy_s >= noise_energy_floor)) | fricative_on
    off_condition = energy_off & zcr_off
    hangover_frames = max(1, int(round(HANGOVER_MS / max(hop_ms, 1e-6))))
    min_run_frames = max(1, int(round(MIN_RUN_MS / max(hop_ms, 1e-6))))
    mask = _hysteresis_mask(on_condition, off_condition, hangover_frames, min_run_frames)
    segments = _mask_to_segments(
        mask, hop_size, frame_size, len(signal_samples), onset_preroll_samples=onset_preroll_samples
    )
    return _postprocess_segments(segments, fs)


def detect_segments_entropy(
    signal_samples: np.ndarray,
    fs: int,
    frame_ms: float = DEFAULT_FRAME_MS,
    hop_ms: float = DEFAULT_HOP_MS,
    entropy_threshold_ratio: float = 0.1,
    onset_preroll_ms: float = ONSET_PREROLL_MS,
) -> List[Tuple[int, int]]:
    """Detect speech segments using spectral entropy."""
    frame_size = int(fs * frame_ms / 1000)
    hop_size = int(fs * hop_ms / 1000)
    onset_preroll_samples = int(fs * onset_preroll_ms / 1000.0)
    if len(signal_samples) < frame_size:
        return []
    entropy = compute_spectral_entropy(signal_samples, frame_size, hop_size)
    energy = compute_short_time_energy(signal_samples, frame_size, hop_size)
    smooth_frames = max(1, int(round(SMOOTH_MS / max(hop_ms, 1e-6))))
    energy_s = _moving_average(energy, smooth_frames)
    noise_mean_e, noise_std_e = _feature_noise_stats(energy_s)
    energy_gate_th = max(
        noise_mean_e + 0.8 * ADAPTIVE_NOISE_STD_FACTOR * noise_std_e,
        0.04 * float(np.max(energy_s) + 1e-12),
    )
    energy_gate = energy_s >= energy_gate_th
    mask_high = _feature_hysteresis_mask(entropy, entropy_threshold_ratio, hop_ms, polarity="high")
    mask_low = _feature_hysteresis_mask(entropy, entropy_threshold_ratio, hop_ms, polarity="low")
    cand_high = mask_high & energy_gate
    cand_low = mask_low & energy_gate
    mask = cand_high if np.count_nonzero(cand_high) >= np.count_nonzero(cand_low) else cand_low
    segments = _mask_to_segments(
        mask, hop_size, frame_size, len(signal_samples), onset_preroll_samples=onset_preroll_samples
    )
    return _postprocess_segments(segments, fs)


def detect_segments(
    signal_samples: np.ndarray,
    fs: int,
    method: str = "energy",
    frame_ms: float = DEFAULT_FRAME_MS,
    hop_ms: float = DEFAULT_HOP_MS,
    energy_threshold_ratio: float = 0.1,
    zcr_threshold_ratio: float = 0.1,
    entropy_threshold_ratio: float = 0.1,
    onset_preroll_ms: float = ONSET_PREROLL_MS,
    fricative_band_low_hz: float = FRICATIVE_BAND_LOW_HZ,
) -> List[Tuple[int, int]]:
    """Detect speech segments based on the chosen method."""
    if method == "energy":
        return detect_segments_energy(
            signal_samples,
            fs,
            frame_ms=frame_ms,
            hop_ms=hop_ms,
            energy_threshold_ratio=energy_threshold_ratio,
            onset_preroll_ms=onset_preroll_ms,
        )
    if method == "zcr":
        return detect_segments_zcr(
            signal_samples,
            fs,
            frame_ms=frame_ms,
            hop_ms=hop_ms,
            zcr_threshold_ratio=zcr_threshold_ratio,
            onset_preroll_ms=onset_preroll_ms,
            fricative_band_low_hz=fricative_band_low_hz,
        )
    if method == "energy_zcr":
        return detect_segments_energy_zcr(
            signal_samples,
            fs,
            frame_ms=frame_ms,
            hop_ms=hop_ms,
            energy_threshold_ratio=energy_threshold_ratio,
            zcr_threshold_ratio=zcr_threshold_ratio,
            onset_preroll_ms=onset_preroll_ms,
            fricative_band_low_hz=fricative_band_low_hz,
        )
    if method == "entropy":
        return detect_segments_entropy(
            signal_samples,
            fs,
            frame_ms=frame_ms,
            hop_ms=hop_ms,
            entropy_threshold_ratio=entropy_threshold_ratio,
            onset_preroll_ms=onset_preroll_ms,
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
    pre_emphasis: float = DEFAULT_PRE_EMPHASIS,
    n_fft: int = None,
    fmin: float = DEFAULT_MEL_FMIN_HZ,
    fmax: float = DEFAULT_MEL_FMAX_HZ,
) -> np.ndarray:
    """Compute MFCC feature matrix for a speech signal."""
    if signal_samples.size == 0:
        return np.empty((0, num_ceps))

    emphasized = apply_pre_emphasis(signal_samples, pre_emphasis)

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
    pre_emphasis: float = DEFAULT_PRE_EMPHASIS,
) -> np.ndarray:
    """Compute magnitude spectrogram of a signal."""
    frame_size = int(fs * frame_ms / 1000)
    hop_size = int(fs * hop_ms / 1000)
    emphasized = apply_pre_emphasis(signal_samples, pre_emphasis)
    if emphasized.size < frame_size:
        return np.empty((0, frame_size // 2 + 1))

    n = np.arange(frame_size)
    hamming = 0.54 - 0.46 * np.cos(np.pi * n / frame_size)
    _, _, zxx = signal.stft(
        emphasized,
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
    pre_emphasis: float = DEFAULT_PRE_EMPHASIS,
) -> np.ndarray:
    """Compute Bark-band energies (bands 1..16) per frame."""
    frame_size = int(fs * frame_ms / 1000)
    hop_size = int(fs * hop_ms / 1000)
    emphasized = apply_pre_emphasis(signal_samples, pre_emphasis)
    if emphasized.size < frame_size:
        return np.empty((0, len(BARK_BAND_LIMITS_HZ)))

    num_frames = 1 + (len(emphasized) - frame_size) // hop_size
    n = np.arange(frame_size)
    hamming = 0.54 - 0.46 * np.cos(np.pi * n / frame_size)
    fft_freqs = np.fft.rfftfreq(frame_size, d=1 / fs)
    band_bins = [
        np.where((fft_freqs >= lo) & (fft_freqs < hi))[0] for lo, hi in BARK_BAND_LIMITS_HZ
    ]

    energies = np.zeros((num_frames, len(BARK_BAND_LIMITS_HZ)))
    for i in range(num_frames):
        start = i * hop_size
        frame = emphasized[start : start + frame_size].astype(float) * hamming
        mag = np.abs(np.fft.rfft(frame))
        for b, bins in enumerate(band_bins):
            if bins.size > 0:
                energies[i, b] = np.sum(mag[bins] ** 2)
    return energies


def _frame_signal(
    signal_samples: np.ndarray,
    fs: int,
    frame_ms: float,
    hop_ms: float,
    pre_emphasis: float,
) -> Tuple[np.ndarray, int]:
    """Frame a signal with Hamming windowing."""
    frame_size = int(fs * frame_ms / 1000)
    hop_size = int(fs * hop_ms / 1000)
    emphasized = apply_pre_emphasis(signal_samples, pre_emphasis)
    if emphasized.size < frame_size:
        return np.empty((0, frame_size)), frame_size
    num_frames = 1 + (len(emphasized) - frame_size) // hop_size
    frames = np.stack([
        emphasized[i * hop_size : i * hop_size + frame_size] for i in range(num_frames)
    ])
    n = np.arange(frame_size)
    hamming = 0.54 - 0.46 * np.cos(np.pi * n / frame_size)
    frames *= hamming
    return frames, frame_size


def compute_cepstrogram(
    signal_samples: np.ndarray,
    fs: int,
    frame_ms: float = DEFAULT_FRAME_MS,
    hop_ms: float = DEFAULT_HOP_MS,
    pre_emphasis: float = DEFAULT_PRE_EMPHASIS,
    n_fft: int = None,
    max_quefrency_ms: float = 20.0,
) -> np.ndarray:
    """Compute frame-wise real cepstrum (cepstrogram)."""
    frames, frame_size = _frame_signal(signal_samples, fs, frame_ms, hop_ms, pre_emphasis)
    if frames.size == 0:
        return np.empty((0, 0))
    if n_fft is None:
        n_fft = frame_size
    mag = np.abs(np.fft.rfft(frames, n=n_fft))
    log_mag = np.log(mag + 1e-12)
    cep = np.fft.irfft(log_mag, n=n_fft, axis=1)
    max_q = min(int(fs * max_quefrency_ms / 1000.0), cep.shape[1])
    return cep[:, :max_q]


def _levinson_durbin(r: np.ndarray, order: int) -> np.ndarray:
    """Levinson-Durbin recursion returning LPC a[1..order]."""
    a = np.zeros(order + 1)
    e = float(r[0]) if r[0] > 1e-12 else 1e-12
    a[0] = 1.0
    for i in range(1, order + 1):
        acc = 0.0
        for j in range(1, i):
            acc += a[j] * r[i - j]
        k = (r[i] - acc) / e
        a_prev = a.copy()
        a[i] = k
        for j in range(1, i):
            a[j] = a_prev[j] - k * a_prev[i - j]
        e *= max(1e-12, (1.0 - k * k))
    return a[1:]


def compute_lpc_features(
    signal_samples: np.ndarray,
    fs: int,
    order: int = DEFAULT_LPC_ORDER,
    frame_ms: float = DEFAULT_FRAME_MS,
    hop_ms: float = DEFAULT_HOP_MS,
    pre_emphasis: float = DEFAULT_PRE_EMPHASIS,
) -> np.ndarray:
    """Compute frame-wise LPC coefficient features."""
    frames, frame_size = _frame_signal(signal_samples, fs, frame_ms, hop_ms, pre_emphasis)
    if frames.size == 0:
        return np.empty((0, order))
    order = min(order, frame_size - 1)
    feats = np.zeros((frames.shape[0], order))
    for i, frame in enumerate(frames):
        r = np.correlate(frame, frame, mode="full")
        mid = len(r) // 2
        ac = r[mid : mid + order + 1]
        feats[i, :] = _levinson_durbin(ac, order)
    return feats


def compute_wavelet_features(
    signal_samples: np.ndarray,
    fs: int,
    min_scale: int = DEFAULT_WAVELET_MIN_SCALE,
    max_scale: int = DEFAULT_WAVELET_MAX_SCALE,
    frame_ms: float = DEFAULT_FRAME_MS,
    hop_ms: float = DEFAULT_HOP_MS,
    pre_emphasis: float = DEFAULT_PRE_EMPHASIS,
) -> np.ndarray:
    """Compute frame-wise wavelet energy features over selected scales."""
    frames, _ = _frame_signal(signal_samples, fs, frame_ms, hop_ms, pre_emphasis)
    if frames.size == 0:
        return np.empty((0, max(0, max_scale - min_scale + 1)))
    min_scale = max(1, int(min_scale))
    max_scale = max(min_scale, int(max_scale))
    scales = np.arange(min_scale, max_scale + 1)
    feats = np.zeros((frames.shape[0], len(scales)))

    def ricker_wavelet(scale: int) -> np.ndarray:
        # Width proportional to scale; odd length keeps symmetric center.
        length = int(max(8 * scale + 1, 17))
        if length % 2 == 0:
            length += 1
        t = np.arange(length) - (length // 2)
        a = float(scale)
        ts = (t / a) ** 2
        wave = (1.0 - ts) * np.exp(-0.5 * ts)
        # Normalize energy to keep scale energies comparable.
        norm = np.sqrt(np.sum(wave ** 2)) + 1e-12
        return wave / norm

    for i, frame in enumerate(frames):
        scale_energies = []
        for scale in scales:
            wave = ricker_wavelet(int(scale))
            coeff = np.convolve(frame, wave, mode="same")
            scale_energies.append(float(np.sum(coeff ** 2)))
        feats[i, :] = np.log(np.array(scale_energies) + 1e-12)
    return feats


def synthesize_lpc_speech(
    signal_samples: np.ndarray,
    fs: int,
    order: int = DEFAULT_LPC_ORDER,
    frame_ms: float = DEFAULT_FRAME_MS,
    hop_ms: float = DEFAULT_HOP_MS,
    pre_emphasis: float = DEFAULT_PRE_EMPHASIS,
) -> np.ndarray:
    """Re-synthesize speech using frame-wise LPC analysis and overlap-add."""
    frame_size = int(fs * frame_ms / 1000)
    hop_size = int(fs * hop_ms / 1000)
    emphasized = apply_pre_emphasis(signal_samples, pre_emphasis)
    if emphasized.size < frame_size:
        return np.array([], dtype=float)

    num_frames = 1 + (len(emphasized) - frame_size) // hop_size
    out_len = (num_frames - 1) * hop_size + frame_size
    y_acc = np.zeros(out_len, dtype=float)
    w_acc = np.zeros(out_len, dtype=float)
    n = np.arange(frame_size)
    win = 0.54 - 0.46 * np.cos(np.pi * n / frame_size)
    order = min(order, frame_size - 1)

    for i in range(num_frames):
        start = i * hop_size
        frame = emphasized[start : start + frame_size].astype(float) * win
        r = np.correlate(frame, frame, mode="full")
        mid = len(r) // 2
        ac = r[mid : mid + order + 1]
        a = _levinson_durbin(ac, order)

        # Residual e[n] = x[n] - sum_k a_k x[n-k]
        e = signal.lfilter(np.concatenate(([1.0], -a)), [1.0], frame)
        # All-pole synthesis x_hat[n] = 1/A(z) * e[n], A(z)=1-sum_k a_k z^-k
        y_frame = signal.lfilter([1.0], np.concatenate(([1.0], -a)), e)

        y_acc[start : start + frame_size] += y_frame * win
        w_acc[start : start + frame_size] += win ** 2

    y = y_acc / (w_acc + 1e-12)
    # De-emphasis inverse filter: y[n] = x[n] + a*y[n-1]
    if pre_emphasis > 0:
        y = signal.lfilter([1.0], [1.0, -pre_emphasis], y)
    return y


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
