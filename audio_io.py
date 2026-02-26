"""Audio I/O adapter for recording and playback."""

from __future__ import annotations

import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sd = None


class AudioIOError(RuntimeError):
    """Raised when audio I/O cannot be performed."""


def is_available() -> bool:
    """Return True when sounddevice backend is available."""
    return sd is not None


def record_audio(fs: int, duration_seconds: float) -> np.ndarray:
    """Record mono audio and return a 1-D float32 array."""
    if sd is None:
        raise AudioIOError("sounddevice is not installed")
    frames = int(duration_seconds * fs)
    recording = sd.rec(frames, samplerate=fs, channels=1, dtype=np.float32)
    sd.wait()
    return recording.flatten()


def stop_playback() -> None:
    """Stop active playback if any."""
    if sd is None:
        raise AudioIOError("sounddevice is not installed")
    sd.stop()


def play_audio(samples: np.ndarray, fs: int, blocking: bool = False) -> None:
    """Play a mono audio buffer."""
    if sd is None:
        raise AudioIOError("sounddevice is not installed")
    sd.play(samples, samplerate=fs, blocking=blocking)
