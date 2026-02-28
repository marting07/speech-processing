"""Audio I/O adapter for recording and playback."""

from __future__ import annotations

import queue
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


class RealtimeAudioStream:
    """Microphone stream wrapper with chunk queue for real-time processing."""

    def __init__(self, fs: int, blocksize: int = 256):
        if sd is None:
            raise AudioIOError("sounddevice is not installed")
        self.fs = fs
        self.blocksize = blocksize
        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stream = None

    def _callback(self, indata, frames, time, status):
        if status:
            return
        self._queue.put(indata[:, 0].copy())

    def start(self) -> None:
        """Start microphone streaming."""
        if self._stream is not None:
            return
        self._stream = sd.InputStream(
            samplerate=self.fs,
            channels=1,
            dtype=np.float32,
            blocksize=self.blocksize,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        """Stop microphone streaming and release resources."""
        if self._stream is None:
            return
        self._stream.stop()
        self._stream.close()
        self._stream = None
        self.clear()

    def read_chunks(self, max_chunks: int = 64) -> list[np.ndarray]:
        """Read currently available chunks without blocking."""
        chunks: list[np.ndarray] = []
        for _ in range(max_chunks):
            try:
                chunks.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return chunks

    def clear(self) -> None:
        """Clear pending queued audio chunks."""
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
