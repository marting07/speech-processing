"""
Speech Recognition Framework Using PyQt6
--------------------------------------

This module implements a simple desktop application for recording speech,
segmenting the recorded signal and saving feature representations for
subsequent recognition.  The application uses PyQt6 for the graphical
interface and embeds a matplotlib canvas to display the waveform.  It can
segment the speech based on several classic voice‐activity detection
criteria: short‑time energy, zero crossing rate, spectral entropy or a
combination of energy and zero crossings.  The segmentation methods draw on
well‑known techniques in the speech processing literature.  For example,
energy and zero crossing based detectors are effective in quiet
environments【661169263223060†L138-L143】, while spectral entropy can improve
robustness in noise【863035641302766†L9-L15】.  After segmentation, the user
may label each word and save its spectrogram and MFCC representation to a
JSON dictionary.  New recordings can then be compared against this
dictionary using Dynamic Time Warping (DTW) to find the closest match.

Requirements
~~~~~~~~~~~~
This application depends on several third‑party libraries that may not
already be installed on your system.  To run the program you should have
the following packages available:

* PyQt6 (for the GUI components)
* matplotlib (for waveform display)
* numpy and scipy (for numerical computations and signal processing)
* sounddevice (for recording audio from the microphone)

If these packages are missing, install them via ``pip``::

    pip install PyQt6 matplotlib numpy scipy sounddevice

Usage
~~~~~
Run the script directly with Python.  When the window opens, press
``Record`` to capture a short utterance (default 3 seconds), then press
``Segment`` to detect speech segments using the selected method.  You can
save each segment to the dictionary with the ``Save Segment`` button and
compare new recordings with ``Compare``.
"""

import json
import sys
import threading
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
from scipy import signal, fftpack

try:
    import sounddevice as sd
except ImportError:
    sd = None  # sounddevice may not be installed – record functions will fail

try:
    # Attempt to import PyQt6 and required classes.  If this fails the GUI
    # component of the module will not be available, but the computational
    # functions remain usable for testing or console applications.
    from PyQt6 import QtWidgets, QtCore
    from PyQt6.QtWidgets import (
        QApplication,
        QMainWindow,
        QWidget,
        QPushButton,
        QHBoxLayout,
        QVBoxLayout,
        QComboBox,
        QLabel,
        QInputDialog,
        QMessageBox,
        QFileDialog,
    )
    # Matplotlib Qt backend for Qt6.  FigureCanvasQTAgg integrates the Agg
    # renderer with the Qt event loop for PyQt6【170087996701686†L123-L130】.
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
except ImportError:
    # If PyQt6 cannot be imported, define placeholders so that import of this
    # module does not fail.  GUI functionality will be disabled.
    QtWidgets = None
    QApplication = None
    QMainWindow = object  # simple base class to prevent NameError
    QWidget = object
    QPushButton = object
    QHBoxLayout = object
    QVBoxLayout = object
    QComboBox = object
    QLabel = object
    QInputDialog = object
    QMessageBox = object
    QFileDialog = object
    FigureCanvas = None
    Figure = None


def compute_short_time_energy(
    signal_samples: np.ndarray, frame_size: int, hop_size: int
) -> np.ndarray:
    """Compute the short‑time energy of a 1‑D signal.

    Parameters
    ----------
    signal_samples : np.ndarray
        The audio samples.
    frame_size : int
        The length of each frame in samples.
    hop_size : int
        The step between successive frames in samples.

    Returns
    -------
    energy : np.ndarray
        An array containing the energy for each frame.
    """
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
    """Compute the zero crossing rate (ZCR) for each frame of a signal.

    ZCR counts how often the waveform crosses the zero axis within a frame
    【661169263223060†L194-L201】.  High rates typically indicate unvoiced
    consonants or noise, whereas low rates correspond to voiced speech.

    Parameters
    ----------
    signal_samples : np.ndarray
        The audio samples.
    frame_size : int
        The length of each frame in samples.
    hop_size : int
        The step between successive frames in samples.

    Returns
    -------
    zcr : np.ndarray
        Zero crossing counts per frame.
    """
    num_frames = 1 + (len(signal_samples) - frame_size) // hop_size
    zcr = np.empty(num_frames)
    # sign of samples (0 is considered positive for stability)
    signs = np.sign(signal_samples)
    signs[signs == 0] = 1
    for i in range(num_frames):
        start = i * hop_size
        frame_sign = signs[start : start + frame_size]
        # count the number of sign changes
        zc = np.sum(np.abs(np.diff(frame_sign)) > 1e-10)
        zcr[i] = zc / frame_size
    return zcr


def compute_spectral_entropy(
    signal_samples: np.ndarray, frame_size: int, hop_size: int
) -> np.ndarray:
    """Compute the spectral entropy for each frame of a signal.

    Spectral entropy measures the randomness of the spectrum within a frame.
    It has been used as an alternative to energy for robust endpoint detection
    【863035641302766†L9-L15】.  Higher entropy indicates more noise‑like content,
    whereas lower entropy corresponds to structured, voiced regions.

    Parameters
    ----------
    signal_samples : np.ndarray
        Audio samples.
    frame_size : int
        Frame length in samples.
    hop_size : int
        Hop length in samples.

    Returns
    -------
    entropy : np.ndarray
        Spectral entropy per frame.
    """
    num_frames = 1 + (len(signal_samples) - frame_size) // hop_size
    entropy = np.empty(num_frames)
    window = np.hamming(frame_size)
    for i in range(num_frames):
        start = i * hop_size
        frame = signal_samples[start : start + frame_size] * window
        # FFT magnitude squared
        spectrum = np.abs(np.fft.rfft(frame)) ** 2
        # Normalize to form probability distribution
        psd = spectrum / np.sum(spectrum + 1e-12)
        # Shannon entropy
        entropy[i] = -np.sum(psd * np.log2(psd + 1e-12))
    return entropy


def detect_segments(
    signal_samples: np.ndarray,
    fs: int,
    method: str = "energy",
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    energy_threshold_ratio: float = 0.1,
    zcr_threshold_ratio: float = 0.1,
    entropy_threshold_ratio: float = 0.1,
) -> List[Tuple[int, int]]:
    """Detect speech segments based on the chosen method.

    The function partitions the audio into frames and computes a feature
    sequence.  Frames exceeding a threshold are marked as speech.  Adjacent
    speech frames are merged into segments and converted back to sample
    indices.

    Parameters
    ----------
    signal_samples : np.ndarray
        Audio samples.
    fs : int
        Sampling frequency.
    method : str, optional
        One of ``'energy'``, ``'zcr'``, ``'energy_zcr'`` or ``'entropy'``.
    frame_ms : float, optional
        Frame length in milliseconds.
    hop_ms : float, optional
        Hop length in milliseconds.
    energy_threshold_ratio : float, optional
        Fraction of the maximum energy used as the threshold.
    zcr_threshold_ratio : float, optional
        Fraction of the maximum ZCR used as the threshold.
    entropy_threshold_ratio : float, optional
        Fraction of the maximum entropy used as the threshold.

    Returns
    -------
    segments : list of (start_sample, end_sample)
        Detected speech segments.
    """
    frame_size = int(fs * frame_ms / 1000)
    hop_size = int(fs * hop_ms / 1000)
    # avoid invalid lengths
    if len(signal_samples) < frame_size:
        return []

    if method == "energy":
        feature = compute_short_time_energy(signal_samples, frame_size, hop_size)
        threshold = energy_threshold_ratio * np.max(feature)
        mask = feature > threshold
    elif method == "zcr":
        feature = compute_zero_crossing_rate(signal_samples, frame_size, hop_size)
        threshold = zcr_threshold_ratio * np.max(feature)
        mask = feature > threshold
    elif method == "energy_zcr":
        energy = compute_short_time_energy(signal_samples, frame_size, hop_size)
        zcr = compute_zero_crossing_rate(signal_samples, frame_size, hop_size)
        energy_th = energy_threshold_ratio * np.max(energy)
        zcr_th = zcr_threshold_ratio * np.max(zcr)
        mask = (energy > energy_th) & (zcr > zcr_th)
    elif method == "entropy":
        feature = compute_spectral_entropy(signal_samples, frame_size, hop_size)
        threshold = entropy_threshold_ratio * np.max(feature)
        mask = feature > threshold
    else:
        raise ValueError(f"Unknown segmentation method: {method}")

    # Merge consecutive speech frames
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
            # convert to sample indices
            start_sample = start_frame * hop_size
            end_sample = min(len(signal_samples), end_frame * hop_size + frame_size)
            segments.append((start_sample, end_sample))
    # Check if last segment reaches the end
    if in_seg:
        end_frame = len(mask)
        start_sample = start_frame * hop_size
        end_sample = len(signal_samples)
        segments.append((start_sample, end_sample))
    return segments


def compute_mel_filterbank(
    n_filters: int, n_fft: int, fs: int, fmin: float = 0.0, fmax: float = None
) -> np.ndarray:
    """Create a Mel filter bank matrix.

    Parameters
    ----------
    n_filters : int
        Number of Mel filters.
    n_fft : int
        FFT length.
    fs : int
        Sampling frequency.
    fmin : float
        Minimum frequency.
    fmax : float, optional
        Maximum frequency.  Defaults to Nyquist.

    Returns
    -------
    filterbank : ndarray of shape (n_filters, n_fft/2+1)
        Mel filter bank.
    """
    if fmax is None:
        fmax = fs / 2
    # convert to Mel scale
    def hz_to_mel(f: float) -> float:
        return 2595 * np.log10(1 + f / 700)

    def mel_to_hz(mel: float) -> float:
        return 700 * (10 ** (mel / 2595) - 1)

    # compute points equally spaced in Mel scale
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
        # rising slope
        for k in range(left, center):
            filterbank[i - 1, k] = (k - left) / (center - left + 1e-12)
        # falling slope
        for k in range(center, right):
            filterbank[i - 1, k] = (right - k) / (right - center + 1e-12)
    return filterbank


def compute_mfcc(
    signal_samples: np.ndarray,
    fs: int,
    num_filters: int = 26,
    num_ceps: int = 13,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    pre_emphasis: float = 0.97,
    n_fft: int = 512,
) -> np.ndarray:
    """Compute Mel–frequency cepstral coefficients for a speech signal.

    This implementation follows the standard pipeline described in the
    literature【782333128503880†L233-L242】【152220305696759†L218-L229】:
    pre‑emphasis, framing, windowing, power spectrum, filter bank and DCT.

    Parameters
    ----------
    signal_samples : ndarray
        Audio samples (1‑D).
    fs : int
        Sampling frequency.
    num_filters : int
        Number of Mel filters.
    num_ceps : int
        Number of cepstral coefficients to retain.
    frame_ms : float
        Frame length in milliseconds.
    hop_ms : float
        Hop length in milliseconds.
    pre_emphasis : float
        Pre‑emphasis filter coefficient.
    n_fft : int
        FFT size.

    Returns
    -------
    mfccs : ndarray of shape (n_frames, num_ceps)
        Computed MFCC feature matrix.
    """
    # Pre‑emphasis
    emphasized = np.append(signal_samples[0], signal_samples[1:] - pre_emphasis * signal_samples[:-1])
    frame_size = int(fs * frame_ms / 1000)
    hop_size = int(fs * hop_ms / 1000)
    # number of frames
    if len(emphasized) < frame_size:
        return np.empty((0, num_ceps))
    num_frames = 1 + (len(emphasized) - frame_size) // hop_size
    # create frames
    frames = np.stack([
        emphasized[i * hop_size : i * hop_size + frame_size] for i in range(num_frames)
    ])
    # apply Hamming window
    frames *= np.hamming(frame_size)
    # compute magnitude spectrum
    mag_frames = np.abs(np.fft.rfft(frames, n_fft))
    pow_frames = (1.0 / n_fft) * (mag_frames ** 2)
    # mel filter bank
    filterbank = compute_mel_filterbank(num_filters, n_fft, fs)
    filterbank_energies = np.dot(pow_frames, filterbank.T)
    # avoid log of zero
    filterbank_energies[filterbank_energies == 0] = 1e-12
    log_energies = np.log(filterbank_energies)
    # Discrete cosine transform
    mfcc = fftpack.dct(log_energies, type=2, norm='ortho', axis=1)[:, :num_ceps]
    return mfcc


def compute_spectrogram(
    signal_samples: np.ndarray, fs: int, n_fft: int = 512, hop_length: int = 256
) -> np.ndarray:
    """Compute magnitude spectrogram of a signal.

    Parameters
    ----------
    signal_samples : ndarray
        Audio samples.
    fs : int
        Sampling rate.
    n_fft : int
        FFT size.
    hop_length : int
        Hop length between successive STFT windows.

    Returns
    -------
    spectrogram : ndarray (n_frames x (n_fft/2+1))
        Magnitude spectrogram.
    """
    f, t, Zxx = signal.stft(signal_samples, fs=fs, nperseg=n_fft, noverlap=n_fft - hop_length)
    return np.abs(Zxx).T  # transpose to shape (n_frames, n_freq_bins)


def dtw_distance(seq1: np.ndarray, seq2: np.ndarray) -> float:
    """Compute the Dynamic Time Warping (DTW) distance between two feature matrices.

    DTW aligns two time‑series by warping the time axis to minimize the total
    distance【801812593074164†L60-L88】.  Here we compute a standard DTW cost
    matrix using Euclidean distances between MFCC vectors.

    Parameters
    ----------
    seq1 : ndarray (n1 x d)
        First feature sequence.
    seq2 : ndarray (n2 x d)
        Second feature sequence.

    Returns
    -------
    dist : float
        Normalized DTW distance (cost divided by path length).
    """
    n1, d1 = seq1.shape
    n2, d2 = seq2.shape
    if d1 != d2:
        raise ValueError("Feature dimensions do not match")
    # initialize cost matrix with infinity
    cost = np.full((n1 + 1, n2 + 1), np.inf)
    cost[0, 0] = 0.0
    # compute cumulative cost
    for i in range(1, n1 + 1):
        for j in range(1, n2 + 1):
            dist = np.linalg.norm(seq1[i - 1] - seq2[j - 1])
            cost[i, j] = dist + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    # backtrack to compute path length
    i, j = n1, n2
    path_len = 0
    while (i > 0) and (j > 0):
        path_len += 1
        # move to the predecessor with minimal cost
        directions = [cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1]]
        idx = np.argmin(directions)
        if idx == 0:
            i -= 1
        elif idx == 1:
            j -= 1
        else:
            i -= 1
            j -= 1
    # handle remaining edges
    path_len += (i + j)
    return cost[n1, n2] / path_len


@dataclass
class SegmentFeature:
    label: str
    spectrogram: List[List[float]]
    mfcc: List[List[float]]


class SpeechRecognitionApp(QMainWindow):
    """Main application window for the speech recognition framework."""

    def __init__(self):
        super().__init__()
        if QtWidgets is None or FigureCanvas is None:
            raise ImportError("PyQt6 and matplotlib are required to run this application.")
        self.setWindowTitle("Speech Recognition Framework")
        self.fs = 16000  # default sampling rate
        self.record_duration = 3.0  # seconds
        self.audio_data: np.ndarray = np.array([])
        self.segments: List[Tuple[int, int]] = []
        self.dictionary_file = "speech_dictionary.json"
        self.init_ui()

    def init_ui(self):
        """Initialize all GUI components."""
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Canvas for plotting
        self.figure = Figure(figsize=(6, 3))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Waveform")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Amplitude")

        # Controls
        controls_layout = QHBoxLayout()
        layout.addLayout(controls_layout)
        # Record button
        self.record_btn = QPushButton("Record")
        self.record_btn.clicked.connect(self.record_audio)
        controls_layout.addWidget(self.record_btn)
        # Segment button
        self.segment_btn = QPushButton("Segment")
        self.segment_btn.clicked.connect(self.segment_audio)
        controls_layout.addWidget(self.segment_btn)
        # Combo box for method selection
        self.method_combo = QComboBox()
        self.method_combo.addItems(["energy", "zcr", "energy_zcr", "entropy"])
        controls_layout.addWidget(QLabel("Method:"))
        controls_layout.addWidget(self.method_combo)
        # Save segment button
        self.save_btn = QPushButton("Save Segment")
        self.save_btn.clicked.connect(self.save_segment)
        controls_layout.addWidget(self.save_btn)
        # Play button
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.play_audio)
        controls_layout.addWidget(self.play_btn)
        # Compare button
        self.compare_btn = QPushButton("Compare")
        self.compare_btn.clicked.connect(self.compare_audio)
        controls_layout.addWidget(self.compare_btn)

        # Status label
        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

    def record_audio(self):
        """Record audio from the microphone and plot the waveform."""
        if sd is None:
            QMessageBox.critical(self, "Missing Dependency", "sounddevice is not installed.")
            return
        try:
            self.status_label.setText("Recording...")
            self.repaint()
            # blocking recording
            frames = int(self.record_duration * self.fs)
            recording = sd.rec(frames, samplerate=self.fs, channels=1, dtype=np.float32)
            sd.wait()
            self.audio_data = recording.flatten()
            self.status_label.setText(f"Recorded {len(self.audio_data)/self.fs:.2f} seconds of audio.")
            self.plot_waveform()
        except Exception as e:
            QMessageBox.critical(self, "Recording Error", str(e))

    def play_audio(self):
        """Play back the recorded audio or a selected segment.

        If a segmentation has been performed, the user is offered the option to
        play the entire recording or one of the detected segments.  Playback
        uses the sounddevice library's ``play`` function, which accepts a
        NumPy array and a sampling rate and plays back the signal on the
        default output device【574514951566577†L41-L60】.  Any currently
        running playback is stopped before starting a new one.
        """
        if sd is None:
            QMessageBox.critical(self, "Missing Dependency", "sounddevice is not installed.")
            return
        if self.audio_data.size == 0:
            QMessageBox.warning(self, "No audio", "Please record audio before playing.")
            return
        # Determine which portion to play
        data_to_play: np.ndarray
        # If segments exist, allow the user to choose
        if self.segments:
            # build list with "Whole recording" option and each segment labeled
            items = ["Whole recording"] + [
                f"Segment {i+1}: {start/self.fs:.2f}s–{end/self.fs:.2f}s"
                for i, (start, end) in enumerate(self.segments)
            ]
            selection, ok = QInputDialog.getItem(
                self,
                "Play Audio",
                "Select what to play:",
                items,
                0,
                False,
            )
            if not ok:
                return
            if selection and selection != "Whole recording":
                seg_idx = items.index(selection) - 1
                start, end = self.segments[seg_idx]
                data_to_play = self.audio_data[start:end]
            else:
                data_to_play = self.audio_data
        else:
            data_to_play = self.audio_data
        # play the selected audio
        try:
            # stop any existing playback before starting a new one
            try:
                sd.stop()
            except Exception:
                pass
            # use blocking playback so the user cannot start another play until finished
            sd.play(data_to_play, samplerate=self.fs, blocking=False)
        except Exception as e:
            QMessageBox.critical(self, "Playback Error", str(e))

    def plot_waveform(self):
        """Plot the current audio waveform and segmentation marks."""
        self.ax.clear()
        if self.audio_data.size == 0:
            self.ax.set_title("No audio recorded")
        else:
            # time axis for the waveform
            t = np.arange(len(self.audio_data)) / self.fs
            self.ax.plot(t, self.audio_data, color='blue')
            self.ax.set_title("Waveform")
            self.ax.set_xlabel("Time (s)")
            self.ax.set_ylabel("Amplitude")
            # draw segmentation lines and highlight each segment with a shaded band
            for start, end in self.segments:
                start_t = start / self.fs
                end_t = end / self.fs
                # shaded region to pair the two vertical lines for clarity
                self.ax.axvspan(start_t, end_t, facecolor='orange', alpha=0.2)
                # draw a pair of vertical lines at the segment boundaries
                self.ax.axvline(start_t, color='red', linestyle='--')
                self.ax.axvline(end_t, color='green', linestyle='--')
        # redraw the canvas with updated plot
        self.canvas.draw()

    def segment_audio(self):
        """Run segmentation on the recorded audio using the selected method."""
        if self.audio_data.size == 0:
            QMessageBox.warning(self, "No audio", "Please record audio before segmenting.")
            return
        method = self.method_combo.currentText()
        try:
            self.segments = detect_segments(self.audio_data, self.fs, method=method)
            if not self.segments:
                QMessageBox.information(self, "Segmentation", "No segments detected.")
            else:
                self.status_label.setText(f"Detected {len(self.segments)} segment(s).")
            self.plot_waveform()
        except Exception as e:
            QMessageBox.critical(self, "Segmentation Error", str(e))

    def save_segment(self):
        """Save the first detected segment with a label to the dictionary file."""
        if self.audio_data.size == 0 or not self.segments:
            QMessageBox.warning(self, "No segment", "Record and segment audio before saving.")
            return
        # ask which segment to save
        segment_items = [f"Segment {i+1}: {start/self.fs:.2f}s–{end/self.fs:.2f}s" for i, (start, end) in enumerate(self.segments)]
        segment_index, ok = QInputDialog.getItem(
            self, "Select Segment", "Choose a segment to save:", segment_items, 0, False
        )
        if not ok:
            return
        selected_idx = segment_items.index(segment_index)
        start, end = self.segments[selected_idx]
        # ask for label
        label, ok = QInputDialog.getText(self, "Label", "Enter the word label:")
        if not ok or not label:
            return
        # extract segment samples
        segment_samples = self.audio_data[start:end]
        # compute features
        spect = compute_spectrogram(segment_samples, self.fs)
        mfcc_feat = compute_mfcc(segment_samples, self.fs)
        # build entry
        entry = {
            "label": label,
            "spectrogram": spect.tolist(),
            "spectrogram_shape": [spect.shape[0], spect.shape[1]],
            "mfcc": mfcc_feat.tolist(),
            "mfcc_shape": [mfcc_feat.shape[0], mfcc_feat.shape[1]],
        }
        # load existing dictionary
        try:
            with open(self.dictionary_file, "r", encoding="utf-8") as f:
                dictionary = json.load(f)
        except FileNotFoundError:
            dictionary = []
        except json.JSONDecodeError:
            dictionary = []
        dictionary.append(entry)
        # save dictionary
        with open(self.dictionary_file, "w", encoding="utf-8") as f:
            json.dump(dictionary, f, indent=2)
        QMessageBox.information(self, "Saved", f"Segment saved under label '{label}'.")

    def compare_audio(self):
        """Record a new sample and compare against the saved dictionary using DTW."""
        if sd is None:
            QMessageBox.critical(self, "Missing Dependency", "sounddevice is not installed.")
            return
        # ensure dictionary exists
        try:
            with open(self.dictionary_file, "r", encoding="utf-8") as f:
                dictionary = json.load(f)
        except FileNotFoundError:
            QMessageBox.warning(self, "No dictionary", "No dictionary file found. Save some segments first.")
            return
        if not dictionary:
            QMessageBox.warning(self, "Empty dictionary", "Dictionary is empty. Save some segments first.")
            return
        # record new sample
        self.status_label.setText("Recording sample for comparison...")
        self.repaint()
        frames = int(self.record_duration * self.fs)
        recording = sd.rec(frames, samplerate=self.fs, channels=1, dtype=np.float32)
        sd.wait()
        new_samples = recording.flatten()
        new_mfcc = compute_mfcc(new_samples, self.fs)
        if new_mfcc.size == 0:
            QMessageBox.information(self, "No features", "Could not extract features from the recording.")
            return
        # compute distances
        best_label = None
        best_distance = None
        for entry in dictionary:
            entry_mfcc = np.array(entry["mfcc"]) if "mfcc" in entry else None
            if entry_mfcc is None or entry_mfcc.size == 0:
                continue
            try:
                dist = dtw_distance(new_mfcc, entry_mfcc)
            except Exception:
                continue
            if (best_distance is None) or (dist < best_distance):
                best_distance = dist
                best_label = entry["label"]
        if best_label is None:
            QMessageBox.information(self, "No match", "Unable to compare with dictionary entries.")
        else:
            QMessageBox.information(self, "Recognition Result", f"Closest match: {best_label} (distance={best_distance:.3f})")


def main():
    """Entry point for the GUI application."""
    if QtWidgets is None or FigureCanvas is None:
        print(
            "Error: PyQt6 and matplotlib are required to run this application."
            " Please install them with 'pip install PyQt6 matplotlib'."
        )
        sys.exit(1)
    app = QApplication(sys.argv)
    window = SpeechRecognitionApp()
    window.show()
    # PyQt6 removed the underscored exec_ method; use exec instead
    sys.exit(app.exec())


if __name__ == "__main__":
    main()