"""PyQt6 UI for the speech processing demo app."""

import sys
from typing import List, Tuple

import numpy as np

try:
    from PyQt6 import QtWidgets
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
    )
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
except ImportError:
    QtWidgets = None
    QApplication = None
    QMainWindow = object
    QWidget = object
    QPushButton = object
    QHBoxLayout = object
    QVBoxLayout = object
    QComboBox = object
    QLabel = object
    QInputDialog = object
    QMessageBox = object
    FigureCanvas = None
    Figure = None

from speech_dsp import (
    DEFAULT_FS,
    DEFAULT_FRAME_MS,
    DEFAULT_HOP_MS,
    DEFAULT_MEL_FILTERS,
    DEFAULT_MEL_FMIN_HZ,
    DEFAULT_MEL_FMAX_HZ,
    compute_bark_band_energies,
    compute_mfcc,
    compute_spectrogram,
    detect_segments,
)
from dictionary_store import (
    append_entry,
    build_entry,
    find_best_match_label,
    load_dictionary,
)
from audio_io import AudioIOError, is_available, play_audio as play_audio_buffer, record_audio, stop_playback


class SpeechRecognitionApp(QMainWindow):
    """Main application window for the speech recognition framework."""

    def __init__(self):
        super().__init__()
        if QtWidgets is None or FigureCanvas is None:
            raise ImportError("PyQt6 and matplotlib are required to run this application.")
        self.setWindowTitle("Speech Recognition Framework")
        self.fs = DEFAULT_FS
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

        self.figure = Figure(figsize=(6, 3))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Waveform")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Amplitude")

        controls_layout = QHBoxLayout()
        layout.addLayout(controls_layout)

        self.record_btn = QPushButton("Record")
        self.record_btn.clicked.connect(self.record_audio)
        controls_layout.addWidget(self.record_btn)

        self.segment_btn = QPushButton("Segment")
        self.segment_btn.clicked.connect(self.segment_audio)
        controls_layout.addWidget(self.segment_btn)

        self.clear_segments_btn = QPushButton("Clear Segments")
        self.clear_segments_btn.clicked.connect(self.clear_segments)
        controls_layout.addWidget(self.clear_segments_btn)

        self.method_combo = QComboBox()
        self.method_combo.addItems(["energy", "zcr", "energy_zcr", "entropy"])
        controls_layout.addWidget(QLabel("Method:"))
        controls_layout.addWidget(self.method_combo)

        self.feature_combo = QComboBox()
        self.feature_combo.addItems(["mfcc", "bark"])
        controls_layout.addWidget(QLabel("Features:"))
        controls_layout.addWidget(self.feature_combo)

        self.save_btn = QPushButton("Save Segment")
        self.save_btn.clicked.connect(self.save_segment)
        controls_layout.addWidget(self.save_btn)

        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.play_audio)
        controls_layout.addWidget(self.play_btn)

        self.compare_btn = QPushButton("Compare")
        self.compare_btn.clicked.connect(self.compare_audio)
        controls_layout.addWidget(self.compare_btn)

        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

    def record_audio(self):
        """Record audio from the microphone and plot the waveform."""
        if not is_available():
            QMessageBox.critical(self, "Missing Dependency", "sounddevice is not installed.")
            return
        try:
            self.status_label.setText("Recording...")
            self.repaint()
            self.audio_data = record_audio(self.fs, self.record_duration)
            self.status_label.setText(f"Recorded {len(self.audio_data)/self.fs:.2f} seconds of audio.")
            self.plot_waveform()
        except AudioIOError as exc:
            QMessageBox.critical(self, "Recording Error", str(exc))
        except Exception as exc:
            QMessageBox.critical(self, "Recording Error", str(exc))

    def play_audio(self):
        """Play back the recorded audio or a selected segment."""
        if not is_available():
            QMessageBox.critical(self, "Missing Dependency", "sounddevice is not installed.")
            return
        if self.audio_data.size == 0:
            QMessageBox.warning(self, "No audio", "Please record audio before playing.")
            return

        data_to_play: np.ndarray
        if self.segments:
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

        try:
            try:
                stop_playback()
            except Exception:
                pass
            play_audio_buffer(data_to_play, self.fs, blocking=False)
        except AudioIOError as exc:
            QMessageBox.critical(self, "Playback Error", str(exc))
        except Exception as exc:
            QMessageBox.critical(self, "Playback Error", str(exc))

    def plot_waveform(self):
        """Plot the current audio waveform and segmentation marks."""
        self.ax.clear()
        if self.audio_data.size == 0:
            self.ax.set_title("No audio recorded")
        else:
            t = np.arange(len(self.audio_data)) / self.fs
            self.ax.plot(t, self.audio_data, color="blue")
            self.ax.set_title("Waveform")
            self.ax.set_xlabel("Time (s)")
            self.ax.set_ylabel("Amplitude")
            for start, end in self.segments:
                start_t = start / self.fs
                end_t = end / self.fs
                self.ax.axvspan(start_t, end_t, facecolor="orange", alpha=0.2)
                self.ax.axvline(start_t, color="red", linestyle="--")
                self.ax.axvline(end_t, color="green", linestyle="--")
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
        except Exception as exc:
            QMessageBox.critical(self, "Segmentation Error", str(exc))

    def clear_segments(self):
        """Clear current segmentation markers from the waveform."""
        self.segments = []
        self.status_label.setText("Segmentation cleared.")
        self.plot_waveform()

    def save_segment(self):
        """Save a selected segment with label and extracted features."""
        if self.audio_data.size == 0 or not self.segments:
            QMessageBox.warning(self, "No segment", "Record and segment audio before saving.")
            return

        segment_items = [
            f"Segment {i+1}: {start/self.fs:.2f}s–{end/self.fs:.2f}s"
            for i, (start, end) in enumerate(self.segments)
        ]
        segment_index, ok = QInputDialog.getItem(
            self, "Select Segment", "Choose a segment to save:", segment_items, 0, False
        )
        if not ok:
            return

        selected_idx = segment_items.index(segment_index)
        start, end = self.segments[selected_idx]

        label, ok = QInputDialog.getText(self, "Label", "Enter the word label:")
        if not ok or not label:
            return

        segment_samples = self.audio_data[start:end]
        spect = compute_spectrogram(
            segment_samples,
            self.fs,
            frame_ms=DEFAULT_FRAME_MS,
            hop_ms=DEFAULT_HOP_MS,
        )
        feature_type = self.feature_combo.currentText()
        if feature_type == "mfcc":
            feature_matrix = compute_mfcc(
                segment_samples,
                self.fs,
                num_filters=DEFAULT_MEL_FILTERS,
                num_ceps=DEFAULT_MEL_FILTERS,
                frame_ms=DEFAULT_FRAME_MS,
                hop_ms=DEFAULT_HOP_MS,
                fmin=DEFAULT_MEL_FMIN_HZ,
                fmax=DEFAULT_MEL_FMAX_HZ,
            )
        else:
            feature_matrix = compute_bark_band_energies(
                segment_samples,
                self.fs,
                frame_ms=DEFAULT_FRAME_MS,
                hop_ms=DEFAULT_HOP_MS,
            )
        if feature_matrix.size == 0:
            QMessageBox.information(
                self,
                "No features",
                f"Could not extract {feature_type} features from the selected segment.",
            )
            return

        entry = build_entry(
            label=label,
            fs=self.fs,
            segment_seconds=(start / self.fs, end / self.fs),
            spectrogram=spect,
            feature_type=feature_type,
            feature_matrix=feature_matrix,
            frame_ms=DEFAULT_FRAME_MS,
            hop_ms=DEFAULT_HOP_MS,
            mel_filters=DEFAULT_MEL_FILTERS,
            mel_fmin_hz=DEFAULT_MEL_FMIN_HZ,
            mel_fmax_hz=DEFAULT_MEL_FMAX_HZ,
        )
        append_entry(self.dictionary_file, entry)

        QMessageBox.information(self, "Saved", f"Segment saved under label '{label}'.")

    def compare_audio(self):
        """Record a new sample and compare against dictionary entries using DTW."""
        if not is_available():
            QMessageBox.critical(self, "Missing Dependency", "sounddevice is not installed.")
            return

        dictionary = load_dictionary(self.dictionary_file)
        if not dictionary:
            QMessageBox.warning(
                self,
                "No dictionary",
                "Dictionary is missing or empty. Save some segments first.",
            )
            return

        self.status_label.setText("Recording sample for comparison...")
        self.repaint()
        try:
            new_samples = record_audio(self.fs, self.record_duration)
        except AudioIOError as exc:
            QMessageBox.critical(self, "Recording Error", str(exc))
            return
        feature_type = self.feature_combo.currentText()
        if feature_type == "mfcc":
            query_features = compute_mfcc(
                new_samples,
                self.fs,
                num_filters=DEFAULT_MEL_FILTERS,
                num_ceps=DEFAULT_MEL_FILTERS,
                frame_ms=DEFAULT_FRAME_MS,
                hop_ms=DEFAULT_HOP_MS,
                fmin=DEFAULT_MEL_FMIN_HZ,
                fmax=DEFAULT_MEL_FMAX_HZ,
            )
        else:
            query_features = compute_bark_band_energies(
                new_samples,
                self.fs,
                frame_ms=DEFAULT_FRAME_MS,
                hop_ms=DEFAULT_HOP_MS,
            )
        if query_features.size == 0:
            QMessageBox.information(
                self,
                "No features",
                f"Could not extract {feature_type} features from the recording.",
            )
            return

        best_label, best_distance = find_best_match_label(
            dictionary,
            query_features,
            feature_type,
        )

        if best_label is None:
            QMessageBox.information(
                self,
                "No match",
                f"No comparable entries found for feature type '{feature_type}'.",
            )
        else:
            QMessageBox.information(
                self,
                "Recognition Result",
                f"Closest match: {best_label} (distance={best_distance:.3f})",
            )


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
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
