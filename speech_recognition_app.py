"""PyQt6 UI for the speech processing demo app."""

import sys
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

try:
    from PyQt6 import QtWidgets
    from PyQt6.QtGui import QAction
    from PyQt6.QtCore import QTimer
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
        QDoubleSpinBox,
        QSpinBox,
        QGridLayout,
        QGroupBox,
        QScrollArea,
        QFormLayout,
        QToolButton,
        QStackedWidget,
    )
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
except ImportError:
    QtWidgets = None
    QAction = object
    QTimer = object
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
    QDoubleSpinBox = object
    QSpinBox = object
    QGridLayout = object
    QGroupBox = object
    QScrollArea = object
    QFormLayout = object
    QToolButton = object
    QStackedWidget = object
    FigureCanvas = None
    Figure = None

from speech_dsp import (
    DEFAULT_FS,
    DEFAULT_FRAME_MS,
    DEFAULT_HOP_MS,
    DEFAULT_MEL_FILTERS,
    DEFAULT_MEL_FMIN_HZ,
    DEFAULT_MEL_FMAX_HZ,
    DEFAULT_PRE_EMPHASIS,
    DEFAULT_LPC_ORDER,
    DEFAULT_WAVELET_MIN_SCALE,
    DEFAULT_WAVELET_MAX_SCALE,
    ONSET_PREROLL_MS,
    FRICATIVE_BAND_LOW_HZ,
    apply_pre_emphasis,
    compute_bark_band_energies,
    compute_cepstrogram,
    compute_energy_trajectory,
    compute_lpc_features,
    compute_mfcc,
    compute_spectrogram,
    compute_wavelet_features,
    detect_segments,
    kalman_filter_1d,
    synthesize_lpc_speech,
)
from dictionary_store import (
    append_entry,
    build_entry,
    find_best_match_label,
    group_feature_sequences_by_label,
    load_dictionary,
)
from speech_hmm import recognize_with_hmm_discrete, recognize_with_hmm_continuous
from audio_io import AudioIOError, is_available, play_audio as play_audio_buffer, record_audio, stop_playback
from audio_io import RealtimeAudioStream


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
        self.lpc_synth_audio: np.ndarray = np.array([])
        self.rt_stream = None
        self.rt_timer = QTimer(self)
        self.rt_timer.setInterval(40)
        self.rt_timer.timeout.connect(self._on_realtime_kalman_tick)
        self.rt_pending_samples = np.array([], dtype=np.float32)
        self.rt_time = []
        self.rt_measured = []
        self.rt_estimated = []
        self.rt_residuals = []
        self.rt_confidence = 0.0
        self.rt_frame_index = 0
        self.rt_energy_scale = 1e-12
        self.rt_kf_state = 0.0
        self.rt_kf_var = 1.0
        self.rt_running = False
        self.init_ui()

    def init_ui(self):
        """Initialize all GUI components with mode switching."""
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)

        self.mode_stack = QStackedWidget()
        root_layout.addWidget(self.mode_stack)

        recognition_page = QWidget()
        kalman_page = QWidget()
        self.mode_stack.addWidget(recognition_page)
        self.mode_stack.addWidget(kalman_page)

        self._build_recognition_page(recognition_page)
        self._build_kalman_page(kalman_page)

        mode_menu = self.menuBar().addMenu("Mode")
        rec_action = QAction("Speech Recognition", self)
        rec_action.triggered.connect(self._switch_to_recognition)
        mode_menu.addAction(rec_action)
        kalman_action = QAction("Kalman Speech Following", self)
        kalman_action.triggered.connect(self._switch_to_kalman)
        mode_menu.addAction(kalman_action)

    def _build_recognition_page(self, page: QWidget):
        """Build speech processing/recognition page."""
        root_layout = QVBoxLayout(page)

        self.figure = Figure(figsize=(6, 3))
        self.canvas = FigureCanvas(self.figure)
        root_layout.addWidget(self.canvas, stretch=3)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Waveform")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Amplitude")

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        root_layout.addWidget(scroll, stretch=2)
        panel = QWidget()
        scroll.setWidget(panel)
        panel_layout = QVBoxLayout(panel)

        self.record_btn = QPushButton("Record")
        self.record_btn.clicked.connect(self.record_audio)

        self.segment_btn = QPushButton("Segment")
        self.segment_btn.clicked.connect(self.segment_audio)

        self.clear_segments_btn = QPushButton("Clear Segments")
        self.clear_segments_btn.clicked.connect(self.clear_segments)

        self.method_combo = QComboBox()
        self.method_combo.addItems(["energy", "zcr", "energy_zcr", "entropy"])

        self.feature_combo = QComboBox()
        self.feature_combo.addItems(["mfcc", "bark", "lpc", "wavelet"])

        self.segment_pick_combo = QComboBox()
        self.segment_pick_combo.addItems(["manual", "auto-best"])
        self.recognition_backend_combo = QComboBox()
        self.recognition_backend_combo.addItems(["dtw", "hmm-discrete", "hmm-continuous"])

        self.save_btn = QPushButton("Save Segment")
        self.save_btn.clicked.connect(self.save_segment)

        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.play_audio)

        self.compare_btn = QPushButton("Compare")
        self.compare_btn.clicked.connect(self.compare_audio)

        self.show_spec_btn = QPushButton("Show Spectrogram")
        self.show_spec_btn.clicked.connect(self.show_spectrogram_view)

        self.show_cep_btn = QPushButton("Show Cepstrogram")
        self.show_cep_btn.clicked.connect(self.show_cepstrogram_view)

        self.show_wavelet_btn = QPushButton("Show Wavelet")
        self.show_wavelet_btn.clicked.connect(self.show_wavelet_view)

        self.lpc_synth_btn = QPushButton("Synthesize LPC")
        self.lpc_synth_btn.clicked.connect(self.synthesize_lpc_audio)

        self.play_lpc_btn = QPushButton("Play LPC")
        self.play_lpc_btn.clicked.connect(self.play_lpc_audio)

        for btn in [
            self.record_btn, self.segment_btn, self.clear_segments_btn,
            self.save_btn, self.play_btn, self.compare_btn,
            self.show_spec_btn, self.show_cep_btn, self.show_wavelet_btn,
            self.lpc_synth_btn, self.play_lpc_btn,
        ]:
            btn.setMinimumWidth(120)

        actions_group = QGroupBox("Actions")
        actions_layout = QGridLayout(actions_group)
        actions_layout.addWidget(self.record_btn, 0, 0)
        actions_layout.addWidget(self.segment_btn, 0, 1)
        actions_layout.addWidget(self.clear_segments_btn, 0, 2)
        actions_layout.addWidget(self.play_btn, 0, 3)
        actions_layout.addWidget(self.save_btn, 1, 0)
        actions_layout.addWidget(self.compare_btn, 1, 1)
        actions_layout.addWidget(self.show_spec_btn, 1, 2)
        actions_layout.addWidget(self.show_cep_btn, 1, 3)
        actions_layout.addWidget(self.show_wavelet_btn, 2, 0)
        actions_layout.addWidget(self.lpc_synth_btn, 2, 1)
        actions_layout.addWidget(self.play_lpc_btn, 2, 2)
        panel_layout.addWidget(actions_group)

        mode_group = QGroupBox("Modes")
        mode_layout = QFormLayout(mode_group)
        mode_layout.addRow("Segmentation method", self.method_combo)
        mode_layout.addRow("Feature extraction", self.feature_combo)
        mode_layout.addRow("Segment selection", self.segment_pick_combo)
        mode_layout.addRow("Recognition backend", self.recognition_backend_combo)
        panel_layout.addWidget(mode_group)

        self.advanced_toggle_btn = QToolButton()
        self.advanced_toggle_btn.setText("Show Advanced Parameters")
        self.advanced_toggle_btn.setCheckable(True)
        self.advanced_toggle_btn.toggled.connect(self.toggle_advanced_params)
        panel_layout.addWidget(self.advanced_toggle_btn)

        params_group = QGroupBox("DSP Parameters")
        self.params_group = params_group
        params_layout = QGridLayout(params_group)
        row = 0

        params_layout.addWidget(QLabel("Preset"), row, 0)
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(["Custom", "Quiet Room (Recommended)", "Noisy Room", "Fast Speech"])
        params_layout.addWidget(self.preset_combo, row, 1)
        self.apply_preset_btn = QPushButton("Apply Preset")
        self.apply_preset_btn.clicked.connect(self.apply_preset)
        params_layout.addWidget(self.apply_preset_btn, row, 2)
        row += 1

        params_layout.addWidget(QLabel("Frame ms"), row, 0)
        self.frame_ms_spin = QDoubleSpinBox()
        self.frame_ms_spin.setRange(10.0, 60.0)
        self.frame_ms_spin.setDecimals(1)
        self.frame_ms_spin.setSingleStep(1.0)
        self.frame_ms_spin.setValue(DEFAULT_FRAME_MS)
        params_layout.addWidget(self.frame_ms_spin, row, 1)
        params_layout.addWidget(QLabel("Rec: 20-40 (30 default)"), row, 2)
        row += 1

        params_layout.addWidget(QLabel("Hop ms"), row, 0)
        self.hop_ms_spin = QDoubleSpinBox()
        self.hop_ms_spin.setRange(5.0, 30.0)
        self.hop_ms_spin.setDecimals(1)
        self.hop_ms_spin.setSingleStep(1.0)
        self.hop_ms_spin.setValue(DEFAULT_HOP_MS)
        params_layout.addWidget(self.hop_ms_spin, row, 1)
        params_layout.addWidget(QLabel("Rec: 5-15 (10 default)"), row, 2)
        row += 1

        params_layout.addWidget(QLabel("Pre-emphasis"), row, 0)
        self.pre_emphasis_spin = QDoubleSpinBox()
        self.pre_emphasis_spin.setRange(0.0, 1.0)
        self.pre_emphasis_spin.setDecimals(2)
        self.pre_emphasis_spin.setSingleStep(0.01)
        self.pre_emphasis_spin.setValue(DEFAULT_PRE_EMPHASIS)
        params_layout.addWidget(self.pre_emphasis_spin, row, 1)
        params_layout.addWidget(QLabel("Rec: 0.95-0.98 (0.97 default)"), row, 2)
        row += 1

        params_layout.addWidget(QLabel("Energy th ratio"), row, 0)
        self.energy_th_spin = QDoubleSpinBox()
        self.energy_th_spin.setRange(0.01, 0.50)
        self.energy_th_spin.setDecimals(2)
        self.energy_th_spin.setSingleStep(0.01)
        self.energy_th_spin.setValue(0.10)
        params_layout.addWidget(self.energy_th_spin, row, 1)
        params_layout.addWidget(QLabel("Rec: 0.05-0.20"), row, 2)
        row += 1

        params_layout.addWidget(QLabel("ZCR th ratio"), row, 0)
        self.zcr_th_spin = QDoubleSpinBox()
        self.zcr_th_spin.setRange(0.01, 0.50)
        self.zcr_th_spin.setDecimals(2)
        self.zcr_th_spin.setSingleStep(0.01)
        self.zcr_th_spin.setValue(0.10)
        params_layout.addWidget(self.zcr_th_spin, row, 1)
        params_layout.addWidget(QLabel("Rec: 0.05-0.20"), row, 2)
        row += 1

        params_layout.addWidget(QLabel("Entropy th ratio"), row, 0)
        self.entropy_th_spin = QDoubleSpinBox()
        self.entropy_th_spin.setRange(0.01, 0.50)
        self.entropy_th_spin.setDecimals(2)
        self.entropy_th_spin.setSingleStep(0.01)
        self.entropy_th_spin.setValue(0.10)
        params_layout.addWidget(self.entropy_th_spin, row, 1)
        params_layout.addWidget(QLabel("Rec: 0.05-0.20"), row, 2)
        row += 1

        params_layout.addWidget(QLabel("Onset pre-roll ms"), row, 0)
        self.onset_preroll_spin = QDoubleSpinBox()
        self.onset_preroll_spin.setRange(0.0, 200.0)
        self.onset_preroll_spin.setDecimals(0)
        self.onset_preroll_spin.setSingleStep(5.0)
        self.onset_preroll_spin.setValue(ONSET_PREROLL_MS)
        params_layout.addWidget(self.onset_preroll_spin, row, 1)
        params_layout.addWidget(QLabel("Rec: 30-80 (50 default)"), row, 2)
        row += 1

        params_layout.addWidget(QLabel("Fricative band low Hz"), row, 0)
        self.fricative_band_low_spin = QSpinBox()
        self.fricative_band_low_spin.setRange(600, 3500)
        self.fricative_band_low_spin.setSingleStep(100)
        self.fricative_band_low_spin.setValue(int(FRICATIVE_BAND_LOW_HZ))
        params_layout.addWidget(self.fricative_band_low_spin, row, 1)
        params_layout.addWidget(QLabel("Rec @8kHz: 1400-2400 (1800 default)"), row, 2)
        row += 1

        params_layout.addWidget(QLabel("LPC order"), row, 0)
        self.lpc_order_spin = QSpinBox()
        self.lpc_order_spin.setRange(6, 24)
        self.lpc_order_spin.setValue(DEFAULT_LPC_ORDER)
        params_layout.addWidget(self.lpc_order_spin, row, 1)
        params_layout.addWidget(QLabel("Rec @8kHz: 10-14 (12 default)"), row, 2)
        row += 1

        params_layout.addWidget(QLabel("Wavelet scales"), row, 0)
        wavelet_layout = QHBoxLayout()
        self.wavelet_min_scale_spin = QSpinBox()
        self.wavelet_min_scale_spin.setRange(1, 64)
        self.wavelet_min_scale_spin.setValue(DEFAULT_WAVELET_MIN_SCALE)
        self.wavelet_max_scale_spin = QSpinBox()
        self.wavelet_max_scale_spin.setRange(1, 64)
        self.wavelet_max_scale_spin.setValue(DEFAULT_WAVELET_MAX_SCALE)
        wavelet_layout.addWidget(self.wavelet_min_scale_spin)
        wavelet_layout.addWidget(QLabel("to"))
        wavelet_layout.addWidget(self.wavelet_max_scale_spin)
        params_layout.addLayout(wavelet_layout, row, 1)
        params_layout.addWidget(QLabel("Rec: 6-24"), row, 2)
        row += 1

        params_layout.addWidget(QLabel("HMM states"), row, 0)
        self.hmm_states_spin = QSpinBox()
        self.hmm_states_spin.setRange(2, 12)
        self.hmm_states_spin.setValue(5)
        params_layout.addWidget(self.hmm_states_spin, row, 1)
        params_layout.addWidget(QLabel("Rec: 4-8"), row, 2)
        row += 1

        params_layout.addWidget(QLabel("Codebook size"), row, 0)
        self.codebook_size_spin = QSpinBox()
        self.codebook_size_spin.setRange(8, 128)
        self.codebook_size_spin.setValue(32)
        params_layout.addWidget(self.codebook_size_spin, row, 1)
        params_layout.addWidget(QLabel("Discrete HMM: 16-64"), row, 2)
        row += 1

        params_layout.addWidget(QLabel("HMM iterations"), row, 0)
        self.hmm_iters_spin = QSpinBox()
        self.hmm_iters_spin.setRange(1, 40)
        self.hmm_iters_spin.setValue(10)
        params_layout.addWidget(self.hmm_iters_spin, row, 1)
        params_layout.addWidget(QLabel("Rec: 5-15"), row, 2)
        panel_layout.addWidget(params_group)
        self.params_group.setVisible(False)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        panel_layout.addWidget(self.status_label)

    def _build_kalman_page(self, page: QWidget):
        """Build Kalman speech-following demo page."""
        layout = QVBoxLayout(page)

        self.kalman_figure = Figure(figsize=(6, 3))
        self.kalman_canvas = FigureCanvas(self.kalman_figure)
        layout.addWidget(self.kalman_canvas, stretch=3)
        self.kalman_ax = self.kalman_figure.add_subplot(111)
        self.kalman_ax.set_title("Kalman Speech Following")
        self.kalman_ax.set_xlabel("Time (s)")
        self.kalman_ax.set_ylabel("Normalized Energy")

        controls_group = QGroupBox("Kalman Demo Controls")
        controls_layout = QGridLayout(controls_group)
        layout.addWidget(controls_group, stretch=1)

        controls_layout.addWidget(QLabel("Source"), 0, 0)
        self.kalman_source_combo = QComboBox()
        self.kalman_source_combo.addItems(["auto-best segment", "manual segment", "whole recording"])
        controls_layout.addWidget(self.kalman_source_combo, 0, 1)

        controls_layout.addWidget(QLabel("Process noise Q"), 1, 0)
        self.kalman_q_spin = QDoubleSpinBox()
        self.kalman_q_spin.setRange(0.000001, 1.0)
        self.kalman_q_spin.setDecimals(6)
        self.kalman_q_spin.setSingleStep(0.0001)
        self.kalman_q_spin.setValue(0.0005)
        controls_layout.addWidget(self.kalman_q_spin, 1, 1)
        controls_layout.addWidget(QLabel("Rec: 1e-5 to 1e-2"), 1, 2)

        controls_layout.addWidget(QLabel("Measurement noise R"), 2, 0)
        self.kalman_r_spin = QDoubleSpinBox()
        self.kalman_r_spin.setRange(0.000001, 1.0)
        self.kalman_r_spin.setDecimals(6)
        self.kalman_r_spin.setSingleStep(0.0001)
        self.kalman_r_spin.setValue(0.01)
        controls_layout.addWidget(self.kalman_r_spin, 2, 1)
        controls_layout.addWidget(QLabel("Rec: 1e-3 to 1e-1"), 2, 2)

        self.run_kalman_btn = QPushButton("Run Kalman Demo")
        self.run_kalman_btn.clicked.connect(self.run_kalman_demo)
        controls_layout.addWidget(self.run_kalman_btn, 3, 0)

        self.play_kalman_source_btn = QPushButton("Play Source")
        self.play_kalman_source_btn.clicked.connect(self.play_kalman_source)
        controls_layout.addWidget(self.play_kalman_source_btn, 3, 1)

        self.start_rt_kalman_btn = QPushButton("Start Live")
        self.start_rt_kalman_btn.clicked.connect(self.start_realtime_kalman)
        controls_layout.addWidget(self.start_rt_kalman_btn, 4, 0)

        self.stop_rt_kalman_btn = QPushButton("Stop Live")
        self.stop_rt_kalman_btn.clicked.connect(self.stop_realtime_kalman)
        self.stop_rt_kalman_btn.setEnabled(False)
        controls_layout.addWidget(self.stop_rt_kalman_btn, 4, 1)

        self.kalman_status_label = QLabel("Record and segment speech, then run demo.")
        self.kalman_status_label.setWordWrap(True)
        layout.addWidget(self.kalman_status_label)

    def _switch_to_recognition(self):
        if self.rt_running:
            self.stop_realtime_kalman()
        self.mode_stack.setCurrentIndex(0)

    def _switch_to_kalman(self):
        self.mode_stack.setCurrentIndex(1)

    def _reset_realtime_kalman_buffers(self):
        self.rt_pending_samples = np.array([], dtype=np.float32)
        self.rt_time = []
        self.rt_measured = []
        self.rt_estimated = []
        self.rt_residuals = []
        self.rt_confidence = 0.0
        self.rt_frame_index = 0
        self.rt_energy_scale = 1e-12
        self.rt_kf_state = 0.0
        self.rt_kf_var = 1.0

    def start_realtime_kalman(self):
        """Start live microphone Kalman following."""
        if self.rt_running:
            return
        if not is_available():
            QMessageBox.critical(self, "Missing Dependency", "sounddevice is not installed.")
            return
        try:
            params = self._current_params()
            hop_size = int(self.fs * params["hop_ms"] / 1000.0)
            blocksize = max(128, hop_size)
            self.rt_stream = RealtimeAudioStream(self.fs, blocksize=blocksize)
            self.rt_stream.start()
            self._reset_realtime_kalman_buffers()
            self.rt_running = True
            self.rt_timer.start()
            self.start_rt_kalman_btn.setEnabled(False)
            self.stop_rt_kalman_btn.setEnabled(True)
            self.kalman_status_label.setText("Live Kalman running...")
        except Exception as exc:
            self.rt_running = False
            self.rt_stream = None
            QMessageBox.critical(self, "Live Kalman Error", str(exc))

    def stop_realtime_kalman(self):
        """Stop live microphone Kalman following."""
        if not self.rt_running and self.rt_stream is None:
            return
        self.rt_timer.stop()
        if self.rt_stream is not None:
            try:
                self.rt_stream.stop()
            except Exception:
                pass
            self.rt_stream = None
        self.rt_running = False
        self.start_rt_kalman_btn.setEnabled(True)
        self.stop_rt_kalman_btn.setEnabled(False)
        self.kalman_status_label.setText("Live Kalman stopped.")

    def _on_realtime_kalman_tick(self):
        """Process incoming microphone chunks and update Kalman plot."""
        if not self.rt_running or self.rt_stream is None:
            return
        params = self._current_params()
        frame_size = int(self.fs * params["frame_ms"] / 1000.0)
        hop_size = int(self.fs * params["hop_ms"] / 1000.0)
        q = float(self.kalman_q_spin.value())
        r = float(self.kalman_r_spin.value())

        chunks = self.rt_stream.read_chunks(max_chunks=128)
        if not chunks:
            return
        self.rt_pending_samples = np.concatenate([self.rt_pending_samples] + chunks).astype(np.float32)

        updated = False
        while self.rt_pending_samples.size >= frame_size:
            frame = self.rt_pending_samples[:frame_size]
            self.rt_pending_samples = self.rt_pending_samples[hop_size:]
            frame_emph = apply_pre_emphasis(frame, params["pre_emphasis"])
            energy = float(np.mean(frame_emph ** 2))
            self.rt_energy_scale = max(self.rt_energy_scale, energy)
            z = energy / (self.rt_energy_scale + 1e-12)

            # Scalar Kalman step.
            self.rt_kf_var = self.rt_kf_var + q
            k = self.rt_kf_var / (self.rt_kf_var + r)
            innovation = z - self.rt_kf_state
            self.rt_kf_state = self.rt_kf_state + k * innovation
            self.rt_kf_var = (1.0 - k) * self.rt_kf_var

            t = (self.rt_frame_index * hop_size + frame_size / 2) / self.fs
            self.rt_frame_index += 1
            self.rt_time.append(t)
            self.rt_measured.append(z)
            self.rt_estimated.append(self.rt_kf_state)
            self.rt_residuals.append(innovation)
            updated = True

        if not updated:
            return

        # Keep latest window for responsiveness.
        max_points = 800
        if len(self.rt_time) > max_points:
            self.rt_time = self.rt_time[-max_points:]
            self.rt_measured = self.rt_measured[-max_points:]
            self.rt_estimated = self.rt_estimated[-max_points:]
            self.rt_residuals = self.rt_residuals[-max_points:]

        residual_window = self.rt_residuals[-120:]
        if len(residual_window) >= 8:
            residual_std = float(np.std(residual_window))
            # Smaller innovation spread means the model is following measurements stably.
            self.rt_confidence = float(np.clip(100.0 * (1.0 - residual_std), 0.0, 100.0))
        else:
            self.rt_confidence = 0.0

        self.kalman_ax.clear()
        self.kalman_ax.plot(self.rt_time, self.rt_measured, label="Measured Energy", color="tab:blue", alpha=0.7)
        self.kalman_ax.plot(self.rt_time, self.rt_estimated, label="Kalman Estimate", color="tab:red", linewidth=2.0)
        self.kalman_ax.set_title(f"Kalman Speech Following (Live) - Confidence {self.rt_confidence:.1f}%")
        self.kalman_ax.set_xlabel("Time (s)")
        self.kalman_ax.set_ylabel("Normalized Energy")
        self.kalman_ax.grid(True, alpha=0.2)
        self.kalman_ax.legend(loc="upper right")
        self.kalman_canvas.draw()
        self.kalman_status_label.setText(
            f"Live Kalman running: {len(self.rt_estimated)} frames, confidence {self.rt_confidence:.1f}%."
        )

    def toggle_advanced_params(self, expanded: bool):
        """Expand/collapse advanced DSP parameter controls."""
        self.params_group.setVisible(expanded)
        self.advanced_toggle_btn.setText(
            "Hide Advanced Parameters" if expanded else "Show Advanced Parameters"
        )

    def _current_params(self) -> dict:
        """Read current DSP/segmentation parameters from UI controls."""
        return {
            "frame_ms": float(self.frame_ms_spin.value()),
            "hop_ms": float(self.hop_ms_spin.value()),
            "pre_emphasis": float(self.pre_emphasis_spin.value()),
            "energy_th": float(self.energy_th_spin.value()),
            "zcr_th": float(self.zcr_th_spin.value()),
            "entropy_th": float(self.entropy_th_spin.value()),
            "onset_preroll_ms": float(self.onset_preroll_spin.value()),
            "fricative_band_low_hz": float(self.fricative_band_low_spin.value()),
            "lpc_order": int(self.lpc_order_spin.value()),
            "wav_min": int(self.wavelet_min_scale_spin.value()),
            "wav_max": int(self.wavelet_max_scale_spin.value()),
            "hmm_states": int(self.hmm_states_spin.value()),
            "codebook_size": int(self.codebook_size_spin.value()),
            "hmm_iters": int(self.hmm_iters_spin.value()),
        }

    def apply_preset(self):
        """Apply predefined DSP parameter presets."""
        preset = self.preset_combo.currentText()
        if preset.startswith("Quiet Room"):
            self.frame_ms_spin.setValue(30.0)
            self.hop_ms_spin.setValue(10.0)
            self.pre_emphasis_spin.setValue(0.97)
            self.energy_th_spin.setValue(0.10)
            self.zcr_th_spin.setValue(0.10)
            self.entropy_th_spin.setValue(0.10)
            self.onset_preroll_spin.setValue(50.0)
            self.fricative_band_low_spin.setValue(1800)
            self.lpc_order_spin.setValue(12)
            self.wavelet_min_scale_spin.setValue(6)
            self.wavelet_max_scale_spin.setValue(24)
            self.hmm_states_spin.setValue(5)
            self.codebook_size_spin.setValue(32)
            self.hmm_iters_spin.setValue(10)
        elif preset == "Noisy Room":
            self.frame_ms_spin.setValue(35.0)
            self.hop_ms_spin.setValue(12.0)
            self.pre_emphasis_spin.setValue(0.97)
            self.energy_th_spin.setValue(0.14)
            self.zcr_th_spin.setValue(0.14)
            self.entropy_th_spin.setValue(0.14)
            self.onset_preroll_spin.setValue(65.0)
            self.fricative_band_low_spin.setValue(1600)
            self.lpc_order_spin.setValue(12)
            self.wavelet_min_scale_spin.setValue(8)
            self.wavelet_max_scale_spin.setValue(28)
            self.hmm_states_spin.setValue(6)
            self.codebook_size_spin.setValue(40)
            self.hmm_iters_spin.setValue(12)
        elif preset == "Fast Speech":
            self.frame_ms_spin.setValue(25.0)
            self.hop_ms_spin.setValue(8.0)
            self.pre_emphasis_spin.setValue(0.97)
            self.energy_th_spin.setValue(0.10)
            self.zcr_th_spin.setValue(0.10)
            self.entropy_th_spin.setValue(0.10)
            self.onset_preroll_spin.setValue(40.0)
            self.fricative_band_low_spin.setValue(2000)
            self.lpc_order_spin.setValue(12)
            self.wavelet_min_scale_spin.setValue(6)
            self.wavelet_max_scale_spin.setValue(22)
            self.hmm_states_spin.setValue(5)
            self.codebook_size_spin.setValue(28)
            self.hmm_iters_spin.setValue(8)
        self.status_label.setText(f"Preset applied: {preset}")

    def _select_audio_for_view(self) -> np.ndarray:
        """Select whole recording or one detected segment for visualization."""
        if self.audio_data.size == 0:
            return np.array([])
        if not self.segments:
            return self.audio_data
        items = ["Whole recording"] + [
            f"Segment {i+1}: {start/self.fs:.2f}s–{end/self.fs:.2f}s"
            for i, (start, end) in enumerate(self.segments)
        ]
        selection, ok = QInputDialog.getItem(
            self, "Select Audio", "Select audio to visualize:", items, 0, False
        )
        if not ok:
            return np.array([])
        if selection == "Whole recording":
            return self.audio_data
        seg_idx = items.index(selection) - 1
        start, end = self.segments[seg_idx]
        return self.audio_data[start:end]

    def _best_segment_index(self) -> int:
        """Select best segment candidate based on duration and energy."""
        if not self.segments:
            return -1
        best_idx = 0
        best_score = -np.inf
        for idx, (start, end) in enumerate(self.segments):
            segment = self.audio_data[start:end]
            if segment.size == 0:
                continue
            duration_samples = end - start
            energy = float(np.mean(segment.astype(float) ** 2))
            # Prioritize duration, then average energy.
            score = duration_samples + 0.1 * duration_samples * energy
            if score > best_score:
                best_score = score
                best_idx = idx
        return best_idx

    def _select_segment_index(self, action_label: str) -> int:
        """Return segment index using manual or auto-best strategy."""
        if not self.segments:
            return -1
        pick_mode = self.segment_pick_combo.currentText()
        if pick_mode == "auto-best":
            idx = self._best_segment_index()
            if idx >= 0:
                start, end = self.segments[idx]
                self.status_label.setText(
                    f"Auto-selected segment {idx+1} ({start/self.fs:.2f}s-{end/self.fs:.2f}s) for {action_label}."
                )
            return idx

        segment_items = [
            f"Segment {i+1}: {start/self.fs:.2f}s–{end/self.fs:.2f}s"
            for i, (start, end) in enumerate(self.segments)
        ]
        segment_index, ok = QInputDialog.getItem(
            self, "Select Segment", f"Choose a segment to {action_label}:", segment_items, 0, False
        )
        if not ok:
            return -1
        return segment_items.index(segment_index)

    def _select_audio_for_kalman(self) -> np.ndarray:
        """Select source audio for Kalman demo according to source mode."""
        if self.audio_data.size == 0:
            return np.array([])
        mode = self.kalman_source_combo.currentText()
        if mode == "whole recording" or not self.segments:
            return self.audio_data
        if mode == "auto-best segment":
            idx = self._best_segment_index()
            if idx < 0:
                return np.array([])
            start, end = self.segments[idx]
            return self.audio_data[start:end]
        # manual segment
        idx = self._select_segment_index("use in Kalman demo")
        if idx < 0:
            return np.array([])
        start, end = self.segments[idx]
        return self.audio_data[start:end]

    def run_kalman_demo(self):
        """Run Kalman tracking over short-time energy and plot following behavior."""
        if self.audio_data.size == 0:
            QMessageBox.warning(self, "No audio", "Please record audio first.")
            return
        params = self._current_params()
        samples = self._select_audio_for_kalman()
        if samples.size == 0:
            return
        times, energy = compute_energy_trajectory(
            samples,
            self.fs,
            frame_ms=params["frame_ms"],
            hop_ms=params["hop_ms"],
            pre_emphasis=params["pre_emphasis"],
        )
        if energy.size == 0:
            QMessageBox.information(self, "No data", "Not enough audio for Kalman demo.")
            return
        estimate = kalman_filter_1d(
            energy,
            process_var=float(self.kalman_q_spin.value()),
            measurement_var=float(self.kalman_r_spin.value()),
            init_state=float(energy[0]),
            init_var=1.0,
        )
        self.kalman_ax.clear()
        self.kalman_ax.plot(times, energy, label="Measured Energy", color="tab:blue", alpha=0.7)
        self.kalman_ax.plot(times, estimate, label="Kalman Estimate", color="tab:red", linewidth=2.0)
        self.kalman_ax.set_title("Kalman Speech Following (Energy Tracking)")
        self.kalman_ax.set_xlabel("Time (s)")
        self.kalman_ax.set_ylabel("Normalized Energy")
        self.kalman_ax.grid(True, alpha=0.2)
        self.kalman_ax.legend(loc="upper right")
        self.kalman_canvas.draw()
        self.kalman_status_label.setText(
            f"Kalman demo done: {len(energy)} frames, Q={self.kalman_q_spin.value():.6f}, R={self.kalman_r_spin.value():.6f}."
        )

    def play_kalman_source(self):
        """Play currently selected Kalman source audio."""
        if not is_available():
            QMessageBox.critical(self, "Missing Dependency", "sounddevice is not installed.")
            return
        samples = self._select_audio_for_kalman()
        if samples.size == 0:
            return
        try:
            try:
                stop_playback()
            except Exception:
                pass
            play_audio_buffer(samples, self.fs, blocking=False)
        except AudioIOError as exc:
            QMessageBox.critical(self, "Playback Error", str(exc))

    def show_spectrogram_view(self):
        """Display spectrogram for selected audio."""
        params = self._current_params()
        samples = self._select_audio_for_view()
        if samples.size == 0:
            return
        spec = compute_spectrogram(
            samples,
            self.fs,
            frame_ms=params["frame_ms"],
            hop_ms=params["hop_ms"],
            pre_emphasis=params["pre_emphasis"],
        )
        if spec.size == 0:
            QMessageBox.information(self, "No data", "Not enough audio for spectrogram.")
            return
        plt.figure("Spectrogram")
        plt.imshow(20 * np.log10(spec.T + 1e-12), origin="lower", aspect="auto", cmap="magma")
        plt.title("Spectrogram (dB)")
        plt.xlabel("Frame")
        plt.ylabel("Frequency bin")
        plt.colorbar()
        plt.tight_layout()
        plt.show()

    def show_cepstrogram_view(self):
        """Display cepstrogram for selected audio."""
        params = self._current_params()
        samples = self._select_audio_for_view()
        if samples.size == 0:
            return
        cep = compute_cepstrogram(
            samples,
            self.fs,
            frame_ms=params["frame_ms"],
            hop_ms=params["hop_ms"],
            pre_emphasis=params["pre_emphasis"],
        )
        if cep.size == 0:
            QMessageBox.information(self, "No data", "Not enough audio for cepstrogram.")
            return
        plt.figure("Cepstrogram")
        plt.imshow(cep.T, origin="lower", aspect="auto", cmap="viridis")
        plt.title("Cepstrogram")
        plt.xlabel("Frame")
        plt.ylabel("Quefrency bin")
        plt.colorbar()
        plt.tight_layout()
        plt.show()

    def show_wavelet_view(self):
        """Display wavelet scalogram-like diagram for selected audio."""
        params = self._current_params()
        samples = self._select_audio_for_view()
        if samples.size == 0:
            return
        feats = compute_wavelet_features(
            samples,
            self.fs,
            min_scale=params["wav_min"],
            max_scale=params["wav_max"],
            frame_ms=params["frame_ms"],
            hop_ms=params["hop_ms"],
            pre_emphasis=params["pre_emphasis"],
        )
        if feats.size == 0:
            QMessageBox.information(self, "No data", "Not enough audio for wavelet diagram.")
            return
        plt.figure("Wavelet Diagram")
        plt.imshow(feats.T, origin="lower", aspect="auto", cmap="plasma")
        plt.title("Wavelet Scalogram (Frame-Scale Energy)")
        plt.xlabel("Frame")
        plt.ylabel("Scale index")
        plt.colorbar()
        plt.tight_layout()
        plt.show()

    def synthesize_lpc_audio(self):
        """Synthesize speech from selected audio using LPC analysis-synthesis."""
        params = self._current_params()
        samples = self._select_audio_for_view()
        if samples.size == 0:
            return
        synth = synthesize_lpc_speech(
            samples,
            self.fs,
            order=params["lpc_order"],
            frame_ms=params["frame_ms"],
            hop_ms=params["hop_ms"],
            pre_emphasis=params["pre_emphasis"],
        )
        if synth.size == 0:
            QMessageBox.information(self, "No data", "Not enough audio for LPC synthesis.")
            return
        peak = float(np.max(np.abs(synth))) + 1e-12
        self.lpc_synth_audio = (0.9 * synth / peak).astype(np.float32)
        self.status_label.setText("LPC synthesis ready. Click Play LPC.")

    def play_lpc_audio(self):
        """Play latest LPC-synthesized audio."""
        if not is_available():
            QMessageBox.critical(self, "Missing Dependency", "sounddevice is not installed.")
            return
        if self.lpc_synth_audio.size == 0:
            QMessageBox.information(self, "No LPC audio", "Run Synthesize LPC first.")
            return
        try:
            try:
                stop_playback()
            except Exception:
                pass
            play_audio_buffer(self.lpc_synth_audio, self.fs, blocking=False)
        except AudioIOError as exc:
            QMessageBox.critical(self, "Playback Error", str(exc))

    def closeEvent(self, event):
        """Ensure realtime resources are released on app close."""
        try:
            self.stop_realtime_kalman()
        except Exception:
            pass
        super().closeEvent(event)

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
        params = self._current_params()
        method = self.method_combo.currentText()
        try:
            self.segments = detect_segments(
                self.audio_data,
                self.fs,
                method=method,
                frame_ms=params["frame_ms"],
                hop_ms=params["hop_ms"],
                energy_threshold_ratio=params["energy_th"],
                zcr_threshold_ratio=params["zcr_th"],
                entropy_threshold_ratio=params["entropy_th"],
                onset_preroll_ms=params["onset_preroll_ms"],
                fricative_band_low_hz=params["fricative_band_low_hz"],
            )
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

        selected_idx = self._select_segment_index("save")
        if selected_idx < 0:
            return
        start, end = self.segments[selected_idx]

        label, ok = QInputDialog.getText(self, "Label", "Enter the word label:")
        if not ok or not label:
            return

        params = self._current_params()
        segment_samples = self.audio_data[start:end]
        spect = compute_spectrogram(
            segment_samples,
            self.fs,
            frame_ms=params["frame_ms"],
            hop_ms=params["hop_ms"],
            pre_emphasis=params["pre_emphasis"],
        )
        feature_type = self.feature_combo.currentText()
        if feature_type == "mfcc":
            feature_matrix = compute_mfcc(
                segment_samples,
                self.fs,
                num_filters=DEFAULT_MEL_FILTERS,
                num_ceps=DEFAULT_MEL_FILTERS,
                frame_ms=params["frame_ms"],
                hop_ms=params["hop_ms"],
                fmin=DEFAULT_MEL_FMIN_HZ,
                fmax=DEFAULT_MEL_FMAX_HZ,
                pre_emphasis=params["pre_emphasis"],
            )
        elif feature_type == "bark":
            feature_matrix = compute_bark_band_energies(
                segment_samples,
                self.fs,
                frame_ms=params["frame_ms"],
                hop_ms=params["hop_ms"],
                pre_emphasis=params["pre_emphasis"],
            )
        elif feature_type == "lpc":
            feature_matrix = compute_lpc_features(
                segment_samples,
                self.fs,
                order=params["lpc_order"],
                frame_ms=params["frame_ms"],
                hop_ms=params["hop_ms"],
                pre_emphasis=params["pre_emphasis"],
            )
        else:
            feature_matrix = compute_wavelet_features(
                segment_samples,
                self.fs,
                min_scale=params["wav_min"],
                max_scale=params["wav_max"],
                frame_ms=params["frame_ms"],
                hop_ms=params["hop_ms"],
                pre_emphasis=params["pre_emphasis"],
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
            frame_ms=params["frame_ms"],
            hop_ms=params["hop_ms"],
            mel_filters=DEFAULT_MEL_FILTERS,
            mel_fmin_hz=DEFAULT_MEL_FMIN_HZ,
            mel_fmax_hz=DEFAULT_MEL_FMAX_HZ,
            pre_emphasis=params["pre_emphasis"],
            lpc_order=params["lpc_order"],
            wavelet_min_scale=params["wav_min"],
            wavelet_max_scale=params["wav_max"],
        )
        append_entry(self.dictionary_file, entry)

        QMessageBox.information(self, "Saved", f"Segment saved under label '{label}'.")

    def compare_audio(self):
        """Compare a selected segmented utterance against dictionary entries."""
        dictionary = load_dictionary(self.dictionary_file)
        if not dictionary:
            QMessageBox.warning(
                self,
                "No dictionary",
                "Dictionary is missing or empty. Save some segments first.",
            )
            return

        if self.audio_data.size == 0:
            QMessageBox.warning(self, "No audio", "Please record audio before comparing.")
            return
        if not self.segments:
            QMessageBox.warning(self, "No segments", "Please segment audio before comparing.")
            return

        selected_idx = self._select_segment_index("compare")
        if selected_idx < 0:
            return
        start, end = self.segments[selected_idx]
        segment_samples = self.audio_data[start:end]

        self.status_label.setText("Comparing selected segment...")
        self.repaint()

        params = self._current_params()
        feature_type = self.feature_combo.currentText()
        if feature_type == "mfcc":
            query_features = compute_mfcc(
                segment_samples,
                self.fs,
                num_filters=DEFAULT_MEL_FILTERS,
                num_ceps=DEFAULT_MEL_FILTERS,
                frame_ms=params["frame_ms"],
                hop_ms=params["hop_ms"],
                fmin=DEFAULT_MEL_FMIN_HZ,
                fmax=DEFAULT_MEL_FMAX_HZ,
                pre_emphasis=params["pre_emphasis"],
            )
        elif feature_type == "bark":
            query_features = compute_bark_band_energies(
                segment_samples,
                self.fs,
                frame_ms=params["frame_ms"],
                hop_ms=params["hop_ms"],
                pre_emphasis=params["pre_emphasis"],
            )
        elif feature_type == "lpc":
            query_features = compute_lpc_features(
                segment_samples,
                self.fs,
                order=params["lpc_order"],
                frame_ms=params["frame_ms"],
                hop_ms=params["hop_ms"],
                pre_emphasis=params["pre_emphasis"],
            )
        else:
            query_features = compute_wavelet_features(
                segment_samples,
                self.fs,
                min_scale=params["wav_min"],
                max_scale=params["wav_max"],
                frame_ms=params["frame_ms"],
                hop_ms=params["hop_ms"],
                pre_emphasis=params["pre_emphasis"],
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
        backend = self.recognition_backend_combo.currentText()
        if backend != "dtw":
            grouped = group_feature_sequences_by_label(dictionary, feature_type)
            if backend == "hmm-discrete":
                best_label, best_distance = recognize_with_hmm_discrete(
                    grouped,
                    query_features,
                    n_states=params["hmm_states"],
                    codebook_size=params["codebook_size"],
                    n_iters=params["hmm_iters"],
                )
            else:
                best_label, best_distance = recognize_with_hmm_continuous(
                    grouped,
                    query_features,
                    n_states=params["hmm_states"],
                    n_iters=params["hmm_iters"],
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
                f"Closest match: {best_label} (score={best_distance:.3f}, backend={backend})",
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
