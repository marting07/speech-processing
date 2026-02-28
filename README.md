# Speech Processing Demo (PyQt6)

Desktop app for recording speech, segmenting isolated words, extracting features, and doing template-based recognition with DTW.

## Features

- Mode menu to switch between:
  - `Speech Recognition`
  - `Kalman Speech Following`
- Record audio from microphone.
- Segment speech with short-time methods (`energy`, `zcr`, `energy_zcr`, `entropy`).
- Segmentation uses adaptive robust thresholds + feature smoothing + hysteresis/hangover + post-merge/min-duration cleanup.
- Fricative onsets are handled with high-band energy cues plus onset pre-roll to better keep starts like Spanish `s` (`seis`, `siete`).
- Choose feature mode per operation: `mfcc`, `bark`, `lpc`, or `wavelet`.
- Choose segment selection strategy: `manual` or `auto-best`.
- Save labeled segments to a JSON dictionary.
- Extract and store:
  - Spectrogram frames
  - One selected feature matrix (`mfcc`/`bark`/`lpc`/`wavelet`) per entry
- Compare selected segmented utterances against dictionary entries using Dynamic Time Warping (DTW).
- Compare using selectable backends:
  - `dtw`
  - `hmm-discrete` (with clustering + vector codebook quantization)
  - `hmm-continuous` (Gaussian emissions)
- Visualize **Spectrogram** and **Cepstrogram** from recorded audio/segments.
- Visualize **Wavelet Scalogram** (frame-scale energy diagram).
- Run **LPC Speech Synthesis** (analyze/resynthesize) and playback.
- Run **Kalman speech-following demo** (short-time energy tracking).
- Tune DSP parameters from UI (frame/hop, thresholds, pre-emphasis, LPC order, wavelet scales) with recommended ranges.
- Apply ready presets: `Quiet Room`, `Noisy Room`, `Fast Speech`.

## Architecture

- `speech_recognition_app.py`: PyQt6 UI and user interaction flow only.
- `speech_dsp.py`: digital signal processing and feature extraction logic.
- `speech_hmm.py`: HMM recognition backends (discrete and continuous).
- `dictionary_store.py`: dictionary schema creation, persistence, and lookup helpers.
- `audio_io.py`: audio capture/playback adapter over `sounddevice`.

## Processing Defaults

The current implementation is configured with:

- Sampling rate: `8000 Hz`
- Frame size: `30 ms`
- Hop size: `10 ms`
- Window: `Hamming`
- Pre-emphasis: `0.97`
- MFCC: `15` Mel filters/coefficients, `fmin=80 Hz`, `fmax=4000 Hz`
- LPC order: `12` (common recommendation for `8 kHz`)
- Wavelet scales: `6..24`
- HMM states: `5`
- Discrete HMM codebook size: `32`
- HMM training iterations: `10`

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

```bash
python3 speech_recognition_app.py
```

## Basic Workflow

1. Click **Record**.
2. Click **Segment** (select segmentation method if needed).
3. Use **Clear Segments** to remove segmentation markers and re-run if needed.
4. Optionally inspect **Show Spectrogram** / **Show Cepstrogram**.
5. Optionally inspect **Show Wavelet**.
6. Optionally run **Synthesize LPC** then **Play LPC**.
7. Choose **Features** (`mfcc`/`bark`/`lpc`/`wavelet`) and **Segment pick** (`manual` or `auto-best`), then click **Save Segment** and enter a label.
8. Repeat to build dictionary entries.
9. Select **Recognition backend** (`dtw`, `hmm-discrete`, `hmm-continuous`).
10. Use the same **Features** mode and click **Compare** to match a segment against dictionary entries.

## Kalman Following Mode

1. Open menu **Mode → Kalman Speech Following**.
2. Choose source (`auto-best segment`, `manual segment`, or `whole recording`).
3. Set Kalman parameters:
   - Process noise `Q` (recommended `1e-5` to `1e-2`)
   - Measurement noise `R` (recommended `1e-3` to `1e-1`)
4. Click **Run Kalman Demo** for offline analysis of selected audio.
5. Or click **Start Live** to track microphone energy in real time, then **Stop Live**.
6. In live mode, confidence is shown as a percentage based on the rolling innovation stability.
7. Optional: **Play Source**.

## Recommended Ranges (UI Labels)

- Frame length: `20-40 ms` (default `30`)
- Hop length: `5-15 ms` (default `10`)
- Pre-emphasis: `0.95-0.98` (default `0.97`)
- Energy/ZCR/Entropy threshold ratios: `0.05-0.20`
- Onset pre-roll: `30-80 ms` (default `50 ms`)
- Fricative band low cutoff at `8 kHz`: `1400-2400 Hz` (default `1800 Hz`)
- LPC order at `8 kHz`: `10-14` (default `12`)
- Wavelet scales: `6-24`
- HMM states: `4-8`
- Discrete codebook size: `16-64`
- HMM iterations: `5-15`

## Dictionary File

The app writes `speech_dictionary.json` in the project root.

- If the file does not exist, it is created on first save.
- New saves append entries.
- Recognition supports current entries and older legacy MFCC-only entries.

Each entry stores metadata, `feature_type`, spectrogram, selected feature matrix, and extraction settings (including pre-emphasis).
