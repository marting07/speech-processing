# Speech Processing Demo (PyQt6)

Desktop app for recording speech, segmenting isolated words, extracting features, and doing template-based recognition with DTW.

## Features

- Record audio from microphone.
- Segment speech with short-time methods (`energy`, `zcr`, `energy_zcr`, `entropy`).
- Save labeled segments to a JSON dictionary.
- Extract and store:
  - Spectrogram frames
  - Bark-band energies (bands 1..16)
  - MFCC frames
- Compare a new recording against dictionary entries using Dynamic Time Warping (DTW).

## Architecture

- `speech_recognition_app.py`: PyQt6 UI and user interaction flow only.
- `speech_dsp.py`: digital signal processing and feature extraction logic.
- `dictionary_store.py`: dictionary schema creation, persistence, and lookup helpers.
- `audio_io.py`: audio capture/playback adapter over `sounddevice`.

## Processing Defaults

The current implementation is configured with:

- Sampling rate: `8000 Hz`
- Frame size: `30 ms`
- Hop size: `10 ms`
- Window: `Hamming`
- MFCC: `15` Mel filters/coefficients, `fmin=80 Hz`, `fmax=4000 Hz`

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
2. Click **Segment** (select method if needed).
3. Click **Save Segment** and enter a label.
4. Repeat to build dictionary entries.
5. Click **Compare** to recognize a new utterance.

## Dictionary File

The app writes `speech_dictionary.json` in the project root.

- If the file does not exist, it is created on first save.
- New saves append entries.
- Recognition supports current entries and older legacy MFCC-only entries.

Each entry stores metadata plus frame-indexed feature matrices (spectrogram, Bark energies, MFCC).
