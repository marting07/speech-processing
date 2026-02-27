# Speech Processing AGENTS

## Purpose

PyQt6 desktop app for speech recording, segmentation, feature extraction, and recognition with DTW/HMM backends plus Kalman speech-following mode.

## File Map

- UI and mode routing: `speech_recognition_app.py`
- DSP and feature extraction: `speech_dsp.py`
- Recognition backends: `speech_hmm.py`
- Dictionary persistence/schema helpers: `dictionary_store.py`
- Audio I/O adapter: `audio_io.py`
- Dictionary data: `speech_dictionary.json`

## Run

- `python3 speech_recognition_app.py`

## Current Workflow Decisions

- Mode switch is menu-driven:
  - `Speech Recognition`
  - `Kalman Speech Following`
- Recognition backends supported in UI:
  - `dtw`
  - `hmm-discrete`
  - `hmm-continuous`
- Segment selection strategy is explicit:
  - `manual`
  - `auto-best` (default for save/compare automation flows)
- Advanced DSP parameters are shown in a collapsible panel; keep compact startup behavior.
- Presets are part of the normal flow (`Quiet Room`, `Noisy Room`, `Fast Speech`, `Custom`).

## Defaults And Ranges

- Sampling rate: `8000 Hz`
- Frame length: `30 ms` (recommended range `20-40 ms`)
- Hop length: `10 ms` (recommended range `5-15 ms`)
- Pre-emphasis: `0.97` (recommended range `0.95-0.98`)
- Threshold ratios (`energy`, `zcr`, `entropy`): typical `0.05-0.20`
- LPC order: `12` at `8 kHz` (recommended `10-14`)
- Wavelet scales: `6-24`

## Kalman Mode Notes

- Supports offline source modes (`auto-best segment`, `manual segment`, `whole recording`) and live microphone tracking.
- Live mode confidence is based on rolling innovation stability; keep this metric visible in UI updates.

## Data Compatibility Notes

- Keep backward compatibility with existing entries in `speech_dictionary.json` (including legacy MFCC-only records).
- New saves should continue to include extraction settings and selected feature matrices alongside spectrogram metadata.
