# Speech Processing AGENTS

## Purpose

Desktop speech segmentation/recognition app (PyQt6) with manual DSP pipeline and JSON-based dictionary storage.

## Main File

- Application + DSP pipeline: `speech_recognition_app.py`
- Dictionary data: `speech_dictionary.json`

## Current Signal Defaults

Defaults are aligned to class notes/presentation captured in recent work:
- Sampling rate: `8000 Hz`
- Frame length: `30 ms`
- Hop length: `10 ms`
- Window: Hamming
- MFCC count: `15`
- Bark energies: bands 1..16

## Dictionary/Data Contract Notes

- Saved entries include segmentation metadata plus:
  - Spectrogram matrix (`spectrogram`)
  - Bark-band energies (`bark_energies`)
  - MFCC frames (`mfcc`)
- Prefer preserving backward compatibility with existing `speech_dictionary.json` entries.

## Session-Specific Inputs

Recent updates were driven by:
- `/Users/marting/Downloads/NotasProcDig.pdf`
- `/Users/marting/Downloads/Presentacion.pdf`

When adjusting processing parameters, use those documents as the source of truth and keep constants synchronized in `speech_recognition_app.py`.

## Run

- `python speech_recognition_app.py`
