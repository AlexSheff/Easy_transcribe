# 🎙️ Easy Transcriber

**Easy Transcriber** is a powerful, fully offline desktop application for high-accuracy audio/video transcription, speaker diarization, and voice fingerprinting. Built for privacy and performance — your data never leaves your machine.

---

## ✨ Key Features

- 🔒 **100% Offline** — No API keys, no cloud, no subscriptions. Everything runs locally.
- 🎙️ **Real-time Microphone Recording** — Capture live speech with automatic silence detection (VAD), saving chunks instantly to disk.
- 🎬 **MP4 / Video to WAV Converter** — Built-in converter: drop any MP4, MKV, or AVI file and get a clean WAV ready for transcription.
- 🗣️ **Speaker Diarization** — Adaptive utterance-chunked agglomerative clustering for accurate turn-taking even in fast-paced dialogs.
- 🧬 **Voice Fingerprinting** — Persistently identifies and remembers speakers across different files and sessions using ECAPA-TDNN embeddings.
- 🧠 **Semantic Clustering** — Groups transcript segments by meaning (optional, configurable).
- 🌍 **Multilingual** — High-quality transcription for English, Russian, Tagalog, and many more via Faster-Whisper.
- 📂 **Multi-format Input** — MP4, MKV, AVI, MP3, WAV, FLAC.
- 📝 **Structured Export** — Generates Markdown, JSON, and SRT output files automatically.

---

## 🚀 Quick Start (Windows)

1. **Setup:** Double-click `setup.bat`
   - Creates a Python virtual environment
   - Installs all dependencies from `requirements.txt`
   - Checks for FFmpeg on your PATH

2. **Launch:** Double-click `run.bat` (or `python app.py`)

> [!IMPORTANT]
> **FFmpeg is required** for audio/video processing.
> Download from [ffmpeg.org](https://ffmpeg.org/) and add the `bin/` folder to your system PATH.

---

## ⚙️ Configuration

All settings are in `app/config/config.yaml`:

```yaml
asr:
  model_size: "medium"   # tiny | base | small | medium | large-v3
  device: "auto"          # cpu | cuda | auto
  language: "auto"        # auto | ru | en | ...

diarization:
  enabled: true
  min_speakers: 1
  max_speakers: 3

voice_fingerprint:
  enabled: true
  threshold: 0.65         # similarity threshold for speaker matching

semantic_clustering:
  enabled: false          # set true to group segments by meaning
  method: "kmeans"        # kmeans | hdbscan
  max_clusters: 8

paths:
  output_dir: "app/output"
  temp_dir: "app/temp"
  chunk_duration: 1800    # max chunk size in seconds (30 min)
```

---

## 📂 Project Structure

```
Easy_transcribe/
├── app/
│   ├── config/
│   │   └── config.yaml          # All runtime settings
│   ├── core/
│   │   ├── audio_processor.py   # FFmpeg wrapper, chunking, MP4→WAV conversion
│   │   ├── transcriber.py       # Faster-Whisper ASR engine
│   │   ├── voice_fingerprint.py # SpeechBrain ECAPA-TDNN speaker embeddings
│   │   ├── mic_recorder.py      # Real-time microphone capture with VAD
│   │   ├── semantic_engine.py   # Sentence-transformer clustering
│   │   └── exporter.py          # Markdown / JSON / SRT export
│   ├── gui/
│   │   └── main_window.py       # PySide6 main UI
│   ├── pipeline/                # Processing pipeline orchestration
│   ├── models/                  # Cached local model files
│   ├── voice_db/                # Persistent speaker fingerprint database
│   └── output/
│       ├── Converted/           # MP4→WAV conversion results
│       └── <session folders>/   # Transcripts per file/session
├── app.py                       # Entry point
├── setup.bat                    # One-click environment setup
├── run.bat                      # One-click launcher
├── requirements.txt
└── tech.md                      # Detailed technical specification
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **ASR Engine** | [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2 backend) |
| **Speaker Diarization** | SpeechBrain · ECAPA-TDNN · AHC Clustering |
| **Voice Fingerprinting** | `speechbrain/spkrec-ecapa-voxceleb` |
| **Semantic Analysis** | Sentence-Transformers · Scikit-learn |
| **GUI** | PySide6 (Qt6) |
| **Audio/Video** | FFmpeg · PyDub · sounddevice · webrtcvad |
| **Deep Learning** | PyTorch (CPU/CUDA) |

---

## 📤 Output Formats

Each transcription session produces files in `app/output/<filename>/`:

- **`.md`** — Human-readable Markdown with speaker labels and timestamps
- **`.json`** — Structured data for downstream processing
- **`.srt`** — Subtitle file compatible with any media player

Converted videos land in `app/output/Converted/` as `.wav` files.

---

## 🛡️ License

MIT License — see [`LICENSE`](LICENSE) for details.

This project is part of the **Neuromicon** ecosystem. See [`tech.md`](tech.md) for full technical specifications.

---

*Made with ❤️ for researchers, journalists, and power users.*
