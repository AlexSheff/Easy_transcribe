# TECHNICAL SPECIFICATION

## LOCAL GUI TRANSCRIPTION & ANALYSIS SYSTEM

**Codename:** Neuromicon Transcriber Node
**Version:** 1.0
**Status:** Implementation Ready

---

# 1. PURPOSE

Design and implement a **local desktop GUI application** that performs:

* MP4 → Audio extraction (MP3/WAV)
* Multilingual transcription (RU / EN / TL / CEB)
* Speaker diarization
* Voice fingerprinting (speaker identity persistence)
* Semantic clustering (meaning-based segmentation)
* Markdown export with structured intelligence layers

System must operate **fully offline**, GPU-optional.

---

# 2. SYSTEM OVERVIEW

## 2.1 Core Functional Modules

```
[Input Layer]
    ↓
[Audio Processing]
    ↓
[ASR Engine]
    ↓
[Diarization Engine]
    ↓
[Voice Fingerprint Engine]
    ↓
[Semantic Analysis Engine]
    ↓
[Output Generator]
    ↓
[GUI Visualization Layer]
```

---

# 3. FUNCTIONAL REQUIREMENTS

## 3.1 File Handling

### Input

* Supported formats:

  * `.mp4`, `.mkv`, `.avi`
  * `.mp3`, `.wav`, `.flac`
* Batch processing support
* Drag & Drop interface

### Output

* `.md` (primary)
* `.json` (structured data)
* `.srt` (optional subtitles)

---

## 3.2 Audio Processing

* Extract audio via FFmpeg
* Normalize:

  * mono channel
  * 16kHz sampling rate
* Optional:

  * noise reduction
  * silence trimming (VAD)

---

## 3.3 ASR (Speech Recognition)

### Requirements:

* Multilingual auto-detection
* Support:

  * Russian
  * English
  * Tagalog
  * Cebuano (best-effort)

### Output format:

```
{
  start,
  end,
  text,
  language
}
```

---

## 3.4 Speaker Diarization

### Requirements:

* Detect speaker segments
* Assign temporary labels:

  * SPEAKER_00, SPEAKER_01...

### Output:

```
{
  start,
  end,
  speaker_id
}
```

---

## 3.5 Voice Fingerprint System

## 3.5.1 Objective

Persistently identify speakers across sessions.

---

## 3.5.2 Functional Logic

1. Extract speaker embeddings from audio segments
2. Compare embeddings with stored database
3. If match found:

   * assign existing identity
4. If not:

   * create new identity

---

## 3.5.3 Data Structure

```
voice_db/
    speaker_001/
        embedding.npy
        metadata.json
    speaker_002/
        ...
```

---

## 3.5.4 Matching Algorithm

* Cosine similarity
* Threshold: configurable (default: 0.75)

---

## 3.5.5 Metadata

```
{
  "name": "Optional user-defined",
  "first_seen": timestamp,
  "confidence": score
}
```

---

## 3.5.6 GUI Features

* Rename speakers
* Merge identities
* View confidence score
* Replay voice samples

---

# 4. SEMANTIC CLUSTERING ENGINE

## 4.1 Objective

Transform raw transcript into **meaningful blocks**.

---

## 4.2 Input

Segmented transcript:

```
[time, speaker, text]
```

---

## 4.3 Processing Pipeline

1. Sentence embedding generation
2. Similarity comparison
3. Clustering
4. Topic labeling

---

## 4.4 Output Structure

```
[
  {
    "cluster_id": 1,
    "topic": "Introduction",
    "segments": [...]
  }
]
```

---

## 4.5 Algorithms

* Embeddings: sentence-transformers
* Clustering:

  * KMeans (default)
  * HDBSCAN (advanced)
* Distance metric:

  * cosine similarity

---

## 4.6 GUI Visualization

* Blocks grouped visually
* Expand / collapse clusters
* Highlight topic transitions
* Timeline view

---

# 5. MARKDOWN OUTPUT SPEC

## 5.1 Format

```
# Transcript

## Speaker: Alex (ID: speaker_001)

[00:00 - 00:05]
Hello everyone...

---

## Semantic Block: Introduction

[Speaker: Alex]
...

[Speaker: Unknown]
...
```

---

## 5.2 Layers

Markdown must include:

* Speaker sections
* Timecodes
* Semantic clusters
* Language tags (optional)

---

# 6. GUI REQUIREMENTS

## 6.1 Technology Stack

* Framework: PySide6 / PyQt6
* Optional: Electron + Python backend

---

## 6.2 Main Screens

### 1. Dashboard

* File upload
* Processing queue
* Status indicators

---

### 2. Transcription View

* Timeline (waveform optional)
* Color-coded speakers
* Click-to-play segments

---

### 3. Speaker Manager

* List of detected speakers
* Rename / merge / delete
* Confidence scores

---

### 4. Semantic View

* Cluster blocks
* Topic labels
* Expandable structure

---

### 5. Export Panel

* Export options:

  * Markdown
  * JSON
  * SRT

---

## 6.3 UX Requirements

* Non-blocking UI (async processing)
* Progress bars
* Error logs
* Resume interrupted jobs

---

# 7. NON-FUNCTIONAL REQUIREMENTS

## 7.1 Performance

* Real-time factor target:

  * GPU: 0.5x–1x
  * CPU: 2x–5x

---

## 7.2 Offline Capability

* No external API required
* Local model storage

---

## 7.3 Scalability

* Batch mode support
* Modular architecture

---

## 7.4 Extensibility

* Plugin-ready architecture:

  * new models
  * new exporters
  * new analytics

---

# 8. PROJECT STRUCTURE

```
app/
├── gui/
├── core/
├── models/
├── pipeline/
├── voice_db/
├── output/
└── config/
```

---

# 9. CONFIGURATION

## config.yaml

```
asr_model: large-v3
device: cuda
language: auto

diarization: enabled
voice_fingerprint: enabled

semantic_clustering:
  method: kmeans
  max_clusters: 10
```

---

# 10. RISK ANALYSIS

| Risk                 | Impact | Mitigation         |
| -------------------- | ------ | ------------------ |
| Cebuano accuracy low | Medium | post-correction    |
| Speaker overlap      | High   | refine diarization |
| GPU absence          | Medium | CPU fallback       |
| Long files           | High   | chunk processing   |

---

# 11. FUTURE EXTENSIONS

* Real-time transcription
* Emotion detection
* Knowledge graph extraction
* Integration with Neuromicon nodes
* Multi-user voice identity federation

---

# 12. ACCEPTANCE CRITERIA

System is considered complete if:

* ✔ Processes MP4 → Markdown end-to-end
* ✔ Correctly separates ≥2 speakers
* ✔ Re-identifies speaker across sessions
* ✔ Generates semantic clusters
* ✔ Runs fully offline
* ✔ GUI stable under batch load

---

# 13. DEPLOYMENT

## Local Run

```
pip install -r requirements.txt
python app.py
```

---

## Optional Build

* PyInstaller standalone executable
* Cross-platform:

  * Windows
  * Linux
  * macOS

---

# FINAL NOTE

This system is not just a transcriber.

It is a **cognitive extraction engine**:

* voice → identity
* speech → structure
* dialogue → meaning

This is the base layer for higher-order systems.
