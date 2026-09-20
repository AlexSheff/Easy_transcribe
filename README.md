# 🎙️ Easy TScribe

![Easy Transcriber Logo](assets/logo.png)

**Easy TScribe** is a professional, 100% private, and fully offline suite for automated audio/video transcription, neural speaker diarization, voice identification, and Markdown document generation. 

Everything runs strictly on your machine — zero cloud uploads, zero external API keys, and 100% data confidentiality.

---

## ✨ Key Capabilities

- 🔒 **100% Offline & Private** — Audio and transcripts never leave your machine (`HF_HUB_OFFLINE=1` enforced).
- 🚀 **Unified Single-Click Launcher (`start.bat`)** — Automatically verifies Node.js, installs dependencies if needed, starts the background Whisper/diarization engine, creates the `transcripts/` folder, and launches the UI in your browser.
- 📁 **Automatic Markdown Export to Project Folder** — Every completed audio/video transcription is automatically saved as a structured Markdown (`.md`) document inside the `./transcripts/` project folder.
- 🗣️ **Neural Speaker Diarization** — Distinguishes speakers using acoustic embeddings and agglomerative cosine clustering with fundamental frequency ($F_0$) pitch tracking.
- ⚡ **Batch Processing Queue** — Drag and drop multiple audio/video files at once (MP4, MKV, AVI, MOV, MP3, WAV, FLAC, M4A, OGG). Track progress per file with real-time status and progress indicators.
- 📝 **Standard Clean Markdown Preset** — Generates standardized transcripts with full header metadata, speaker turn counts, timestamps, and dialogues.
- 🎛️ **Audio Converter & Live Mic Recorder** — Built-in converter to standardized 16kHz mono WAV and live voice capture with real-time spectrum visualization.
- 🖥️ **Modern Non-Overloaded UI** — Intuitive dashboard with 3 primary metric cards, system status bar, terminal logs, interactive playback scrubber, inline segment editor, and quick modal tools.

---

## 🚀 Quick Start (Windows)

Simply double-click:

```
start.bat
```

The launcher will:
1. Check for **Node.js** (prompts if missing).
2. Install npm dependencies automatically on first run.
3. Ensure the **`transcripts/`** folder exists in the project root.
4. Auto-detect Python and launch the local **`faster-whisper` + Diarization** daemon in the background.
5. Open **`http://localhost:3000`** in your default web browser and start the application.

---

## 💻 Manual Launch (Cross-Platform: Windows / Linux / macOS)

```bash
# 1. Install dependencies
npm install

# 2. (Optional) Run the local Python faster-whisper daemon
python server_faster_whisper.py

# 3. Start the application
npm run dev
```

Open `http://localhost:3000` in your web browser.

---

## 📂 Auto-Saved Output Format

Transcripts are automatically written to `./transcripts/transcript_<filename>.md`:

```markdown
# Transcript: 2026-09-04 02-17-31.mp4
Date: 2026-09-20 21:38:42
Total Segments: 117

## Speakers

- Male Speaker 001: 84 segment(s)
- Male Speaker 002: 31 segment(s)
- Male Speaker 003: 2 segment(s)

## Transcript

**[00:02] Male Speaker 001:** Initial dialogue statement recorded here.
**[00:15] Male Speaker 002:** Response from second speaker with detected turn-taking.
```

---

## 📂 Project Structure

```
Easy_transcribe/
├── assets/
│   └── logo.png                  # Application branding
├── transcripts/                  # Auto-saved Markdown transcripts (.md)
├── src/
│   ├── components/
│   │   ├── Sidebar.tsx           # Model selector, primary actions, and status
│   │   ├── StatCards.tsx         # Dashboard metrics (Queue, Processed, Active Model)
│   │   ├── StatusBox.tsx         # Operation status & progress bar
│   │   ├── TerminalOutput.tsx    # Live system console log stream
│   │   ├── TranscriptView.tsx    # Audio player, timestamped turns & inline editor
│   │   ├── BatchQueuePanel.tsx   # Batch processing queue and batch export
│   │   ├── ConverterModal.tsx    # Media-to-WAV converter
│   │   ├── RecorderModal.tsx     # Microphone capture with VAD & live spectrum
│   │   ├── SpeakerManagerModal.tsx # Speaker database, rename & merge tools
│   │   ├── SemanticViewModal.tsx # Thematic topic clusters
│   │   ├── ModelSettingsModal.tsx # Whisper & VAD silence thresholds
│   │   └── ExportModal.tsx       # Standard, Obsidian, SRT, and JSON exports
│   ├── services/
│   │   ├── audioConverter.ts     # In-browser Web Audio API decoding and WAV encoder
│   │   ├── pitchAnalyzer.ts      # Autocorrelation F0 pitch analysis
│   │   ├── exporter.ts           # Standard & Obsidian Markdown generators
│   │   └── transcriptionEngine.ts # Offline pipeline orchestrator & Python bridge
│   ├── types.ts                  # Shared TypeScript types
│   ├── App.tsx                   # Main application layout and state management
│   ├── main.tsx                  # React entry point
│   └── index.css                 # Global styling
├── server_faster_whisper.py      # Local faster-whisper & Pyannote diarization server
├── start.bat                     # Single master launcher for Windows
├── package.json                  # Scripts and dependencies
├── vite.config.ts                # Vite config with auto-save markdown plugin
└── README.md                     # Documentation
```

---

## 🛡️ Privacy Guarantee

- **No Remote Network Calls**: Models operate from local disk caches (`%USERPROFILE%\.cache\huggingface\hub`).
- **No Cloud Dependencies**: Audio conversion, VAD, transcription, and diarization run entirely on local compute.
- **Local Persistence**: Markdown files are saved directly into the local `transcripts/` directory.

---

## 📄 License

MIT License. Designed for high productivity and privacy-first transcription.

