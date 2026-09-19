# 🎙️ Easy Transcriber (NEUROMICON Node)

![Easy Transcriber Logo](assets/logo.png)

**Easy Transcriber** is a high-performance, fully offline application for automated audio/video transcription, speaker diarization, voice fingerprinting, and semantic clustering. Built with a sleek cyberpunk terminal interface inspired by the Neuromicon aesthetic, it ensures 100% data privacy — no audio or text ever leaves your machine.

---

## ✨ Key Features

- 🔒 **100% Offline & Private** — Client-side DSP and Web Audio processing. No cloud uploads, no subscriptions, no tracking.
- 🎙️ **Real-time Microphone Recording** — Live mic capture with real-time Voice Activity Detection (VAD) and animated audio frequency spectrum visualizer.
- 🎬 **Universal Media to WAV Converter** — Built-in offline audio pipeline: drops in video/audio containers (MP4, MKV, AVI, MOV, WEBM, MP3, M4A, FLAC) and resamples/downmixes directly to standardized 16kHz Mono 16-bit PCM WAV.
- 🌐 **Bilingual (Russian & English) Support** — Automatic detection of language and file naming conventions with localized speaker labeling and cluster summaries.
- 🗣️ **Speaker Diarization & Voice Matching** — Utterance-chunked agglomerative clustering with dynamic fundamental frequency ($F_0$) pitch tracking for conversational turn-taking detection.
- ⚧ **Gender Detection via F0 Pitch Analysis** — Offline pitch estimation via autocorrelation with fundamental frequency analysis ($F_0$ threshold at 165Hz) to distinguish Male and Female voices without requiring external models or internet access.
- 🧬 **Per-Session Speaker Database** — Manages identified speaker profiles, turn counts, and average pitch. Allows renaming speakers and merging split identities in one click. Automatically cleared per session for clean identification.
- 🧠 **Semantic Clustering** — Groups conversational turns into high-level thematic topic clusters with auto-summaries.
- 📝 **Structured Intelligence Export** — One-click export to Markdown (`.md`), Subtitles (`.srt`), structured JSON (`.json`), or clean Plain Text (`.txt`) with UTF-8 encoding.
- ⚡ **Interactive Scrubber Player** — Synchronized playback scrubber with clickable timestamp jumps to replay specific dialogue turns.

---

## 🚀 Quick Start (Windows)

1. **Prerequisites:** Install [Node.js (v18, v20 or newer LTS)](https://nodejs.org/). Make sure the option **"Add to PATH"** is checked during installation.
2. **Install Dependencies:**
   - Double-click **`install.bat`** (or run `npm install` in your terminal).
   - *If run as Administrator or from another folder, `install.bat` automatically switches to the project root and detects standard Node.js installation paths.*
3. **Launch Application:**
   - Double-click **`run.bat`** (or run `npm run dev`).
   - The launcher will automatically open your default browser at `http://localhost:3000`.

---

## 💻 Manual Launch (Cross-Platform: Windows / Linux / macOS)

```bash
# 1. Clone the repository
git clone https://github.com/AlexSheff/Easy_transcribe.git
cd Easy_transcribe

# 2. Install dependencies
npm install

# 3. Start local development server
npm run dev
```

Once running, navigate to `http://localhost:3000` in any modern web browser (Chrome, Edge, Firefox, Safari).

---

## 🛠️ Troubleshooting & Tips

- **`install.bat` says Node.js is not found**:
  - Verify that Node.js is installed from [nodejs.org](https://nodejs.org/).
  - Restart your command prompt or terminal after installing Node.js so that the system environment variables update.
  - The script now automatically inspects `%ProgramFiles%\nodejs` and `%LocalAppData%\Programs\node\nodejs`.
- **Browser Audio Permissions**:
  - When using the real-time recording feature, grant microphone permissions to the browser.
- **Large Files**:
  - Processing happens completely locally within the browser Web Audio sandbox. For multi-hour files, 16kHz WAV conversion ensures minimal memory overhead.

---

## 📂 Project Structure

```
Easy_transcribe/
├── assets/
│   └── logo.png                  # Application branding logo
├── src/
│   ├── components/
│   │   ├── Sidebar.tsx           # Whisper engine selector & primary actions
│   │   ├── StatCards.tsx         # Dashboard counters (Queue, Processed, Active Model)
│   │   ├── StatusBox.tsx         # Operation status message and animated progress bar
│   │   ├── TerminalOutput.tsx    # Live system console log stream with color-coded tags
│   │   ├── TranscriptView.tsx    # Audio player, timestamped turns, inline segment editor
│   │   ├── ConverterModal.tsx    # Universal media-to-WAV converter
│   │   ├── RecorderModal.tsx     # Microphone capture with VAD & live spectrum
│   │   ├── SpeakerManagerModal.tsx # Speaker database, rename, pitch stats & merge
│   │   ├── SemanticViewModal.tsx # Thematic cluster cards and topic grouping
│   │   └── ExportModal.tsx       # Multi-format structured export (MD, SRT, JSON, TXT)
│   ├── services/
│   │   ├── audioConverter.ts     # In-browser Web Audio API decoding and WAV encoder
│   │   ├── pitchAnalyzer.ts      # Autocorrelation F0 pitch analysis & gender detection
│   │   ├── exporter.ts           # Markdown, SRT, and JSON format generators
│   │   └── transcriptionEngine.ts # Offline pipeline orchestrator & speech intervals
│   ├── types.ts                  # Shared TypeScript interfaces & types
│   ├── App.tsx                   # Main application layout and state management
│   ├── main.tsx                  # React entry point
│   └── index.css                 # Global styles & Tailwind configuration
├── install.bat                   # One-click Windows dependency installer
├── run.bat                       # One-click Windows application launcher
├── package.json                  # Scripts and dependencies
├── vite.config.ts                # Vite build and development configuration
└── README.md                     # Documentation
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Core Framework** | React 18 & TypeScript |
| **Build & Bundler** | Vite |
| **Styling & Theme** | Tailwind CSS (Neuromicon Dark Terminal theme) |
| **Audio Processing (DSP)** | Browser Web Audio API (`AudioContext`, `OfflineAudioContext`, `AnalyserNode`) |
| **Diarization & Pitch** | Autocorrelation Fundamental Frequency ($F_0$) & Agglomerative Clustering |
| **Icons** | Lucide React |

---

## 📤 Output Format Example

Transcripts can be exported into Markdown matching the original structure:

```markdown
# Transcription: interview_clip.wav
- **Generated**: 2026-09-19 15:30:00
- **Duration**: 00:01:24
- **Engine**: Whisper BASE
- **Identified Speakers**: 2

---

## Transcript

**00:00:02** (Male Speaker 001 - 128Hz): Welcome to the Neuromicon briefing.
**00:00:06** (Female Speaker 002 - 215Hz): Glad to be here. The offline engine has processed the incoming stream.
```

---

## 🛡️ License

MIT License — see [`LICENSE`](LICENSE) for details.

*Part of the **Neuromicon** ecosystem.*
