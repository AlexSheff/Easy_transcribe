# 🎙️ Easy TScribe

![Easy Transcriber Logo](assets/logo.png)

Local audio/video transcription with speaker diarization and Markdown export.
Recognition and diarization run on your machine with models from your Hugging Face cache.
The UI makes **no requests to external hosts**: no CDN scripts, no web fonts, no model downloads.

---

## What it does

- **Transcription** with [faster-whisper](https://github.com/SYSTRAN/faster-whisper): models `tiny`, `base`, `small`, `medium` (and any other `Systran/faster-whisper-*` model in your cache). Language auto-detect or fixed.
- **Speaker diarization** with SpeechBrain ECAPA-TDNN embeddings + agglomerative clustering. Speakers are numbered by talk time (`Speaker 001` is the most active). Rename or merge them in the Speaker Manager.
- **Batch queue**, in-browser media → 16 kHz WAV converter, microphone recorder, inline segment editor, audio scrubber.
- **Export**: Markdown (several presets), SRT, JSON, plain text. Transcripts are also saved automatically to `./transcripts/`.
- **Optional in-browser backend** (transformers.js, no Python). Advanced: you must place the model files in `public/models/`. It has **no diarization** (single speaker).

### Known limitations

- Diarization quality depends on the ECAPA model being present. Without it the server falls back to a weak acoustic diarizer and `/health` reports `acoustic-multifeature`. Overlapping speech is not handled.
- The number of speakers is chosen automatically between 1 and **Max Speakers** (default 3, set in Engine Settings) using a silhouette score on the longer segments; it can never exceed the maximum. If it under-splits a conversation with a known number of people, tick "Exactly this many". If it splits a single speaker, raise `EASY_TSCRIBE_MIN_SILHOUETTE` (default `0.08`).
- Speaker gender is **not** guessed. Semantic topic clustering was removed because it was not real.
- Diarization of very short segments (< 0.6 s) is inherited from neighbouring segments.

---

## Setup

Requirements: Node.js 18+, Python 3.10+, and models in your Hugging Face cache
(`%USERPROFILE%\.cache\huggingface\hub` on Windows, `~/.cache/huggingface/hub` elsewhere; override with `HF_HUB_CACHE`).

```bash
npm install
pip install -r requirements.txt

# terminal 1: local engine (127.0.0.1:8000)
python server_faster_whisper.py            # default model: medium
python server_faster_whisper.py small      # or another cached model / a model directory

# terminal 2: UI
npm run dev                                # http://localhost:3000
```

**Windows:** double-click `start.bat` (checks Node and Python packages, starts both).

The model dropdown in the UI selects which cached faster-whisper model the server uses per request.
`GET http://127.0.0.1:8000/health` lists the models the server found.

### Server configuration (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `HF_HUB_CACHE` / `HF_HOME` | `~/.cache/huggingface/hub` | Where cached models live |
| `EASY_TSCRIBE_ORIGINS` | – | Extra allowed browser origins (comma separated) |
| `EASY_TSCRIBE_MAX_UPLOAD_MB` | `1024` | Upload size limit |
| `EASY_TSCRIBE_MAX_SPEAKERS` | `3` | Default upper bound on speakers (the UI setting overrides it) |
| `EASY_TSCRIBE_MIN_SILHOUETTE` | `0.08` | How clearly voices must separate before splitting into more than one speaker |

## Security model

- The Python server binds to `127.0.0.1` only, accepts browser requests only from `http://localhost:3000` / `http://127.0.0.1:3000`, and rejects non-loopback `Host` headers (DNS-rebinding protection).
- The Vite dev server listens on `localhost` only.
- `transcripts/` is git-ignored. Do not commit recordings or transcripts.

## Development

```bash
npm run lint      # TypeScript check
npm run build     # production build
pip install -r requirements-dev.txt
npm test          # server tests (use a fake Whisper model, no GPU or models needed)
```

## Project structure

```
Easy_transcribe/
├── server_faster_whisper.py   # local engine: faster-whisper + ECAPA diarization + transcript saving
├── tests/test_server.py       # server tests
├── scripts/copy-ort-wasm.js   # copies ONNX WASM locally for the in-browser backend
├── src/
│   ├── App.tsx                # app state and layout
│   ├── components/            # Sidebar, TranscriptView, BatchQueuePanel, modals...
│   └── services/              # transcriptionEngine, audioConverter, exporter, whisperLoader
├── start.bat / run.bat / install.bat
└── requirements.txt
```

## License

MIT
