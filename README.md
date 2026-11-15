# 🎙️ Easy TScribe

![Easy Transcriber Logo](assets/logo.png)

Local audio/video transcription with speaker diarization and Markdown export.
Recognition and diarization run on your machine with models from your Hugging Face cache.
The UI makes **no requests to external hosts**: no CDN scripts, no web fonts, no model downloads.

---

## What it does

- **Transcription** with [faster-whisper](https://github.com/SYSTRAN/faster-whisper): models `tiny`, `base`, `small`, `medium`, `large-v3-turbo`, and `large-v3` (plus any other `Systran/faster-whisper-*` model in your cache).
- **Language selection** (`auto` or explicit language code such as `ru`, `en`, `uk`, `de`, `fr`, etc.) selectable directly in the sidebar.
- **Direct server-side media decoding & normalization**: audio/video files (`MP4`, `MKV`, `AVI`, `MOV`, `MP3`, `WAV`, `FLAC`, `M4A`, `OGG`, `WEBM`) stream straight to the local server without slow browser Web Audio decoding; optional `-18 dBFS` loudness normalization boosts quiet recordings before VAD.
- **Speed & accuracy tuning**:
  - `BatchedInferencePipeline` toggle for GPU throughput acceleration (automatic fallback to standard `transcribe` when unavailable).
  - Selectable quantization / `compute_type` (`auto`, `float16`, `int8_float16`, `int8`).
  - Tunable Silero VAD **Speech Threshold** and **Min Silence (Turns)**, optional `initial_prompt` (names/terminology) and `hotwords`, plus `hallucination_silence_threshold` when word timestamps are active.
- **Speaker diarization** with SpeechBrain ECAPA-TDNN embeddings (mini-batch inference + sliding 2.5s windows for long utterances) + agglomerative clustering with configurable **Split Sensitivity** (`min_silhouette`). Speakers are numbered by talk time (`Speaker 001` is the most active). Rename/merge speakers globally or reassign individual segments inline.
- **Real-time progress, cancellation & saved outputs**:
  - Live server progress bar (`GET /api/progress`) and one-click **Cancel** (`POST /api/cancel`).
  - Automatic saving to `./outputs/` and `./transcripts/` using your chosen Markdown preset (`standard`, `obsidian`, `meeting`, `clean`, `timestamps`).
  - Batch queue automatically skips files already transcribed in `outputs/` or `transcripts/`, with **Open .MD** and **Re-run** controls, plus a **Browse Saved .MD** archive modal.
  - Interactive word-level playback (click any word to seek, live word highlight, dotted underline for low-confidence words).
- **Optional in-browser backend** (transformers.js, no Python). Advanced: place model files in `public/models/`. It has **no diarization** (single speaker).

### Known limitations

- Diarization quality depends on the ECAPA model being present. Without it the server falls back to a weak acoustic diarizer and `/health` reports `acoustic-multifeature`. Overlapping speech is not handled.
- Decoding non-WAV containers (`.mp4`, `.mkv`, `.mp3`, etc.) on the Python server uses PyAV (`av`, installed with `faster-whisper`) or `ffmpeg` if available on `PATH`.
- The number of speakers is chosen automatically between 1 and **Max Speakers** (default 3, set in Engine Settings) using a silhouette score on the longer segments; it can never exceed the maximum. Adjust **Split Sensitivity** (`min_silhouette`, default `0.08`) or tick **Exact speaker count** when needed.
- Speaker gender is **not** guessed.
- Diarization of very short segments (< 0.6 s) is inherited from the temporally nearest embedded segment.

---

## Setup

Requirements: Node.js 18+, Python 3.10+, and models in your Hugging Face cache
(`%USERPROFILE%\.cache\huggingface\hub` on Windows, `~/.cache/huggingface/hub` elsewhere; override with `HF_HUB_CACHE`).

```bash
npm install
pip install -r requirements.txt

# terminal 1: local engine (127.0.0.1:8000)
python server_faster_whisper.py            # default model: medium
python server_faster_whisper.py large-v3-turbo

# terminal 2: UI
npm run dev                                # http://localhost:3000
```

**Windows:** double-click `start.bat` (checks Node and Python packages, starts both).

The model dropdown in the UI selects which cached faster-whisper model the server uses per request.
`GET http://127.0.0.1:8000/health` lists the models the server found.

### Server endpoints

- `GET /health` — server status, cached Whisper models, active diarizer (`speechbrain-ecapa` or `acoustic-multifeature`).
- `POST /transcribe` — streams raw media, runs transcription + diarization, auto-saves `.md`. Supports `model`, `language`, `device`, `compute_type`, `beam_size`, `min_silence_ms`, `vad_threshold`, `max_speakers`, `exact_speakers`, `min_silhouette`, `use_batched`, `normalize_audio`, `word_timestamps`, `initial_prompt`, `hotwords`, `skip_existing`, `job_id`.
- `GET /api/progress?job_id=...` & `POST /api/cancel?job_id=...` — live job progress polling and cancellation.
- `GET /api/check-transcript` / `POST /api/check-transcripts` — checks if a transcript already exists in `outputs/` or `transcripts/`.
- `GET /api/transcripts` & `GET /api/transcript?path=...` — lists and reads saved transcripts safely within `outputs/` and `transcripts/`.
- `POST /api/save-markdown` — saves edited or formatted Markdown into `transcripts/` and `outputs/`.

### Server configuration (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `HF_HUB_CACHE` / `HF_HOME` | `~/.cache/huggingface/hub` | Where cached models live |
| `EASY_TSCRIBE_ORIGINS` | – | Extra allowed browser origins (comma separated) |
| `EASY_TSCRIBE_MAX_UPLOAD_MB` | `1024` | Upload size limit |
| `EASY_TSCRIBE_MAX_SPEAKERS` | `3` | Default upper bound on speakers (the UI setting overrides it) |
| `EASY_TSCRIBE_MIN_SILHOUETTE` | `0.08` | Default minimum silhouette score to split speakers (UI overrides it) |
| `EASY_TSCRIBE_OUTPUTS_DIR` | `./outputs` | Directory checked for existing transcripts and mirrored saves |

## Security model

- The Python server binds to `127.0.0.1` only, accepts browser requests only from `http://localhost:3000` / `http://127.0.0.1:3000`, and rejects non-loopback `Host` headers (DNS-rebinding protection).
- File reads and writes under `outputs/` and `transcripts/` resolve canonical paths (`os.path.realpath`) to prevent directory traversal.
- `transcripts/` and `outputs/` should not be committed with private recordings or transcripts.

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
│   ├── App.tsx                # app state, queue runner, cancellation & layout
│   ├── components/            # Sidebar, TranscriptView, BatchQueuePanel, SavedOutputsModal, modals...
│   └── services/              # transcriptionEngine, audioConverter, exporter, whisperLoader
├── start.bat / run.bat / install.bat
└── requirements.txt
```

## License

MIT
