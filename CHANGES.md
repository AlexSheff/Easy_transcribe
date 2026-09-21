# Changes in this revision

**Privacy / offline guarantee**
- Removed the jsDelivr CDN fallback and the `new Function` dynamic import (whisperLoader).
- ONNX Runtime WASM is now served locally (`public/ort-wasm`, copied by `scripts/copy-ort-wasm.js`) instead of the library's CDN default.
- Removed Google Fonts links from `index.html` (the app contacted Google on every load).
- `.gitignore` now covers `node_modules/`, `transcripts/`, `.env*`, audio/video files.

**Correctness**
- No more fake output: removed the "acoustic" backend that produced `[Audio segment #N]` text with alternating speakers, fabricated word timings, random/hard-coded confidence, positional "semantic clusters", and the F0-based Male/Female labels.
- Browser backend: `medium` is mapped to its own model, `large-v3` is rejected with a clear error instead of silently using `base`. Failures are shown as errors.
- Server now honours the language, model and device selected in the UI (the language was previously ignored). Model names map to your local Systran cache (tiny/base/small/medium).
- Confidence is now the real segment probability from Whisper.
- Diarization: the number of speakers is now capped (Max Speakers, default 3) and chosen by silhouette score on longer segments; other segments join the nearest speaker; tiny outlier clusters are merged. Previously a fixed distance threshold produced dozens of speakers on long recordings (45 on a 586-segment file). Segments < 0.6 s inherit the neighbouring speaker.
- `/health` response was malformed (CORS headers sent before the status line); fixed.

**Robustness**
- A browser that disconnects while the server is sending the result (WinError 10053) no longer causes a traceback and a bogus 500; the transcript is already saved in `transcripts/`.

**Security**
- Python server: CORS limited to http://localhost:3000 / http://127.0.0.1:3000, Host header check (DNS rebinding), upload size limit, streamed uploads, threaded server with serialized inference, no local paths in `/health`, safe file writes into `transcripts/`.
- Vite dev server: `localhost` only (was `0.0.0.0`), unauthenticated file-write middleware removed (saving is done by the Python server).

**Housekeeping**
- Removed hard-coded `C:\Users\admin_fdr\...` paths, Gemini leftovers, unused pyannote/GigaAM code paths, dead modules (`pitchAnalyzer`, `SemanticViewModal`), `bun.lock` (npm is the documented tool; `package-lock.json` added).
- Added `requirements.txt`, `requirements-dev.txt`, server tests (10), README rewritten to match reality.

**Not done (needs your machine / real audio)**
- Replacing the diarizer with a stronger pipeline (pyannote 3.1 needs models you don't have cached; only `segmentation-3.0` is present).
- Tuning `EASY_TSCRIBE_MIN_SILHOUETTE` on real recordings (default is a starting value; tests use synthetic embeddings); testing the in-browser backend end to end.
- `npm audit` findings from `@xenova/transformers` (transitive `sharp`/`onnx-proto`; Node-side, not in the browser bundle).
