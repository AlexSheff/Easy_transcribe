#!/usr/bin/env python3
"""
Easy TScribe - local speech recognition + speaker diarization server.

Runs 100% offline on 127.0.0.1 using models already present in the Hugging Face
hub cache:
    - Systran/faster-whisper-{tiny,base,small,medium,...}   (speech recognition)
    - speechbrain/spkrec-ecapa-voxceleb                     (speaker embeddings)

If the ECAPA model is not available, a much weaker built-in acoustic diarizer is
used and /health reports diarizer_type = "acoustic-multifeature".

Usage:
    python server_faster_whisper.py [DEFAULT_MODEL] [PORT]

    DEFAULT_MODEL  model name ("tiny", "base", "small", "medium", ...) or a path to a
                   CTranslate2 model directory. Default: medium.
    PORT           default 8000

Environment variables:
    HF_HUB_CACHE / HF_HOME            location of the Hugging Face cache
    EASY_TSCRIBE_ORIGINS              extra allowed browser origins (comma separated)
    EASY_TSCRIBE_MAX_UPLOAD_MB        request size limit (default 1024)
    EASY_TSCRIBE_MAX_SPEAKERS         upper bound on detected speakers (default 3)
    EASY_TSCRIBE_MIN_SILHOUETTE       how clearly separated speakers must be to split (default 0.08)
"""

import sys
import os
import time
import datetime
import traceback
import shutil
import json
import glob
import math
import wave
import tempfile
import threading
import urllib.parse
import warnings
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Suppress benign secondary warnings (torchvision image extension, SpeechBrain 1.0 migration deprecations)
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.*")
warnings.filterwarnings("ignore", message=".*image Python extension.*")
warnings.filterwarnings("ignore", message=".*speechbrain.pretrained.*")
warnings.filterwarnings("ignore", message=".*SYMLINK strategy on Windows.*")
warnings.filterwarnings("ignore", message=".*Pretrainer collection using symlinks.*")
warnings.filterwarnings("ignore", message=".*Lazy import of LazyModule.*")

# Fix Windows privilege requirement for symlinks (e.g. SpeechBrain / HuggingFace cache)
if sys.platform == "win32":
    def _win_symlink_copy(src, dst, *args, **kwargs):
        try:
            src_abs = os.path.abspath(str(src))
            dst_abs = os.path.abspath(str(dst))
            if src_abs == dst_abs:
                return
            if os.path.exists(dst):
                return
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)
        except Exception:
            pass
    os.symlink = _win_symlink_copy
    try:
        import pathlib
        pathlib.Path.symlink_to = lambda self, target, target_is_directory=False: _win_symlink_copy(target, self)
    except Exception:
        pass

# Enforce strict offline execution
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_MODEL = "medium"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRANSCRIPTS_DIR = os.path.join(BASE_DIR, "transcripts")

DEFAULT_ORIGINS = {"http://localhost:3000", "http://127.0.0.1:3000"}
ALLOWED_ORIGINS = set(DEFAULT_ORIGINS)
for _o in os.environ.get("EASY_TSCRIBE_ORIGINS", "").split(","):
    if _o.strip():
        ALLOWED_ORIGINS.add(_o.strip().rstrip("/"))

MAX_UPLOAD_BYTES = int(float(os.environ.get("EASY_TSCRIBE_MAX_UPLOAD_MB", "1024")) * 1024 * 1024)
DEFAULT_MAX_SPEAKERS = max(1, int(os.environ.get("EASY_TSCRIBE_MAX_SPEAKERS", "3")))
# Minimum silhouette score for accepting a split into more than one speaker.
MIN_SILHOUETTE = float(os.environ.get("EASY_TSCRIBE_MIN_SILHOUETTE", "0.08"))
# Clusters holding less than this share of the speech are treated as noise and merged.
MIN_CLUSTER_SHARE = 0.05
# Segments at least this long are used to find the speakers; shorter ones are assigned afterwards.
RELIABLE_EMBED_SEC = 1.0

# Segments shorter than this get no embedding of their own and inherit the neighbouring speaker.
MIN_EMBED_SEC = 0.6

SPEAKER_COLORS = ["#00ffcc", "#ff0077", "#ffcc00", "#0099ff", "#a855f7", "#22c55e", "#f97316"]


def get_hf_hub_root():
    """Hugging Face hub cache directory (HF_HUB_CACHE > HF_HOME/hub > ~/.cache/huggingface/hub)."""
    if os.environ.get("HF_HUB_CACHE"):
        return os.environ["HF_HUB_CACHE"]
    if os.environ.get("HF_HOME"):
        return os.path.join(os.environ["HF_HOME"], "hub")
    return os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")


HF_HUB = get_hf_hub_root()


def find_snapshot_dir(hub_root, repo_dir_name):
    """Locates the latest snapshot directory for a given model folder in the HF hub cache."""
    target_pattern = os.path.join(hub_root, repo_dir_name, "snapshots", "*")
    snapshots = sorted(glob.glob(target_pattern), key=os.path.getmtime, reverse=True)
    if snapshots and os.path.isdir(snapshots[0]):
        return snapshots[0]
    return None


def list_local_whisper_models(hub_root=None):
    """Names of Systran faster-whisper models present in the local cache."""
    hub_root = hub_root or HF_HUB
    prefix = "models--Systran--faster-whisper-"
    names = []
    for path in glob.glob(os.path.join(hub_root, prefix + "*")):
        name = os.path.basename(path)[len(prefix):]
        if find_snapshot_dir(hub_root, prefix + name):
            names.append(name)
    order = {"tiny": 0, "base": 1, "small": 2, "medium": 3}
    return sorted(names, key=lambda n: (order.get(n, 99), n))


class ModelNotFound(Exception):
    pass


def resolve_whisper_model(name):
    """Maps a model name ("small") or a directory path to something WhisperModel can load."""
    if not name:
        name = DEFAULT_MODEL
    if os.path.isdir(name):
        return name
    if not all(c.isalnum() or c in "._-" for c in name):
        raise ModelNotFound(f"Invalid model name: {name!r}")
    snap = find_snapshot_dir(HF_HUB, f"models--Systran--faster-whisper-{name}")
    if snap is None:
        available = ", ".join(list_local_whisper_models()) or "none"
        raise ModelNotFound(
            f"Whisper model '{name}' is not in the local Hugging Face cache ({HF_HUB}). "
            f"Available: {available}."
        )
    return snap


SNAPSHOT_SPEECHBRAIN_ECAPA = find_snapshot_dir(HF_HUB, "models--speechbrain--spkrec-ecapa-voxceleb")

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

default_model_ref = DEFAULT_MODEL
_whisper_cache = {}                     # (model_path, device) -> WhisperModel
_model_load_lock = threading.Lock()
TRANSCRIBE_LOCK = threading.Lock()      # one transcription at a time (GPU/RAM safety)

diarizer_model = None
diarizer_type = "none"


def pick_device(requested="auto"):
    if requested in ("cuda", "cpu"):
        return requested
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def get_whisper_model(model_ref, device="auto"):
    """Loads (and caches) a faster-whisper model."""
    model_path = resolve_whisper_model(model_ref)
    device = pick_device(device)
    key = (model_path, device)
    with _model_load_lock:
        if key in _whisper_cache:
            return _whisper_cache[key]

        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise RuntimeError("Package 'faster-whisper' is not installed. Run: pip install -r requirements.txt")

        compute_type = "float16" if device == "cuda" else "int8"
        print(f"[*] Loading Whisper '{model_path}' on {device} ({compute_type})...")
        try:
            model = WhisperModel(model_path, device=device, compute_type=compute_type, local_files_only=True)
        except Exception as e:
            print(f"[!] Falling back with compute_type='auto': {e}")
            model = WhisperModel(model_path, device=device, compute_type="auto", local_files_only=True)
        print("[+] Whisper model loaded (offline).")
        _whisper_cache[key] = model
        return model


def init_speaker_diarization():
    """Loads the local SpeechBrain ECAPA-TDNN speaker encoder if available."""
    global diarizer_model, diarizer_type
    print("[*] Inspecting local models for speaker diarization...")

    if SNAPSHOT_SPEECHBRAIN_ECAPA and os.path.exists(SNAPSHOT_SPEECHBRAIN_ECAPA):
        print(f"[*] Found SpeechBrain ECAPA-TDNN snapshot: {SNAPSHOT_SPEECHBRAIN_ECAPA}")
        try:
            try:
                from speechbrain.inference.classifiers import EncoderClassifier
            except ImportError:
                try:
                    from speechbrain.inference.speaker import EncoderClassifier
                except ImportError:
                    from speechbrain.pretrained import EncoderClassifier

            # Prevent symlink attempts and operate in place, fully offline
            def _offline_safe_link(src, dst, local_strategy=None):
                if os.path.abspath(str(src)) == os.path.abspath(str(dst)):
                    return str(dst)
                if os.path.exists(dst):
                    return str(dst)
                try:
                    shutil.copy2(src, dst)
                    return str(dst)
                except Exception:
                    return str(src)

            try:
                import speechbrain.utils.fetching as sb_fetch
                sb_fetch.link_with_strategy = _offline_safe_link
                if hasattr(sb_fetch, "LocalStrategy"):
                    sb_fetch.LocalStrategy.SYMLINK = sb_fetch.LocalStrategy.COPY
            except Exception:
                pass
            try:
                import speechbrain.utils.parameter_transfer as sb_pt
                if hasattr(sb_pt, "link_with_strategy"):
                    sb_pt.link_with_strategy = _offline_safe_link
            except Exception:
                pass

            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[*] Loading SpeechBrain ECAPA-TDNN on {device}...")

            hparams_kwargs = {
                "source": SNAPSHOT_SPEECHBRAIN_ECAPA,
                "savedir": SNAPSHOT_SPEECHBRAIN_ECAPA,
                "run_opts": {"device": device},
            }
            try:
                from speechbrain.utils.fetching import LocalStrategy
                hparams_kwargs["local_strategy"] = LocalStrategy.COPY
            except Exception:
                pass

            diarizer_model = EncoderClassifier.from_hparams(**hparams_kwargs)
            diarizer_type = "speechbrain-ecapa"
            print("[+] SpeechBrain ECAPA-TDNN speaker encoder loaded (offline).")
            return
        except Exception as err:
            print(f"[!] Could not load SpeechBrain ({err}).")

    diarizer_type = "acoustic-multifeature"
    print("[!] Using the built-in acoustic diarizer. It is a weak fallback: install speechbrain "
          "and cache speechbrain/spkrec-ecapa-voxceleb for real speaker embeddings.")


# ---------------------------------------------------------------------------
# Audio + diarization
# ---------------------------------------------------------------------------

def load_wav(wav_path):
    """Reads a PCM WAV once and returns (float32 mono samples, sample_rate)."""
    import numpy as np
    with wave.open(wav_path, "rb") as wf:
        framerate = wf.getframerate()
        nchannels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())

    if sampwidth == 2:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif sampwidth == 1:
        samples = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes")

    if nchannels > 1:
        samples = samples[::nchannels]
    return samples, framerate


def slice_samples(samples, sr, start_sec, end_sec):
    start = max(0, int(start_sec * sr))
    end = min(len(samples), int(end_sec * sr))
    return samples[start:max(start + 1, end)]


def compute_acoustic_features(samples, framerate):
    """24-dim acoustic descriptor (spectral shape + F0). Returns (features, f0)."""
    import numpy as np

    if len(samples) < int(framerate * 0.15):
        return np.zeros(24, dtype=np.float32), 0.0

    max_val = np.max(np.abs(samples))
    norm_samples = samples / max_val if max_val > 1e-5 else samples

    # F0 via autocorrelation
    min_period = int(framerate / 350)
    max_period = int(framerate / 65)
    f0 = 0.0
    if len(norm_samples) >= max_period * 2:
        window_samples = norm_samples[:max_period * 3]
        corr = np.correlate(window_samples, window_samples, mode="full")
        corr = corr[len(corr) // 2:]
        if max_period < len(corr):
            sub_corr = corr[min_period:max_period]
            if len(sub_corr) > 0 and np.max(sub_corr) > 0.05:
                peak_idx = np.argmax(sub_corr) + min_period
                f0 = float(framerate / peak_idx)

    n_fft = min(2048, len(norm_samples))
    window = np.hanning(n_fft)
    spec = np.abs(np.fft.rfft(norm_samples[:n_fft] * window))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / framerate)

    total_power = np.sum(spec) + 1e-8
    centroid = np.sum(freqs * spec) / total_power
    spread = np.sqrt(np.sum(((freqs - centroid) ** 2) * spec) / total_power)

    band_edges = np.linspace(50, 7500, 17)
    band_energies = []
    for b in range(16):
        mask = (freqs >= band_edges[b]) & (freqs < band_edges[b + 1])
        band_energies.append(np.sum(spec[mask]) / total_power)

    features = np.zeros(24, dtype=np.float32)
    features[0] = f0 / 250.0
    features[1] = centroid / 3000.0
    features[2] = spread / 1500.0
    features[3] = float(np.mean(np.abs(norm_samples)))
    features[4:20] = band_energies
    features[20] = float(np.std(norm_samples))
    features[21] = float(np.max(norm_samples) - np.min(norm_samples))
    features[22] = float(np.sum(np.abs(np.diff(np.sign(norm_samples)))) / len(norm_samples))
    features[23] = f0 / 120.0
    return features, f0


def embed_segment(samples, sr):
    """Speaker embedding + F0 for one segment, or (None, 0.0) if the segment is too short."""
    import numpy as np

    if samples is None or len(samples) < int(sr * MIN_EMBED_SEC):
        return None, 0.0

    features, f0 = compute_acoustic_features(samples, sr)

    if diarizer_type == "speechbrain-ecapa" and diarizer_model is not None:
        try:
            import torch
            tensor_audio = torch.from_numpy(samples).unsqueeze(0)
            with torch.no_grad():
                emb = diarizer_model.encode_batch(tensor_audio).squeeze().cpu().numpy()
            norm = np.linalg.norm(emb)
            if norm > 1e-6:
                emb = emb / norm
            return emb, f0
        except Exception as e:
            print(f"[!] ECAPA embedding failed for a segment, using acoustic features: {e}")

    norm = np.linalg.norm(features)
    if norm > 1e-6:
        features = features / norm
    return features, f0


def _cosine_sims(matrix, centroids):
    import numpy as np
    m = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)
    c = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-9)
    return m @ c.T


def cluster_speakers(embeddings, durations, max_speakers, exact=False):
    """
    Assigns a speaker label to every embedding, never using more than `max_speakers` labels.

    1. The speakers are found on the longer (more reliable) segments only.
    2. The number of speakers k in [1, max_speakers] is chosen by silhouette score
       (or fixed to max_speakers when `exact` is set).
    3. Every other segment is assigned to the nearest speaker centroid.
    4. In auto mode, clusters with a tiny share of the speech are merged into the nearest one.
    """
    import numpy as np

    n = len(embeddings)
    if n == 0:
        return []
    max_speakers = max(1, int(max_speakers))
    if n < 2 or max_speakers == 1:
        return [0] * n

    try:
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics import silhouette_score
    except ImportError:
        print("[!] scikit-learn is not installed; all speech assigned to one speaker. "
              "Run: pip install -r requirements.txt")
        return [0] * n

    X = np.array(embeddings, dtype=np.float64)
    dur = np.array(durations, dtype=np.float64)

    reliable = np.where(dur >= RELIABLE_EMBED_SEC)[0]
    if len(reliable) < max(4, max_speakers + 1):
        reliable = np.arange(n)
    Xr = X[reliable]
    k_max = min(max_speakers, len(reliable) - 1)
    if k_max < 2:
        return [0] * n

    def fit(k):
        return AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average").fit_predict(Xr)

    if exact:
        best_k, best_labels = k_max, fit(k_max)
    else:
        best_k, best_labels, best_score = 1, np.zeros(len(reliable), dtype=int), -1.0
        for k in range(2, k_max + 1):
            labels_k = fit(k)
            if len(set(labels_k)) < 2:
                continue
            score = float(silhouette_score(Xr, labels_k, metric="cosine"))
            print(f"    k={k}: silhouette={score:.3f}")
            if score > best_score:
                best_k, best_labels, best_score = k, labels_k, score
        if best_score < MIN_SILHOUETTE:
            print(f"[*] Speakers are not clearly separable (best silhouette {best_score:.3f} < {MIN_SILHOUETTE}); using 1 speaker.")
            return [0] * n
        print(f"[*] Selected {best_k} speaker(s) (silhouette {best_score:.3f}, max {max_speakers}).")

    centroids = np.array([Xr[best_labels == c].mean(axis=0) for c in range(best_k)])
    labels = _cosine_sims(X, centroids).argmax(axis=1)

    if not exact and best_k > 1:
        # Merge clusters that hold only a sliver of the speech (usually noise / outliers).
        while True:
            present = sorted(set(labels.tolist()))
            if len(present) < 2:
                break
            shares = {c: dur[labels == c].sum() for c in present}
            total = sum(shares.values()) or 1.0
            smallest = min(present, key=lambda c: shares[c])
            if shares[smallest] / total >= MIN_CLUSTER_SHARE:
                break
            others = [c for c in present if c != smallest]
            cents = np.array([X[labels == c].mean(axis=0) for c in others])
            own = X[labels == smallest].mean(axis=0, keepdims=True)
            target = others[int(_cosine_sims(own, cents).argmax())]
            labels[labels == smallest] = target

    # Compact label ids
    remap = {c: i for i, c in enumerate(sorted(set(labels.tolist())))}
    return [remap[int(c)] for c in labels]


def perform_speaker_diarization(samples, sr, raw_segments, max_speakers=None, exact=False):
    """
    Groups transcript segments into speakers ("Speaker 001" = the one with most segments).
    Segments too short to embed inherit the label of the nearest embedded segment.
    Returns (segments, speaker_metadata).
    """
    import numpy as np

    if not raw_segments:
        return [], {}
    max_speakers = DEFAULT_MAX_SPEAKERS if max_speakers is None else max_speakers

    print(f"[*] Computing speaker embeddings for {len(raw_segments)} segments "
          f"(engine: {diarizer_type}, max speakers: {max_speakers}{', exact' if exact else ''})...")

    embeddings, durations, pitches, embedded_idx = [], [], [], []
    for idx, seg in enumerate(raw_segments):
        chunk = slice_samples(samples, sr, seg["start"], seg["end"])
        emb, f0 = embed_segment(chunk, sr)
        pitches.append(f0)
        if emb is not None:
            embeddings.append(emb)
            durations.append(len(chunk) / sr)
            embedded_idx.append(idx)

    labels = [0] * len(raw_segments)
    if embeddings:
        emb_labels = cluster_speakers(embeddings, durations, max_speakers, exact)
        assigned = {i: lab for i, lab in zip(embedded_idx, emb_labels)}
        for idx in range(len(raw_segments)):
            if idx in assigned:
                labels[idx] = assigned[idx]
            else:
                # Too short to embed: inherit from the nearest embedded segment.
                nearest = min(embedded_idx, key=lambda j: abs(j - idx))
                labels[idx] = assigned[nearest]

    cluster_counts = Counter(labels)
    sorted_clusters = [c for c, _ in cluster_counts.most_common()]

    cluster_to_speaker_id, speaker_metadata = {}, {}
    for rank, cluster_id in enumerate(sorted_clusters):
        spk_num = rank + 1
        spk_id = f"speaker_{spk_num:03d}"
        cluster_to_speaker_id[cluster_id] = spk_id

        cluster_pitches = [pitches[i] for i, lab in enumerate(labels) if lab == cluster_id and pitches[i] > 60]
        median_f0 = float(np.median(cluster_pitches)) if cluster_pitches else 0.0

        speaker_metadata[spk_id] = {
            "id": spk_id,
            "name": f"Speaker {spk_num:03d}",
            "gender": "Unknown",
            "segmentsCount": cluster_counts[cluster_id],
            "pitchF0": round(median_f0, 1) if median_f0 else 0,
            "color": SPEAKER_COLORS[rank % len(SPEAKER_COLORS)],
        }

    final_segments = []
    for idx, seg in enumerate(raw_segments):
        spk_id = cluster_to_speaker_id[labels[idx]]
        meta = speaker_metadata[spk_id]
        final_segments.append({
            **seg,
            "speakerId": spk_id,
            "speakerName": meta["name"],
            "gender": "Unknown",
            "pitchF0": pitches[idx] if pitches[idx] > 0 else meta["pitchF0"],
        })
    return final_segments, speaker_metadata


# ---------------------------------------------------------------------------
# Markdown helpers
# ---------------------------------------------------------------------------

def sanitize_filename(name, default="transcript"):
    cleaned = "".join(c if c.isalnum() or c in "._- " else "_" for c in name).strip(" .")
    return (cleaned or default)[:150]


def build_markdown(source_name, final_segments, speaker_metadata):
    lines = [
        f"# Transcript: {source_name}",
        f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total Segments: {len(final_segments)}",
        "",
        "## Speakers",
        "",
    ]
    for spk in speaker_metadata.values():
        lines.append(f"- {spk['name']}: {spk['segmentsCount']} segment(s)")
    lines += ["", "## Transcript", ""]
    for seg in final_segments:
        start_s = int(seg["start"])
        time_str = f"{start_s // 60:02d}:{start_s % 60:02d}"
        label = seg.get("speakerName", seg.get("speakerId", "Speaker"))
        lines.append(f"**[{time_str}] {label}:** {seg['text']}")
        lines.append("")
    return "\n".join(lines).strip()


def write_transcript_file(filename, content):
    """Writes into TRANSCRIPTS_DIR only. Returns (absolute_path, file_name)."""
    base = sanitize_filename(filename)
    if not base.lower().endswith(".md"):
        base += ".md"
    os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)
    root = os.path.realpath(TRANSCRIPTS_DIR)
    target = os.path.realpath(os.path.join(root, base))
    if os.path.dirname(target) != root:
        raise ValueError("Invalid file name")
    with open(target, "w", encoding="utf-8") as f:
        f.write(content)
    return target, base


# ---------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------

class TranscribeHandler(BaseHTTPRequestHandler):
    server_version = "EasyTScribe"

    def log_message(self, fmt, *args):
        sys.stderr.write("[http] " + (fmt % args) + "\n")

    # -- request validation -------------------------------------------------

    def _host_ok(self):
        """Blocks DNS rebinding: only accept requests addressed to a loopback name."""
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]").lower()
        return host in ("127.0.0.1", "localhost", "::1")

    def _origin_ok(self):
        origin = self.headers.get("Origin")
        return origin is None or origin.rstrip("/") in ALLOWED_ORIGINS

    def _guard(self):
        """Returns True if the request may proceed; otherwise a response has been sent."""
        if not self._host_ok():
            self._json(403, {"error": "Forbidden host"})
            return False
        if not self._origin_ok():
            self._json(403, {"error": "Origin not allowed"})
            return False
        return True

    # -- response helpers ---------------------------------------------------

    def _cors_headers(self):
        origin = self.headers.get("Origin")
        if origin and origin.rstrip("/") in ALLOWED_ORIGINS:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Requested-With")

    def _json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        try:
            self.send_response(status)          # status line first, then headers
            self._cors_headers()
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (ConnectionError, TimeoutError):
            # The browser closed the connection (tab closed/reloaded, request cancelled, sleep...).
            # Nothing can be sent any more; the transcript was already saved to ./transcripts/.
            print("[!] Client disconnected before the response could be delivered "
                  "(the result is still saved in ./transcripts/).")
            self.close_connection = True

    def _content_length(self):
        try:
            return int(self.headers.get("Content-Length", 0))
        except ValueError:
            return -1

    # -- routes -------------------------------------------------------------

    def do_OPTIONS(self):
        if not self._guard():
            return
        self.send_response(204)
        self._cors_headers()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        if not self._guard():
            return
        path = urllib.parse.urlparse(self.path).path
        if path in ("/health", "/", "/api/health"):
            self._json(200, {
                "status": "online",
                "engine": "faster-whisper",
                "default_model": default_model_ref,
                "whisper_models": list_local_whisper_models(),
                "diarizer_type": diarizer_type,
                "speechbrain_ecapa": SNAPSHOT_SPEECHBRAIN_ECAPA is not None,
                "offline": True,
            })
        else:
            self._json(404, {"error": "Not found"})

    def do_POST(self):
        if not self._guard():
            return
        path = urllib.parse.urlparse(self.path).path
        if path in ("/api/save-markdown", "/save-markdown"):
            return self._handle_save_markdown()
        if path in ("/transcribe", "/api/transcribe"):
            return self._handle_transcribe()
        self._json(404, {"error": "Not found"})

    def _handle_save_markdown(self):
        length = self._content_length()
        if length <= 0 or length > 50 * 1024 * 1024:
            return self._json(400, {"error": "Invalid or too large body"})
        try:
            data = json.loads(self.rfile.read(length).decode("utf-8"))
            filename = data.get("filename") or f"transcript_{int(time.time())}.md"
            target, base = write_transcript_file(str(filename), str(data.get("content", "")))
            self._json(200, {
                "success": True,
                "filename": base,
                "relativePath": f"transcripts/{base}",
                "fullPath": target,
            })
        except Exception as e:
            self._json(500, {"success": False, "error": str(e)})

    def _handle_transcribe(self):
        tmp_path = None
        try:
            params = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)

            def p(name, default=None):
                return params.get(name, [default])[0]

            try:
                min_silence_ms = int(p("min_silence_ms", 250))
                beam_size = int(p("beam_size", 5))
                max_speakers = max(1, min(12, int(p("max_speakers", DEFAULT_MAX_SPEAKERS))))
                exact_speakers = p("exact_speakers", "0") in ("1", "true", "True")
            except ValueError:
                return self._json(400, {"error": "Invalid numeric query parameter"})

            model_name = p("model") or default_model_ref
            device = p("device", "auto")
            language = p("language")
            if language in (None, "", "auto"):
                language = None
            source_name = p("filename", f"audio_{int(time.time())}.wav")

            length = self._content_length()
            if length <= 0:
                return self._json(400, {"error": "Empty or invalid body"})
            if length > MAX_UPLOAD_BYTES:
                return self._json(413, {"error": f"Upload exceeds limit of {MAX_UPLOAD_BYTES // (1024 * 1024)} MB"})

            # Resolve the model before reading the upload so a bad name fails fast.
            resolve_whisper_model(model_name)

            # Stream the upload to disk (no second copy in RAM).
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp_path = tmp.name
                remaining = length
                while remaining > 0:
                    chunk = self.rfile.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    tmp.write(chunk)
                    remaining -= len(chunk)
            if remaining > 0:
                return self._json(400, {"error": "Incomplete upload"})

            print(f"[*] Processing '{source_name}' ({length} bytes, model={model_name}, "
                  f"lang={language or 'auto'}, min_silence={min_silence_ms}ms)...")

            with TRANSCRIBE_LOCK:
                whisper_model = get_whisper_model(model_name, device)
                segments_generator, info = whisper_model.transcribe(
                    tmp_path,
                    beam_size=beam_size,
                    best_of=5,
                    patience=1.0,
                    temperature=[0.0, 0.2, 0.4],
                    condition_on_previous_text=False,
                    word_timestamps=True,
                    language=language,
                    vad_filter=True,
                    vad_parameters=dict(
                        min_silence_duration_ms=min_silence_ms,
                        speech_pad_ms=200,
                        threshold=0.35,
                    ),
                )

                raw_results = []
                for seg in segments_generator:
                    words = [
                        {
                            "word": w.word,
                            "start": round(w.start, 2),
                            "end": round(w.end, 2),
                            "probability": round(w.probability, 2),
                        }
                        for w in (seg.words or [])
                    ]
                    avg_logprob = getattr(seg, "avg_logprob", None)
                    confidence = None
                    if avg_logprob is not None:
                        confidence = round(max(0.0, min(1.0, math.exp(avg_logprob))), 3)
                    raw_results.append({
                        "start": round(seg.start, 2),
                        "end": round(seg.end, 2),
                        "text": seg.text.strip(),
                        "confidence": confidence,
                        "words": words,
                    })

                print(f"[+] Whisper produced {len(raw_results)} segments.")

                samples, sr = load_wav(tmp_path)
                final_segments, speaker_metadata = perform_speaker_diarization(
                    samples, sr, raw_results, max_speakers, exact_speakers
                )

            print(f"[+] Diarization complete: {len(speaker_metadata)} speaker(s) ({diarizer_type}).")
            speakers_summary = "\n".join(
                f"- {s['name']}: {s['segmentsCount']} segment(s)" for s in speaker_metadata.values()
            )

            # Server-side copy of the transcript (the UI later overwrites it with the richer export).
            auto_saved_rel_path = None
            try:
                base_source = sanitize_filename(os.path.splitext(os.path.basename(source_name))[0])
                _, saved = write_transcript_file(
                    f"transcript_{base_source}.md",
                    build_markdown(source_name, final_segments, speaker_metadata),
                )
                auto_saved_rel_path = f"transcripts/{saved}"
                print(f"[+] Saved {auto_saved_rel_path}")
            except Exception as save_err:
                print(f"[!] Auto-save failed: {save_err}")

            self._json(200, {
                "status": "success",
                "model": model_name,
                "language": info.language,
                "language_probability": round(info.language_probability, 3),
                "duration": round(info.duration, 2),
                "total_segments": len(final_segments),
                "segments": final_segments,
                "speakers": speaker_metadata,
                "speakers_summary": speakers_summary,
                "diarizer_type": diarizer_type,
                "auto_saved_path": auto_saved_rel_path,
            })

        except ModelNotFound as err:
            self._json(404, {"status": "error", "error": str(err)})
        except Exception as err:
            traceback.print_exc()
            try:
                self._json(500, {"status": "error", "error": str(err)})
            except Exception:
                pass
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass


def make_server(port):
    return ThreadingHTTPServer((HOST, port), TranscribeHandler)


def main():
    global default_model_ref
    default_model_ref = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    port = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_PORT

    print("=" * 65)
    print("  EASY TSCRIBE - LOCAL SPEECH & DIARIZATION SERVER")
    print("=" * 65)
    print(f"[*] Hugging Face cache: {HF_HUB}")
    print(f"[*] Local Whisper models: {', '.join(list_local_whisper_models()) or 'none found'}")

    try:
        get_whisper_model(default_model_ref)
    except (ModelNotFound, RuntimeError) as e:
        print(f"[!] Default model not preloaded: {e}")
    init_speaker_diarization()

    httpd = make_server(port)
    print(f"[*] Listening on http://{HOST}:{port} (allowed origins: {', '.join(sorted(ALLOWED_ORIGINS))})")
    print("=" * 65)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Shutting down.")
        httpd.server_close()


if __name__ == "__main__":
    main()
