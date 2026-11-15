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
import subprocess
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
OUTPUTS_DIR = os.environ.get("EASY_TSCRIBE_OUTPUTS_DIR") or os.path.join(BASE_DIR, "outputs")

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
    order = {"tiny": 0, "base": 1, "small": 2, "medium": 3, "large-v3": 4, "large-v3-turbo": 5}
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
_whisper_cache = {}                     # (model_path, device, compute_type) -> WhisperModel
_model_load_lock = threading.Lock()
TRANSCRIBE_LOCK = threading.Lock()      # one transcription at a time (GPU/RAM safety)

_ACTIVE_JOBS = {}                       # job_id -> {"progress": int, "stage": str, "cancelled": bool}
_JOBS_LOCK = threading.Lock()

diarizer_model = None
diarizer_type = "none"


def set_job_progress(job_id, progress, stage):
    if not job_id:
        return
    with _JOBS_LOCK:
        job = _ACTIVE_JOBS.setdefault(job_id, {"progress": 0, "stage": "", "cancelled": False})
        job["progress"] = int(max(0, min(100, progress)))
        job["stage"] = str(stage)


def is_job_cancelled(job_id):
    if not job_id:
        return False
    with _JOBS_LOCK:
        job = _ACTIVE_JOBS.get(job_id)
        return bool(job and job.get("cancelled"))


def cancel_job(job_id):
    if not job_id:
        return False
    with _JOBS_LOCK:
        job = _ACTIVE_JOBS.get(job_id)
        if job is None:
            return False
        job["cancelled"] = True
        job["stage"] = "Cancelling..."
        return True


def pick_device(requested="auto"):
    if requested in ("cuda", "cpu"):
        return requested
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def get_whisper_model(model_ref, device="auto", compute_type="auto"):
    """Loads (and caches) a faster-whisper model."""
    model_path = resolve_whisper_model(model_ref)
    device = pick_device(device)
    if compute_type not in ("float16", "int8_float16", "int8", "float32"):
        compute_type = "float16" if device == "cuda" else "int8"
    key = (model_path, device, compute_type)
    with _model_load_lock:
        if key in _whisper_cache:
            return _whisper_cache[key]

        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise RuntimeError("Package 'faster-whisper' is not installed. Run: pip install -r requirements.txt")

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
        usable = (len(samples) // nchannels) * nchannels
        samples = samples[:usable].reshape(-1, nchannels).mean(axis=1)
    return samples, framerate


def write_pcm16_wav(samples, sr, dst_path):
    """Writes float32 mono samples (-1..1) as a standard 16-bit PCM WAV."""
    import numpy as np
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    with wave.open(dst_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes(pcm.tobytes())


def normalize_audio_samples(samples):
    """Brings quiet recordings up to ~-18 dBFS RMS without clipping peaks."""
    import numpy as np
    if len(samples) == 0:
        return samples
    rms = float(np.sqrt(np.mean(samples * samples)))
    peak = float(np.max(np.abs(samples)))
    if rms < 1e-5 or peak < 1e-5:
        return samples
    target_rms = 0.125  # ~ -18 dBFS
    gain = min(target_rms / rms, 0.95 / peak, 12.0)
    if gain <= 1.05:
        return samples
    return np.clip(samples * gain, -0.98, 0.98).astype(np.float32)


def _resample_linear(samples, src_sr, dst_sr=16000):
    import numpy as np
    if src_sr == dst_sr or len(samples) < 2:
        return samples
    new_len = max(1, int(round(len(samples) * float(dst_sr) / float(src_sr))))
    x_old = np.linspace(0.0, 1.0, len(samples), endpoint=False)
    x_new = np.linspace(0.0, 1.0, new_len, endpoint=False)
    return np.interp(x_new, x_old, samples).astype(np.float32)


def prepare_16k_wav(src_path, normalize=False):
    """
    Ensures `src_path` is converted into a 16 kHz mono 16-bit PCM WAV.
    Supports native PCM WAV directly (with optional resampling/normalization) and
    falls back to PyAV (`av`) or `ffmpeg` for MP4/MKV/MP3/M4A/WEBM/OGG/FLAC.
    Returns (wav_path, temp_path_to_cleanup_or_None).
    """
    # 1. Fast path: already a PCM WAV
    try:
        samples, sr = load_wav(src_path)
        if sr == 16000 and not normalize:
            return src_path, None
        if sr != 16000:
            samples = _resample_linear(samples, sr, 16000)
        if normalize:
            samples = normalize_audio_samples(samples)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as out:
            dst = out.name
        write_pcm16_wav(samples, 16000, dst)
        return dst, dst
    except (wave.Error, EOFError):
        pass

    # 2. Try PyAV (installed alongside faster-whisper)
    try:
        import av
        import numpy as np
        chunks = []
        with av.open(src_path) as container:
            audio_streams = [s for s in container.streams if s.type == "audio"]
            if not audio_streams:
                raise ValueError("Media file contains no audio stream")
            resampler = av.AudioResampler(format="flt", layout="mono", rate=16000)
            for frame in container.decode(audio_streams[0]):
                for resampled in resampler.resample(frame):
                    arr = resampled.to_ndarray().reshape(-1).astype(np.float32)
                    if len(arr) > 0:
                        chunks.append(arr)
            for resampled in resampler.resample(None):
                arr = resampled.to_ndarray().reshape(-1).astype(np.float32)
                if len(arr) > 0:
                    chunks.append(arr)
        if not chunks:
            raise ValueError("Decoded audio stream is empty")
        samples = np.concatenate(chunks)
        if normalize:
            samples = normalize_audio_samples(samples)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as out:
            dst = out.name
        write_pcm16_wav(samples, 16000, dst)
        return dst, dst
    except ImportError:
        pass
    except Exception as av_err:
        print(f"[!] PyAV decode failed ({av_err}), trying ffmpeg...")

    # 3. Fallback to ffmpeg CLI if installed on PATH
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as out:
            dst = out.name
        cmd = [ffmpeg_bin, "-y", "-i", src_path, "-vn", "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", dst]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode == 0 and os.path.isfile(dst) and os.path.getsize(dst) > 44:
            if normalize:
                samples, sr = load_wav(dst)
                write_pcm16_wav(normalize_audio_samples(samples), sr, dst)
            return dst, dst
        if os.path.exists(dst):
            try:
                os.remove(dst)
            except Exception:
                pass

    raise ValueError(
        "Could not decode media file. Send a WAV file, or install PyAV ('pip install av') / ffmpeg on PATH."
    )


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


_DEFAULT_EMBED_SEGMENT = embed_segment


def embed_segments_ecapa_batch(chunks, sr, batch_size=16):
    """
    Computes L2-normalized ECAPA-TDNN embeddings in mini-batches for speed.
    Long chunks (>4.5s) are averaged across overlapping 2.5s windows for cleaner speaker vectors.
    """
    import numpy as np
    import torch

    results = [None] * len(chunks)
    pitches = [0.0] * len(chunks)
    work_items = []  # (orig_idx, window_samples)

    win_len = int(sr * 2.5)
    hop_len = int(sr * 1.25)
    long_thresh = int(sr * 4.5)

    for idx, chunk in enumerate(chunks):
        if chunk is None or len(chunk) < int(sr * MIN_EMBED_SEC):
            continue
        _, f0 = compute_acoustic_features(chunk, sr)
        pitches[idx] = f0
        if len(chunk) > long_thresh:
            for start in range(0, len(chunk) - int(sr * 1.0), hop_len):
                sub = chunk[start:min(len(chunk), start + win_len)]
                if len(sub) >= int(sr * MIN_EMBED_SEC):
                    work_items.append((idx, sub))
        else:
            work_items.append((idx, chunk))

    if not work_items:
        return results, pitches

    # Sort by length so padding inside each batch is minimal
    work_items.sort(key=lambda item: len(item[1]))
    accum = {}
    counts = {}

    for b_start in range(0, len(work_items), batch_size):
        batch = work_items[b_start:b_start + batch_size]
        max_len = max(len(w[1]) for w in batch)
        batch_tensor = torch.zeros((len(batch), max_len), dtype=torch.float32)
        wav_lens = torch.zeros(len(batch), dtype=torch.float32)
        for j, (_, arr) in enumerate(batch):
            t = torch.from_numpy(arr.astype(np.float32, copy=False))
            batch_tensor[j, :len(arr)] = t
            wav_lens[j] = float(len(arr)) / float(max_len)

        with torch.no_grad():
            try:
                embs = diarizer_model.encode_batch(batch_tensor, wav_lens)
            except TypeError:
                embs = diarizer_model.encode_batch(batch_tensor)
            embs_np = embs.squeeze(1).cpu().numpy() if embs.ndim == 3 else embs.cpu().numpy()
            if embs_np.ndim == 1:
                embs_np = embs_np[None, :]

        for j, (orig_idx, _) in enumerate(batch):
            vec = embs_np[j]
            norm = np.linalg.norm(vec)
            if norm > 1e-6:
                vec = vec / norm
            if orig_idx in accum:
                accum[orig_idx] = accum[orig_idx] + vec
                counts[orig_idx] += 1
            else:
                accum[orig_idx] = vec
                counts[orig_idx] = 1

    for orig_idx, vec_sum in accum.items():
        avg = vec_sum / max(1, counts[orig_idx])
        norm = np.linalg.norm(avg)
        if norm > 1e-6:
            avg = avg / norm
        results[orig_idx] = avg

    return results, pitches


def _cosine_sims(matrix, centroids):
    import numpy as np
    m = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)
    c = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-9)
    return m @ c.T


def cluster_speakers(embeddings, durations, max_speakers, exact=False, min_silhouette=None):
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

    min_sil = MIN_SILHOUETTE if min_silhouette is None else float(min_silhouette)

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
        if best_score < min_sil:
            print(f"[*] Speakers are not clearly separable (best silhouette {best_score:.3f} < {min_sil}); using 1 speaker.")
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


def perform_speaker_diarization(samples, sr, raw_segments, max_speakers=None, exact=False, min_silhouette=None):
    """
    Groups transcript segments into speakers ("Speaker 001" = the one with most segments).
    Segments too short to embed inherit the label of the nearest embedded segment by timestamp.
    Returns (segments, speaker_metadata).
    """
    import numpy as np

    if not raw_segments:
        return [], {}
    max_speakers = DEFAULT_MAX_SPEAKERS if max_speakers is None else max_speakers

    print(f"[*] Computing speaker embeddings for {len(raw_segments)} segments "
          f"(engine: {diarizer_type}, max speakers: {max_speakers}{', exact' if exact else ''})...")

    chunks = [slice_samples(samples, sr, seg["start"], seg["end"]) for seg in raw_segments]
    embeddings, durations, pitches, embedded_idx = [], [], [], []

    if (
        diarizer_type == "speechbrain-ecapa"
        and diarizer_model is not None
        and embed_segment is _DEFAULT_EMBED_SEGMENT
    ):
        try:
            batch_embs, pitches = embed_segments_ecapa_batch(chunks, sr)
            for idx, emb in enumerate(batch_embs):
                if emb is not None:
                    embeddings.append(emb)
                    durations.append(len(chunks[idx]) / sr)
                    embedded_idx.append(idx)
        except Exception as e:
            print(f"[!] Batched ECAPA failed ({e}), falling back to per-segment embedding...")
            embeddings, durations, pitches, embedded_idx = [], [], [], []

    if not pitches:
        for idx, chunk in enumerate(chunks):
            emb, f0 = embed_segment(chunk, sr)
            pitches.append(f0)
            if emb is not None:
                embeddings.append(emb)
                durations.append(len(chunk) / sr)
                embedded_idx.append(idx)

    labels = [0] * len(raw_segments)
    if embeddings:
        emb_labels = cluster_speakers(embeddings, durations, max_speakers, exact, min_silhouette=min_silhouette)
        assigned = {i: lab for i, lab in zip(embedded_idx, emb_labels)}
        midpoints = [(float(s["start"]) + float(s["end"])) * 0.5 for s in raw_segments]
        for idx in range(len(raw_segments)):
            if idx in assigned:
                labels[idx] = assigned[idx]
            else:
                # Too short to embed: inherit from the temporally nearest embedded segment.
                nearest = min(embedded_idx, key=lambda j: abs(midpoints[j] - midpoints[idx]))
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
        start_s = max(0, int(seg["start"]))
        hours, rem = divmod(start_s, 3600)
        minutes, seconds = divmod(rem, 60)
        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        label = seg.get("speakerName", seg.get("speakerId", "Speaker"))
        lines.append(f"**[{time_str}] {label}:** {seg['text']}")
        lines.append("")
    return "\n".join(lines).strip()


def _output_search_dirs():
    """Returns unique (dir_path, label) pairs for outputs and transcripts."""
    dirs = []
    seen = set()
    for path, label in ((OUTPUTS_DIR, "outputs"), (TRANSCRIPTS_DIR, "transcripts")):
        if not path:
            continue
        real = os.path.realpath(path)
        if real not in seen:
            seen.add(real)
            dirs.append((path, label))
    return dirs


def find_existing_transcript(source_name):
    """
    Checks if a transcript for `source_name` already exists and is non-empty in
    OUTPUTS_DIR or TRANSCRIPTS_DIR. Returns metadata dict or None.
    """
    if not source_name:
        return None
    raw_base = os.path.splitext(os.path.basename(str(source_name)))[0]
    clean_base = sanitize_filename(raw_base, default="")
    if not clean_base:
        return None

    candidates = [
        f"transcript_{clean_base}.md",
        f"{clean_base}.md",
        f"transcript_{clean_base}.txt",
        f"{clean_base}.txt",
        f"{clean_base}.srt",
        f"transcript_{clean_base}.json",
        f"{clean_base}.json",
    ]
    candidate_lower_map = {c.lower(): i for i, c in enumerate(candidates)}

    for directory, label in _output_search_dirs():
        if not os.path.isdir(directory):
            continue
        root = os.path.realpath(directory)
        # Fast exact check first
        for cand in candidates:
            target = os.path.realpath(os.path.join(root, cand))
            if os.path.dirname(target) == root and os.path.isfile(target) and os.path.getsize(target) > 0:
                return {
                    "exists": True,
                    "filename": cand,
                    "relativePath": f"{label}/{cand}",
                    "fullPath": target,
                }
        # Case-insensitive fallback scan
        try:
            best_match = None
            best_rank = len(candidates) + 1
            with os.scandir(root) as it:
                for entry in it:
                    if not entry.is_file():
                        continue
                    rank = candidate_lower_map.get(entry.name.lower())
                    if rank is not None and rank < best_rank:
                        target = os.path.realpath(entry.path)
                        if os.path.dirname(target) == root and os.path.getsize(target) > 0:
                            best_rank = rank
                            best_match = (entry.name, target)
            if best_match is not None:
                entry_name, target = best_match
                return {
                    "exists": True,
                    "filename": entry_name,
                    "relativePath": f"{label}/{entry_name}",
                    "fullPath": target,
                }
        except OSError:
            continue
    return None


def _write_file_to_dir(directory, base, content):
    os.makedirs(directory, exist_ok=True)
    root = os.path.realpath(directory)
    target = os.path.realpath(os.path.join(root, base))
    if os.path.dirname(target) != root:
        raise ValueError("Invalid file name")
    with open(target, "w", encoding="utf-8") as f:
        f.write(content)
    return target


def write_transcript_file(filename, content):
    """Writes into TRANSCRIPTS_DIR (and mirrors to OUTPUTS_DIR when active). Returns (absolute_path, file_name)."""
    base = sanitize_filename(filename)
    if not base.lower().endswith(".md"):
        base += ".md"
    target = _write_file_to_dir(TRANSCRIPTS_DIR, base, content)

    default_transcripts = os.path.realpath(os.path.join(BASE_DIR, "transcripts"))
    default_outputs = os.path.realpath(os.path.join(BASE_DIR, "outputs"))
    if OUTPUTS_DIR and (
        os.path.realpath(TRANSCRIPTS_DIR) == default_transcripts
        or os.path.realpath(OUTPUTS_DIR) != default_outputs
    ):
        if os.path.realpath(OUTPUTS_DIR) != os.path.realpath(TRANSCRIPTS_DIR):
            _write_file_to_dir(OUTPUTS_DIR, base, content)

    return target, base


def list_saved_transcripts():
    """Lists transcript files in OUTPUTS_DIR and TRANSCRIPTS_DIR, newest first."""
    items = []
    seen_rel = set()
    allowed_exts = (".md", ".txt", ".srt", ".json")
    for directory, label in _output_search_dirs():
        if not os.path.isdir(directory):
            continue
        root = os.path.realpath(directory)
        try:
            with os.scandir(root) as it:
                for entry in it:
                    if not entry.is_file() or not entry.name.lower().endswith(allowed_exts):
                        continue
                    target = os.path.realpath(entry.path)
                    if os.path.dirname(target) != root:
                        continue
                    stat = entry.stat()
                    if stat.st_size <= 0:
                        continue
                    rel = f"{label}/{entry.name}"
                    if rel in seen_rel:
                        continue
                    seen_rel.add(rel)
                    items.append({
                        "filename": entry.name,
                        "relativePath": rel,
                        "size": stat.st_size,
                        "modifiedAt": datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "_mtime": stat.st_mtime,
                    })
        except OSError:
            continue
    items.sort(key=lambda x: x["_mtime"], reverse=True)
    for item in items:
        item.pop("_mtime", None)
    return items


def read_saved_transcript(relative_path):
    """Safely reads a saved transcript referenced as 'outputs/...' or 'transcripts/...'."""
    rel = str(relative_path or "").replace("\\", "/").lstrip("/")
    if "/" not in rel:
        raise ValueError("Invalid transcript path")
    label, filename = rel.split("/", 1)
    if label not in ("outputs", "transcripts") or not filename or "/" in filename or filename in (".", ".."):
        raise ValueError("Invalid transcript path")

    dir_map = {lbl: path for path, lbl in _output_search_dirs()}
    directory = dir_map.get(label)
    if not directory or not os.path.isdir(directory):
        raise FileNotFoundError("Transcript directory not found")

    root = os.path.realpath(directory)
    target = os.path.realpath(os.path.join(root, filename))
    if os.path.dirname(target) != root or not os.path.isfile(target):
        raise FileNotFoundError("Transcript file not found")
    if os.path.getsize(target) > 20 * 1024 * 1024:
        raise ValueError("Transcript file too large to preview")

    with open(target, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    return {
        "filename": os.path.basename(target),
        "relativePath": f"{label}/{os.path.basename(target)}",
        "content": content,
    }


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
        raw_host = (self.headers.get("Host") or "").strip()
        if not raw_host:
            return False
        try:
            hostname = (urllib.parse.urlsplit(f"//{raw_host}").hostname or "").lower()
        except ValueError:
            return False
        return hostname in ("127.0.0.1", "localhost", "::1")

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
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
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
        elif path in ("/api/check-transcript", "/check-transcript"):
            params = urllib.parse.parse_qs(parsed.query)
            filename = (params.get("filename") or [""])[0]
            if not filename.strip():
                return self._json(400, {"error": "Missing 'filename' query parameter"})
            match = find_existing_transcript(filename)
            if match:
                self._json(200, {
                    "exists": True,
                    "filename": match["filename"],
                    "relativePath": match["relativePath"],
                })
            else:
                self._json(200, {"exists": False})
        elif path in ("/api/transcripts", "/transcripts"):
            self._json(200, {"items": list_saved_transcripts()})
        elif path in ("/api/transcript", "/transcript"):
            params = urllib.parse.parse_qs(parsed.query)
            rel_path = (params.get("path") or [""])[0]
            try:
                self._json(200, read_saved_transcript(rel_path))
            except FileNotFoundError as err:
                self._json(404, {"error": str(err)})
            except ValueError as err:
                self._json(400, {"error": str(err)})
        elif path in ("/api/progress", "/progress"):
            params = urllib.parse.parse_qs(parsed.query)
            job_id = (params.get("job_id") or [""])[0]
            with _JOBS_LOCK:
                job = _ACTIVE_JOBS.get(job_id)
                if not job:
                    self._json(200, {"active": False, "progress": 0, "stage": "", "cancelled": False})
                else:
                    self._json(200, {
                        "active": True,
                        "progress": job["progress"],
                        "stage": job["stage"],
                        "cancelled": job["cancelled"],
                    })
        else:
            self._json(404, {"error": "Not found"})

    def do_POST(self):
        if not self._guard():
            return
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        if path in ("/api/check-transcripts", "/check-transcripts"):
            return self._handle_check_transcripts()
        if path in ("/api/save-markdown", "/save-markdown"):
            return self._handle_save_markdown()
        if path in ("/api/cancel", "/cancel"):
            params = urllib.parse.parse_qs(parsed.query)
            job_id = (params.get("job_id") or [""])[0]
            return self._json(200, {"cancelled": cancel_job(job_id)})
        if path in ("/transcribe", "/api/transcribe"):
            return self._handle_transcribe()
        self._json(404, {"error": "Not found"})

    def _handle_check_transcripts(self):
        length = self._content_length()
        if length <= 0 or length > 2 * 1024 * 1024:
            self.close_connection = True
            return self._json(400, {"error": "Invalid or too large body"})
        try:
            data = json.loads(self.rfile.read(length).decode("utf-8"))
            filenames = data.get("filenames")
            if not isinstance(filenames, list) or len(filenames) > 5000:
                return self._json(400, {"error": "Expected 'filenames' array (max 5000 items)"})
            results = {}
            for name in filenames:
                name_str = str(name)
                match = find_existing_transcript(name_str)
                if match:
                    results[name_str] = {
                        "exists": True,
                        "filename": match["filename"],
                        "relativePath": match["relativePath"],
                    }
                else:
                    results[name_str] = {"exists": False}
            self._json(200, {"results": results})
        except Exception as e:
            self._json(400, {"error": f"Invalid JSON payload: {e}"})

    def _handle_save_markdown(self):
        length = self._content_length()
        if length <= 0 or length > 50 * 1024 * 1024:
            self.close_connection = True
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
        prepared_cleanup_path = None
        job_id = None
        try:
            params = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)

            def p(name, default=None):
                return params.get(name, [default])[0]

            try:
                min_silence_ms = max(50, min(5000, int(p("min_silence_ms", 250))))
                beam_size = max(1, min(10, int(p("beam_size", 5))))
                max_speakers = max(1, min(12, int(p("max_speakers", DEFAULT_MAX_SPEAKERS))))
                vad_threshold = max(0.05, min(0.95, float(p("vad_threshold", 0.35))))
                min_silhouette = max(0.01, min(0.50, float(p("min_silhouette", MIN_SILHOUETTE))))
                batch_size = max(1, min(32, int(p("batch_size", 8))))
                exact_speakers = p("exact_speakers", "0") in ("1", "true", "True")
                skip_existing = p("skip_existing", "0") in ("1", "true", "True")
                use_batched = p("use_batched", "0") in ("1", "true", "True")
                normalize_audio = p("normalize_audio", "0") in ("1", "true", "True")
                word_timestamps = p("word_timestamps", "1") not in ("0", "false", "False")
            except ValueError:
                self.close_connection = True
                return self._json(400, {"error": "Invalid numeric query parameter"})

            model_name = p("model") or default_model_ref
            device = p("device", "auto")
            compute_type = p("compute_type", "auto")
            language = p("language")
            if language in (None, "", "auto"):
                language = None
            source_name = p("filename", f"audio_{int(time.time())}.wav")
            initial_prompt = (p("initial_prompt") or "").strip()[:1000]
            hotwords = (p("hotwords") or "").strip()[:500]
            job_id = (p("job_id") or "").strip()[:80] or None

            if skip_existing:
                existing = find_existing_transcript(source_name)
                if existing:
                    rem = self._content_length()
                    if 0 < rem <= 4 * 1024 * 1024:
                        while rem > 0:
                            chunk = self.rfile.read(min(1024 * 1024, rem))
                            if not chunk:
                                break
                            rem -= len(chunk)
                    else:
                        self.close_connection = True
                    return self._json(200, {
                        "status": "skipped",
                        "skipped": True,
                        "filename": existing["filename"],
                        "existing_path": existing["relativePath"],
                    })

            length = self._content_length()
            if length <= 0:
                self.close_connection = True
                return self._json(400, {"error": "Empty or invalid body"})
            if length > MAX_UPLOAD_BYTES:
                self.close_connection = True
                return self._json(413, {"error": f"Upload exceeds limit of {MAX_UPLOAD_BYTES // (1024 * 1024)} MB"})

            # Resolve the model before reading the upload so a bad name fails fast.
            try:
                resolve_whisper_model(model_name)
            except ModelNotFound:
                self.close_connection = True
                raise

            if job_id:
                set_job_progress(job_id, 10, f"Receiving {source_name}...")

            ext = os.path.splitext(os.path.basename(source_name))[1]
            if not ext or not all(c.isalnum() or c == "." for c in ext):
                ext = ".wav"

            # Stream the upload to disk (no second copy in RAM).
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp_path = tmp.name
                remaining = length
                while remaining > 0:
                    chunk = self.rfile.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    tmp.write(chunk)
                    remaining -= len(chunk)
            if remaining > 0:
                self.close_connection = True
                return self._json(400, {"error": "Incomplete upload"})

            if is_job_cancelled(job_id):
                return self._json(499, {"status": "cancelled", "error": "Cancelled by user"})

            if job_id:
                set_job_progress(job_id, 22, "Decoding & normalizing 16kHz audio...")

            try:
                wav_path, prepared_cleanup_path = prepare_16k_wav(tmp_path, normalize=normalize_audio)
            except (ValueError, wave.Error) as dec_err:
                return self._json(400, {"status": "error", "error": str(dec_err)})

            print(f"[*] Processing '{source_name}' ({length} bytes, model={model_name}, "
                  f"lang={language or 'auto'}, min_silence={min_silence_ms}ms, vad={vad_threshold})...")

            with TRANSCRIBE_LOCK:
                if is_job_cancelled(job_id):
                    return self._json(499, {"status": "cancelled", "error": "Cancelled by user"})

                if job_id:
                    set_job_progress(job_id, 30, f"Loading Whisper ({model_name})...")
                whisper_model = get_whisper_model(model_name, device, compute_type)

                transcribe_kwargs = {
                    "beam_size": beam_size,
                    "best_of": 5,
                    "patience": 1.0,
                    "temperature": [0.0, 0.2, 0.4],
                    "condition_on_previous_text": False,
                    "word_timestamps": word_timestamps,
                    "language": language,
                    "vad_filter": True,
                    "vad_parameters": dict(
                        min_silence_duration_ms=min_silence_ms,
                        speech_pad_ms=200,
                        threshold=vad_threshold,
                    ),
                }
                if initial_prompt:
                    transcribe_kwargs["initial_prompt"] = initial_prompt
                if hotwords:
                    transcribe_kwargs["hotwords"] = hotwords
                if word_timestamps:
                    transcribe_kwargs["hallucination_silence_threshold"] = 2.0

                segments_generator = None
                info = None
                if use_batched and hasattr(whisper_model, "model"):
                    try:
                        from faster_whisper import BatchedInferencePipeline
                        batched_pipe = BatchedInferencePipeline(model=whisper_model)
                        batched_kwargs = dict(transcribe_kwargs)
                        batched_kwargs["batch_size"] = batch_size
                        segments_generator, info = batched_pipe.transcribe(wav_path, **batched_kwargs)
                    except Exception as batch_err:
                        print(f"[!] BatchedInferencePipeline fallback to standard transcribe: {batch_err}")
                        segments_generator = None

                if segments_generator is None:
                    segments_generator, info = whisper_model.transcribe(wav_path, **transcribe_kwargs)

                total_dur = max(0.1, float(getattr(info, "duration", 0.0) or 0.1))
                if job_id:
                    set_job_progress(job_id, 35, "Transcribing speech...")

                raw_results = []
                for seg in segments_generator:
                    if is_job_cancelled(job_id):
                        return self._json(499, {"status": "cancelled", "error": "Cancelled by user"})
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
                    if job_id:
                        pct = 35 + int(47 * min(1.0, float(seg.end) / total_dur))
                        set_job_progress(job_id, pct, f"Transcribing {int(seg.end)}s / {int(total_dur)}s...")

                print(f"[+] Whisper produced {len(raw_results)} segments.")

                if is_job_cancelled(job_id):
                    return self._json(499, {"status": "cancelled", "error": "Cancelled by user"})

                if job_id:
                    set_job_progress(job_id, 85, f"Speaker diarization ({len(raw_results)} segments)...")

                try:
                    samples, sr = load_wav(wav_path)
                except (wave.Error, ValueError) as wav_err:
                    return self._json(400, {"status": "error", "error": f"Invalid WAV audio: {wav_err}"})
                final_segments, speaker_metadata = perform_speaker_diarization(
                    samples, sr, raw_results, max_speakers, exact_speakers, min_silhouette=min_silhouette
                )

            if job_id:
                set_job_progress(job_id, 96, "Saving transcript...")

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
            if prepared_cleanup_path and os.path.exists(prepared_cleanup_path):
                try:
                    os.remove(prepared_cleanup_path)
                except Exception:
                    pass
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
            if job_id:
                with _JOBS_LOCK:
                    _ACTIVE_JOBS.pop(job_id, None)


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
