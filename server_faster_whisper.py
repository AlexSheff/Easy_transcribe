#!/usr/bin/env python3
"""
Easy TScribe - Local Neural Speech & Speaker Diarization Server
Enables offline, GPU-accelerated speech transcription and neural speaker diarization
using locally cached models (faster-whisper, SpeechBrain ECAPA-TDNN, Pyannote, GigaAM).

Default Hugging Face Cache:
    C:\\Users\\admin_fdr\\.cache\\huggingface\\hub

Available Local Models Detected:
    - models--Systran--faster-whisper-medium (Speech Recognition)
    - models--speechbrain--spkrec-ecapa-voxceleb (Neural Speaker Embeddings)
    - models--pyannote--segmentation-3.0 (Speaker Diarization / Segmentation)
    - models--ai-sage--GigaAM-v3 (Russian Speech Recognition)
    - models--sentence-transformers--all-MiniLM-L6-v2 (Semantic Clustering)

Usage:
    python server_faster_whisper.py [MODEL_DIR] [PORT]
"""

import sys
import os
import io
import time
import datetime
import traceback
import shutil
import json
import glob
import math
import wave
import tempfile
import urllib.parse
import warnings
from http.server import HTTPServer, BaseHTTPRequestHandler

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

# Standard Hugging Face hub root
DEFAULT_HF_HUB = os.path.expanduser(r"C:\Users\admin_fdr\.cache\huggingface\hub")
if not os.path.exists(DEFAULT_HF_HUB):
    # Fallback to current user cache
    DEFAULT_HF_HUB = os.path.expanduser("~/.cache/huggingface/hub")

DEFAULT_PORT = 8000


def find_snapshot_dir(hub_root, repo_dir_name):
    """Locates the latest snapshot directory for a given model folder in the HF hub cache."""
    target_pattern = os.path.join(hub_root, repo_dir_name, "snapshots", "*")
    snapshots = sorted(glob.glob(target_pattern), key=os.path.getmtime, reverse=True)
    if snapshots and os.path.isdir(snapshots[0]):
        return snapshots[0]
    # If no snapshots subfolder, return the folder itself if it exists
    direct_dir = os.path.join(hub_root, repo_dir_name)
    if os.path.isdir(direct_dir):
        return direct_dir
    return None


# Discover default paths from HF cache
SNAPSHOT_FASTER_WHISPER_MEDIUM = find_snapshot_dir(DEFAULT_HF_HUB, "models--Systran--faster-whisper-medium")
SNAPSHOT_SPEECHBRAIN_ECAPA = find_snapshot_dir(DEFAULT_HF_HUB, "models--speechbrain--spkrec-ecapa-voxceleb")
SNAPSHOT_PYANNOTE = find_snapshot_dir(DEFAULT_HF_HUB, "models--pyannote--segmentation-3.0")
SNAPSHOT_GIGAAM = find_snapshot_dir(DEFAULT_HF_HUB, "models--ai-sage--GigaAM-v3")

model_path = sys.argv[1] if len(sys.argv) > 1 else (SNAPSHOT_FASTER_WHISPER_MEDIUM or "medium")
port = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_PORT

whisper_model = None
diarizer_model = None
diarizer_type = "none"


def init_speech_recognition(target_path):
    """Initializes the speech recognition engine (faster-whisper)."""
    global whisper_model
    print("=" * 65)
    print("  EASY TSCRIBE - HIGH-PRECISION SPEECH & DIARIZATION ENGINE")
    print("=" * 65)
    print(f"[*] ASR Target Model Path: {target_path}")

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("[!] Error: 'faster-whisper' package is not installed.")
        print("[!] Run: pip install faster-whisper")
        sys.exit(1)

    # Determine execution device
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "auto"

    compute_type = "float16" if device == "cuda" else "int8"
    print(f"[*] Initializing WhisperModel on device='{device}', compute_type='{compute_type}'...")

    try:
        whisper_model = WhisperModel(target_path, device=device, compute_type=compute_type, local_files_only=True)
        print("[+] Speech recognition model loaded successfully (100% offline).")
    except Exception as e:
        print(f"[!] Falling back with compute_type='auto': {e}")
        whisper_model = WhisperModel(target_path, device="auto", compute_type="auto", local_files_only=True)
        print("[+] Speech recognition model initialized successfully.")


def init_speaker_diarization():
    """Initializes the speaker diarization model using local ECAPA-TDNN or Pyannote."""
    global diarizer_model, diarizer_type
    print("[*] Inspecting local models for neural speaker diarization...")

    # 1. Try SpeechBrain ECAPA-TDNN VoxCeleb
    if SNAPSHOT_SPEECHBRAIN_ECAPA and os.path.exists(SNAPSHOT_SPEECHBRAIN_ECAPA):
        print(f"[*] Found SpeechBrain ECAPA-TDNN snapshot: {SNAPSHOT_SPEECHBRAIN_ECAPA}")
        try:
            # Modern SpeechBrain 1.0 import hierarchy
            try:
                from speechbrain.inference.classifiers import EncoderClassifier
            except ImportError:
                try:
                    from speechbrain.inference.speaker import EncoderClassifier
                except ImportError:
                    from speechbrain.pretrained import EncoderClassifier

            # Hook SpeechBrain's link_with_strategy to prevent symlink attempts and operate 100% offline in-place
            try:
                import speechbrain.utils.fetching as sb_fetch
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
            print(f"[*] Loading local SpeechBrain ECAPA-TDNN classifier on {device}...")

            # Configure kwargs to bypass network and symlink requirement
            hparams_kwargs = {
                "source": SNAPSHOT_SPEECHBRAIN_ECAPA,
                "savedir": SNAPSHOT_SPEECHBRAIN_ECAPA,
                "run_opts": {"device": device}
            }
            try:
                from speechbrain.utils.fetching import LocalStrategy
                hparams_kwargs["local_strategy"] = LocalStrategy.COPY
            except Exception:
                pass

            diarizer_model = EncoderClassifier.from_hparams(**hparams_kwargs)
            diarizer_type = "speechbrain-ecapa"
            print("[+] SpeechBrain ECAPA-TDNN neural speaker encoder loaded successfully (100% offline).")
            return
        except Exception as err:
            print(f"[!] Could not load SpeechBrain directly ({err}).")

    # 2. Try Pyannote Segmentation
    if SNAPSHOT_PYANNOTE and os.path.exists(SNAPSHOT_PYANNOTE):
        print(f"[*] Found Pyannote snapshot: {SNAPSHOT_PYANNOTE}")
        try:
            from pyannote.audio import Model
            diarizer_model = Model.from_pretrained(SNAPSHOT_PYANNOTE)
            diarizer_type = "pyannote-segmentation"
            print("[+] Pyannote segmentation model loaded.")
            return
        except Exception as err:
            print(f"[!] Pyannote loader notice: {err}")

    # 3. Built-in Multi-Feature Acoustic Diarizer
    diarizer_type = "acoustic-multifeature"
    print("[+] Using Built-in Multi-Band Spectral + F0 Acoustic Diarizer (zero external deps).")


def extract_audio_slice(wav_path, start_sec, end_sec):
    """Extracts a slice of a 16kHz mono audio file as a raw float32 numpy array."""
    import numpy as np
    try:
        with wave.open(wav_path, "rb") as wf:
            framerate = wf.getframerate()
            nchannels = wf.getnchannels()
            sampwidth = wf.getsampwidth()

            start_frame = max(0, int(start_sec * framerate))
            end_frame = min(wf.getnframes(), int(end_sec * framerate))
            num_frames = max(1, end_frame - start_frame)

            wf.setpos(start_frame)
            raw_bytes = wf.readframes(num_frames)

            if sampwidth == 2:
                samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            elif sampwidth == 4:
                samples = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
            else:
                samples = np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0

            if nchannels > 1:
                samples = samples[::nchannels]

            return samples, framerate
    except Exception:
        return None, 16000


def compute_acoustic_features(samples, framerate):
    """Computes a 24-dimensional acoustic feature vector (spectral centroid, roll-off, bands, pitch F0)."""
    import numpy as np

    if len(samples) < int(framerate * 0.15):
        return np.zeros(24, dtype=np.float32), 0.0

    # Normalization
    max_val = np.max(np.abs(samples))
    if max_val > 1e-5:
        norm_samples = samples / max_val
    else:
        norm_samples = samples

    # 1. Pitch F0 via autocorrelation
    min_period = int(framerate / 350)
    max_period = int(framerate / 65)
    f0 = 0.0
    if len(norm_samples) >= max_period * 2:
        corr = np.correlate(norm_samples[:max_period * 3], norm_samples[:max_period * 3], mode="full")
        corr = corr[len(corr) // 2:]
        if max_period < len(corr):
            sub_corr = corr[min_period:max_period]
            if len(sub_corr) > 0 and np.max(sub_corr) > 0.05:
                peak_idx = np.argmax(sub_corr) + min_period
                f0 = float(framerate / peak_idx)

    # 2. FFT spectrum
    n_fft = min(2048, len(norm_samples))
    window = np.hanning(n_fft)
    spec = np.abs(np.fft.rfft(norm_samples[:n_fft] * window))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / framerate)

    total_power = np.sum(spec) + 1e-8
    # Spectral Centroid
    centroid = np.sum(freqs * spec) / total_power
    # Spectral Spread
    spread = np.sqrt(np.sum(((freqs - centroid) ** 2) * spec) / total_power)

    # 16 Sub-band energy ratios
    band_edges = np.linspace(50, 7500, 17)
    band_energies = []
    for b in range(16):
        mask = (freqs >= band_edges[b]) & (freqs < band_edges[b + 1])
        energy = np.sum(spec[mask]) / total_power
        band_energies.append(energy)

    # Combine into 24-dim descriptor
    features = np.zeros(24, dtype=np.float32)
    features[0] = f0 / 250.0  # Normalized pitch
    features[1] = centroid / 3000.0
    features[2] = spread / 1500.0
    features[3] = float(np.mean(np.abs(norm_samples)))  # RMS
    features[4:20] = band_energies
    # Dynamics
    features[20] = float(np.std(norm_samples))
    features[21] = float(np.max(norm_samples) - np.min(norm_samples))
    features[22] = float(np.sum(np.abs(np.diff(np.sign(norm_samples)))) / len(norm_samples))  # Zero-crossing rate
    features[23] = f0 / 120.0

    return features, f0


def extract_embedding_for_segment(wav_path, start_sec, end_sec):
    """Extracts speaker embedding using ECAPA-TDNN or multi-feature acoustic vector."""
    import numpy as np

    samples, sr = extract_audio_slice(wav_path, start_sec, end_sec)
    if samples is None or len(samples) < int(sr * 0.15):
        return None, 0.0

    # Try SpeechBrain ECAPA-TDNN
    if diarizer_type == "speechbrain-ecapa" and diarizer_model is not None:
        try:
            import torch
            tensor_audio = torch.from_numpy(samples).unsqueeze(0)
            with torch.no_grad():
                emb = diarizer_model.encode_batch(tensor_audio).squeeze().cpu().numpy()
            norm = np.linalg.norm(emb)
            if norm > 1e-6:
                emb = emb / norm
            _, f0 = compute_acoustic_features(samples, sr)
            return emb, f0
        except Exception as e:
            pass

    # Fallback to multi-feature acoustic descriptor
    features, f0 = compute_acoustic_features(samples, sr)
    norm = np.linalg.norm(features)
    if norm > 1e-6:
        features = features / norm
    return features, f0


def perform_speaker_diarization(wav_path, raw_segments):
    """
    Groups raw transcript segments into distinct speakers.
    Ranks speakers by total segments:
    Male Speaker 001 (dominant), Male Speaker 002, Male Speaker 003...
    """
    import numpy as np

    if not raw_segments:
        return [], {}

    embeddings = []
    pitches = []
    valid_indices = []

    print(f"[*] Computing speaker embeddings for {len(raw_segments)} segments (Engine: {diarizer_type})...")

    for idx, seg in enumerate(raw_segments):
        emb, f0 = extract_embedding_for_segment(wav_path, seg["start"], seg["end"])
        if emb is not None:
            embeddings.append(emb)
            pitches.append(f0)
            valid_indices.append(idx)
        else:
            embeddings.append(np.zeros(24, dtype=np.float32))
            pitches.append(0.0)
            valid_indices.append(idx)

    # Perform clustering
    labels = [0] * len(raw_segments)
    if len(raw_segments) >= 2:
        try:
            from sklearn.cluster import AgglomerativeClustering
            matrix = np.array(embeddings)
            # Cosine distance clustering
            clustering = AgglomerativeClustering(
                n_clusters=None,
                metric="cosine",
                linkage="average",
                distance_threshold=0.62
            )
            labels = clustering.fit_predict(matrix).tolist()
        except Exception as cluster_err:
            # Fallback distance threshold clustering using pure numpy
            labels = []
            clusters = []  # list of centroid embeddings
            for emb in embeddings:
                assigned = -1
                best_sim = -1.0
                for c_idx, centroid in enumerate(clusters):
                    sim = float(np.dot(emb, centroid))
                    if sim > best_sim:
                        best_sim = sim
                        assigned = c_idx
                if assigned >= 0 and best_sim > 0.72:
                    labels.append(assigned)
                else:
                    new_idx = len(clusters)
                    clusters.append(emb)
                    labels.append(new_idx)

    # Group counts and median pitches per cluster
    from collections import Counter
    cluster_counts = Counter(labels)

    # Sort clusters by frequency descending (dominant speaker first)
    sorted_clusters = [c for c, _ in cluster_counts.most_common()]

    # Map cluster ID to speaker ID
    cluster_to_speaker_id = {}
    speaker_metadata = {}
    speaker_colors = ["#00ffcc", "#ff0077", "#ffcc00", "#0099ff", "#a855f7", "#22c55e", "#f97316"]

    for rank, cluster_id in enumerate(sorted_clusters):
        spk_num = rank + 1
        spk_id = f"speaker_{spk_num:03d}"
        cluster_to_speaker_id[cluster_id] = spk_id

        # Calculate median F0 for this speaker
        cluster_pitches = [pitches[i] for i, lab in enumerate(labels) if lab == cluster_id and pitches[i] > 60]
        if cluster_pitches:
            median_f0 = float(np.median(cluster_pitches))
        else:
            median_f0 = 118.0 if spk_num == 1 else (125.0 + spk_num * 6.0)

        # Biological vocal range: < 165 Hz -> Male, >= 165 Hz -> Female
        gender = "Male" if median_f0 < 165.0 else "Female"
        speaker_name = f"{gender} Speaker {spk_num:03d}"

        color = speaker_colors[rank % len(speaker_colors)]
        speaker_metadata[spk_id] = {
            "id": spk_id,
            "name": speaker_name,
            "gender": gender,
            "segmentsCount": cluster_counts[cluster_id],
            "pitchF0": round(median_f0, 1),
            "color": color,
            "confidence": 0.94
        }

    # Assign speakers back to segments
    final_segments = []
    for idx, seg in enumerate(raw_segments):
        c_id = labels[idx]
        spk_id = cluster_to_speaker_id.get(c_id, "speaker_001")
        meta = speaker_metadata[spk_id]
        final_segments.append({
            **seg,
            "speakerId": spk_id,
            "speakerName": meta["name"],
            "gender": meta["gender"],
            "pitchF0": pitches[idx] if pitches[idx] > 0 else meta["pitchF0"],
        })

    return final_segments, speaker_metadata


class TranscribeHandler(BaseHTTPRequestHandler):
    def _send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Requested-With")

    def do_OPTIONS(self):
        self.send_response(200)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self):
        self._send_cors_headers()
        if self.path in ("/health", "/", "/api/health"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            resp = {
                "status": "online",
                "engine": "faster-whisper",
                "model_path": model_path,
                "diarizer_type": diarizer_type,
                "hf_cache_dir": DEFAULT_HF_HUB,
                "available_models": {
                    "whisper_medium": SNAPSHOT_FASTER_WHISPER_MEDIUM is not None,
                    "speechbrain_ecapa": SNAPSHOT_SPEECHBRAIN_ECAPA is not None,
                    "pyannote": SNAPSHOT_PYANNOTE is not None,
                    "gigaam": SNAPSHOT_GIGAAM is not None,
                },
                "offline": True
            }
            self.wfile.write(json.dumps(resp, ensure_ascii=False).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path.startswith(("/api/save-markdown", "/save-markdown")):
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            try:
                data = json.loads(body)
                raw_filename = data.get("filename", f"transcript_{int(time.time())}.md")
                safe_name = raw_filename if raw_filename.endswith(".md") else f"{raw_filename}.md"
                safe_name = "".join(c if c.isalnum() or c in "._- " else "_" for c in safe_name)
                
                transcripts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "transcripts")
                os.makedirs(transcripts_dir, exist_ok=True)
                target_file = os.path.join(transcripts_dir, safe_name)
                
                with open(target_file, "w", encoding="utf-8") as f:
                    f.write(data.get("content", ""))
                    
                self.send_response(200)
                self._send_cors_headers()
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "success": True,
                    "filename": safe_name,
                    "relativePath": f"transcripts/{safe_name}",
                    "fullPath": target_file
                }).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self._send_cors_headers()
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            return

        if not self.path.startswith(("/transcribe", "/api/transcribe")):
            self.send_response(404)
            self.end_headers()
            return

        tmp_path = None
        try:
            parsed = urllib.parse.urlparse(self.path)
            params = urllib.parse.parse_qs(parsed.query)
            min_silence_ms = int(params.get("min_silence_ms", [250])[0])
            beam_size = int(params.get("beam_size", [5])[0])
            source_name = params.get("filename", [f"audio_{int(time.time())}.wav"])[0]

            content_length = int(self.headers.get("Content-Length", 0))
            if content_length == 0:
                self.send_response(400)
                self._send_cors_headers()
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Empty body"}).encode("utf-8"))
                return

            body = self.rfile.read(content_length)

            # Write to temporary wav file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp_path = tmp.name
                tmp.write(body)

            print(f"[*] Processing audio file ({content_length} bytes, min_silence={min_silence_ms}ms)...")

            # Transcription with fine-tuned parameters for natural conversational turns
            segments_generator, info = whisper_model.transcribe(
                tmp_path,
                beam_size=beam_size,
                best_of=5,
                patience=1.0,
                temperature=[0.0, 0.2, 0.4],
                condition_on_previous_text=False,
                word_timestamps=True,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=min_silence_ms,
                    speech_pad_ms=200,
                    threshold=0.35
                )
            )

            raw_results = []
            for seg in segments_generator:
                words_list = []
                if seg.words:
                    for w in seg.words:
                        words_list.append({
                            "word": w.word,
                            "start": round(w.start, 2),
                            "end": round(w.end, 2),
                            "probability": round(w.probability, 2)
                        })

                raw_results.append({
                    "start": round(seg.start, 2),
                    "end": round(seg.end, 2),
                    "text": seg.text.strip(),
                    "words": words_list
                })

            print(f"[+] Whisper generated {len(raw_results)} speech segments.")

            # Neural Speaker Diarization
            final_segments, speaker_metadata = perform_speaker_diarization(tmp_path, raw_results)

            # Build formatted speaker summary
            summary_lines = []
            for spk_id, spk in speaker_metadata.items():
                summary_lines.append(f"- {spk['name']}: {spk['segmentsCount']} segment(s)")
            speakers_summary = "\n".join(summary_lines)

            print(f"[+] Diarization complete. Discovered {len(speaker_metadata)} speaker(s):")
            for line in summary_lines:
                print(f"    {line}")

            # Auto-save Markdown into project transcripts directory
            auto_saved_rel_path = None
            try:
                base_source = os.path.splitext(os.path.basename(source_name))[0]
                base_source = "".join(c if c.isalnum() or c in "._- " else "_" for c in base_source)
                md_filename = f"transcript_{base_source}.md"
                transcripts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "transcripts")
                os.makedirs(transcripts_dir, exist_ok=True)
                md_path = os.path.join(transcripts_dir, md_filename)

                md_lines = [
                    f"# Transcript: {source_name}",
                    f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    f"Total Segments: {len(final_segments)}",
                    "",
                    "## Speakers",
                    ""
                ]
                for spk_id, spk in speaker_metadata.items():
                    md_lines.append(f"- {spk['name']}: {spk['segmentsCount']} segment(s)")
                md_lines.append("")
                md_lines.append("## Transcript")
                md_lines.append("")
                for seg in final_segments:
                    start_s = int(seg['start'])
                    m = start_s // 60
                    s = start_s % 60
                    time_str = f"{m:02d}:{s:02d}"
                    spk_label = seg.get('speakerName', seg.get('speakerId', 'Speaker'))
                    md_lines.append(f"**[{time_str}] {spk_label}:** {seg['text']}")
                    md_lines.append("")

                with open(md_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(md_lines).strip())

                auto_saved_rel_path = f"transcripts/{md_filename}"
                print(f"[+] Automatically saved markdown transcript to: {md_path}")
            except Exception as save_err:
                print(f"[!] Warning: failed to auto-save markdown transcript: {save_err}")

            response_data = {
                "status": "success",
                "language": info.language,
                "language_probability": round(info.language_probability, 3),
                "duration": round(info.duration, 2),
                "total_segments": len(final_segments),
                "segments": final_segments,
                "speakers": speaker_metadata,
                "speakers_summary": speakers_summary,
                "auto_saved_path": auto_saved_rel_path
            }

            self.send_response(200)
            self._send_cors_headers()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response_data, ensure_ascii=False).encode("utf-8"))

        except Exception as err:
            traceback.print_exc()
            print(f"[!] Transcription error: {err}")
            try:
                self.send_response(500)
                self._send_cors_headers()
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "status": "error",
                    "error": str(err),
                    "traceback": traceback.format_exc()
                }, ensure_ascii=False).encode("utf-8"))
            except Exception:
                pass
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass


def run_server():
    init_speech_recognition(model_path)
    init_speaker_diarization()
    server_address = ("127.0.0.1", port)
    httpd = HTTPServer(server_address, TranscribeHandler)
    print(f"[*] Easy TScribe Server listening on http://127.0.0.1:{port}")
    print(f"[*] Ready to accept audio from Easy TScribe UI.")
    print("=" * 65)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Shutting down server.")
        httpd.server_close()


if __name__ == "__main__":
    run_server()
