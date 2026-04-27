import logging
import os
import torch
from faster_whisper import WhisperModel
from faster_whisper.utils import download_model
from typing import List, Dict, Any, Callable
import huggingface_hub
from tqdm.auto import tqdm

class Transcriber:
    def __init__(self, model_size="medium", device="auto", compute_type=None, download_root=None):
        self.logger = logging.getLogger(__name__)
        self.model_size = model_size
        
        # Auto-detect device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.logger.info(f"Setting up device: {self.device}")

        # If CPU is used, float16 is not supported by ctranslate2
        if self.device == "cpu":
            if compute_type is None or compute_type == "float16":
                self.compute_type = "int8"
            else:
                self.compute_type = compute_type
        else:
            # GPU supports float16 in most cases, but fall back to 'default' if not specified
            self.compute_type = compute_type if compute_type else "float16"

        self.model = None # Lazy initialization
        self.download_root = download_root

    def update_config(self, model_size=None, device=None, compute_type=None):
        """Updates internal config and triggers reload if needed."""
        changed = False
        if model_size and model_size != self.model_size:
            self.model_size = model_size
            changed = True
        
        if device and device != self.device:
            self.device = device
            changed = True
            
        if compute_type and compute_type != self.compute_type:
            self.compute_type = compute_type
            changed = True
            
        if changed:
            self.model = None # Force reload on next use
            self.logger.info(f"Transcriber config updated to {self.model_size} ({self.device}). Reload pending.")

    def initialize_model(self):
        """Initializes the Whisper model. Should be called after ensuring download."""
        if self.model is None:
            self.logger.info(f"Initializing Whisper model: {self.model_size} on {self.device}")
            try:
                self.model = WhisperModel(
                    self.model_size, 
                    device=self.device, 
                    compute_type=self.compute_type,
                    download_root=self.download_root
                )
            except Exception as e:
                self.logger.error(f"Failed to initialize model: {e}")
                raise

    @staticmethod
    def download_model(model_size: str, progress_callback: Callable[[float, str], None] = None):
        """
        Explicitly download the model with progress reporting.
        """
        repo_id = f"Systran/faster-whisper-{model_size}"
        
        class GUIProgress(tqdm):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def update(self, n=1):
                super().update(n)
                if progress_callback and self.total:
                    prog = self.n / self.total
                    msg = f"Downloading {model_size}: {self.desc if self.desc else 'files'}"
                    progress_callback(prog, msg)
        
        try:
            huggingface_hub.snapshot_download(
                repo_id,
                tqdm_class=GUIProgress if progress_callback else None,
                max_workers=1 
            )
        except Exception as e:
            # Fallback for small models if snapshot fails
            pass

    def transcribe(self, audio_path: str, language: str = None, **kwargs) -> List[Dict[str, Any]]:
        """
        Transcribes audio and returns segments with metadata.
        """
        if self.model is None:
            self.initialize_model()

        self.logger.info(f"Starting transcription: {audio_path}")
        
        # Default high-quality settings, can be overridden by kwargs
        params = {
            "beam_size": 10,
            "best_of": 5,
            "language": language,
            "word_timestamps": True,
            "vad_filter": True,
            "vad_parameters": dict(
                min_silence_duration_ms=800,
                speech_pad_ms=400
            )
        }
        params.update(kwargs)

        segments, info = self.model.transcribe(audio_path, **params)

        self.logger.info(f"Detected language: {info.language} ({info.language_probability:.2f})")

        results = []
        for segment in segments:
            results.append({
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "text": segment.text.strip(),
                "language": info.language
            })
            
        return results
