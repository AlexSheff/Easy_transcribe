import os
import torch
import numpy as np
from pathlib import Path
import json
import logging
from scipy.spatial.distance import cosine
from speechbrain.inference.speaker import EncoderClassifier
from scipy.io import wavfile

class VoiceFingerprint:
    def __init__(self, db_path="app/voice_db", threshold=0.70, device="auto", max_speakers=10):
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.threshold = threshold
        self.max_speakers = max_speakers
        self.logger = logging.getLogger(__name__)
        
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.logger.info(f"Initializing SpeechBrain on {self.device}")
        try:
            self.encoder = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                run_opts={"device": self.device}
            )
        except Exception as e:
            self.logger.error(f"Failed to load SpeechBrain: {e}")
            raise

        self.speakers = self._load_db()

    def _load_db(self):
        """Loads existing speaker embeddings from disk."""
        speakers = {}
        if not self.db_path.exists():
            return speakers
            
        for speaker_dir in self.db_path.iterdir():
            if speaker_dir.is_dir():
                embedding_path = speaker_dir / "embedding.npy"
                metadata_path = speaker_dir / "metadata.json"
                if embedding_path.exists() and metadata_path.exists():
                    try:
                        embedding = np.load(embedding_path)
                        with open(metadata_path, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                        speakers[speaker_dir.name] = {
                            "embedding": embedding,
                            "metadata": metadata
                        }
                    except Exception as e:
                        self.logger.warning(f"Failed to load speaker {speaker_dir.name}: {e}")
        return speakers

    def extract_embedding(self, audio_path, start_sec, end_sec):
        """Extracts speaker embedding for a specific segment."""
        try:
            # Using scipy.io.wavfile to avoid torchaudio/torchcodec backend issues
            # Since our audio is already 16kHz WAV, this is very efficient.
            if not os.path.exists(str(audio_path)):
                self.logger.error(f"Audio file NOT FOUND: {audio_path}")
                return None
                
            fs, data = wavfile.read(str(audio_path))
            
            # Extract segment
            start_sample = int(start_sec * fs)
            end_sample = int(end_sec * fs)
            
            if start_sample >= len(data):
                return None
                
            waveform = data[start_sample:end_sample]
            
            # Ensure segment is long enough
            if len(waveform) < int(0.2 * fs): # Lowered to 0.2s for maximum coverage
                return None

            # Convert to float32 and normalize
            waveform = waveform.astype(np.float32)
            max_val = np.max(np.abs(waveform))
            if max_val > 0:
                waveform = waveform / max_val
            
            # Convert to torch tensor [batch, time]
            waveform_t = torch.from_numpy(waveform).unsqueeze(0).to(self.device).float()
            
            # Extract embedding - wrapping in try/except to catch SpeechBrain specific errors
            try:
                with torch.no_grad():
                    embedding = self.encoder.encode_batch(waveform_t)
                    embedding = embedding.squeeze().cpu().numpy()
                return embedding
            except Exception as e:
                self.logger.error(f"SpeechBrain core failed: {e}")
                return None
                
        except Exception as e:
            self.logger.warning(f"Embedding process failed at {start_sec:.1f}s: {e}")
            return None

    def identify_speaker(self, embedding: np.ndarray) -> str:
        """
        Compares new embedding against DB.
        Returns speaker_id.
        """
        if embedding is None:
            return "unknown"
            
        if np.all(embedding == 0) or np.any(np.isnan(embedding)):
            return "unknown"

        # Adaptive threshold: slightly lower for new speakers to avoid 'unknown' explosion
        current_threshold = self.threshold # 0.65 recommended now

        best_match = None
        best_score = -1

        for speaker_id, data in self.speakers.items():
            # Similarity = 1 - distance
            score = 1 - cosine(embedding, data["embedding"])
            if score > best_score:
                best_score = score
                best_match = speaker_id

        if best_score >= current_threshold:
            self.logger.debug(f"Matched {best_match} (score: {best_score:.2f})")
            return best_match
        
        # If we have reached the limit, force use best_match even if below threshold
        if len(self.speakers) >= self.max_speakers:
            if best_match:
                self.logger.debug(f"Speaker limit ({self.max_speakers}) reached. Force mapping to {best_match} (score: {best_score:.2f})")
                return best_match
            # If for some reason we have NO speakers yet (unlikely with reached limit), 
            # we must create at least one.
        
        # Create new identity
        new_id = f"speaker_{len(self.speakers) + 1:03d}"
        self._save_speaker(new_id, embedding)
        self.logger.info(f"New speaker detected: {new_id} (top score was {best_score:.2f})")
        return new_id

    def _save_speaker(self, speaker_id, embedding):
        speaker_dir = self.db_path / speaker_id
        speaker_dir.mkdir(exist_ok=True)
        np.save(speaker_dir / "embedding.npy", embedding)
        
        metadata = {
            "name": f"Unknown {speaker_id}",
            "confidence": 1.0,
            "samples": 1
        }
        metadata_path = speaker_dir / "metadata.json"
        with open(metadata_path, "w", encoding='utf-8') as f:
            json.dump(metadata, f, indent=4)
        
        self.speakers[speaker_id] = {"embedding": embedding, "metadata": metadata}
