import os
import shutil
import torch
import numpy as np
from pathlib import Path
import json
import logging
from scipy.spatial.distance import cosine
from scipy.signal import welch
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

        # Gender detection is done offline via pitch (F0) analysis — no model download needed.
        # Male fundamental frequency: ~80–165 Hz, Female: ~165–265 Hz.
        self.gender_pitch_threshold = 165.0  # Hz boundary between male/female
        self.logger.info("Gender detection: offline pitch-based F0 analysis (no download required)")

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

    def extract_gender(self, audio_path, start_sec, end_sec):
        """
        Estimates speaker gender via offline pitch (F0) analysis using Welch PSD.
        Works without any model download.
        Male voices: dominant F0 roughly 80–165 Hz.
        Female voices: dominant F0 roughly 165–265 Hz.
        Returns: 'male' | 'female' | 'unknown'
        """
        try:
            fs, data = wavfile.read(str(audio_path))
            start_sample = int(start_sec * fs)
            end_sample = int(end_sec * fs)

            if start_sample >= len(data):
                return "unknown"

            waveform = data[start_sample:end_sample]

            # Need at least 0.5s for reliable pitch estimation
            if len(waveform) < int(0.5 * fs):
                return "unknown"

            waveform = waveform.astype(np.float32)
            max_val = np.max(np.abs(waveform))
            if max_val == 0:
                return "unknown"
            waveform = waveform / max_val

            # Compute Power Spectral Density using Welch method
            freqs, psd = welch(waveform, fs=fs, nperseg=min(512, len(waveform)))

            # Focus on the voiced speech range: 60–300 Hz
            voiced_mask = (freqs >= 60) & (freqs <= 300)
            if not np.any(voiced_mask):
                return "unknown"

            voiced_psd = psd[voiced_mask]
            voiced_freqs = freqs[voiced_mask]

            # Find the dominant frequency peak in the voiced range
            peak_idx = np.argmax(voiced_psd)
            dominant_f0 = voiced_freqs[peak_idx]

            self.logger.debug(f"Pitch F0: {dominant_f0:.1f} Hz (threshold {self.gender_pitch_threshold} Hz)")

            if dominant_f0 < self.gender_pitch_threshold:
                return "male"
            else:
                return "female"

        except Exception as e:
            self.logger.warning(f"Gender extraction failed: {e}")

        return "unknown"

    def update_speaker_gender(self, speaker_id, gender):
        """Updates the gender metadata for a known speaker."""
        if speaker_id not in self.speakers or gender == "unknown":
            return
            
        # Capitalize for display (Male, Female)
        display_gender = gender.capitalize()
        
        # Format the name
        speaker_idx = speaker_id.split('_')[-1]
        new_name = f"{display_gender} Speaker {speaker_idx}"
        
        self.speakers[speaker_id]["metadata"]["gender"] = display_gender
        self.speakers[speaker_id]["metadata"]["name"] = new_name
        
        # Save to disk
        metadata_path = self.db_path / speaker_id / "metadata.json"
        with open(metadata_path, "w", encoding='utf-8') as f:
            json.dump(self.speakers[speaker_id]["metadata"], f, indent=4)
        
        self.logger.info(f"Updated {speaker_id} gender to {display_gender}")

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
        
        # Default name
        speaker_idx = speaker_id.split('_')[-1]
        name = f"Unknown Speaker {speaker_idx}"
        
        metadata = {
            "name": name,
            "confidence": 1.0,
            "samples": 1,
            "gender": "Unknown"
        }
        metadata_path = speaker_dir / "metadata.json"
        with open(metadata_path, "w", encoding='utf-8') as f:
            json.dump(metadata, f, indent=4)
        
        self.speakers[speaker_id] = {"embedding": embedding, "metadata": metadata}

    def clear(self):
        """Clears all speaker embeddings from memory and disk."""
        # Remove speaker directories from disk
        for speaker_dir in self.db_path.iterdir():
            if speaker_dir.is_dir() and speaker_dir.name.startswith("speaker_"):
                try:
                    shutil.rmtree(speaker_dir)
                except Exception as e:
                    self.logger.warning(f"Failed to remove {speaker_dir}: {e}")
        # Clear in-memory state
        self.speakers = {}
        self.logger.info("Voice DB cleared for new session")
