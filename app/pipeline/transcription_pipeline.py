import yaml
import logging
import os
import torch
from app.core.audio_processor import AudioProcessor
from app.core.transcriber import Transcriber
from app.core.semantic_engine import SemanticEngine
from app.core.voice_fingerprint import VoiceFingerprint
from app.core.exporter import MarkdownExporter

class TranscriptionPipeline:
    def __init__(self, config_path="app/config/config.yaml"):
        self.logger = logging.getLogger(__name__)
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.audio_processor = AudioProcessor()
        self.transcriber = Transcriber(
            model_size=self.config['asr']['model_size'],
            device=self.config['asr']['device'],
            compute_type=self.config['asr'].get('compute_type', 'int8')
        )
        self.semantic_engine = None # Lazy load
        self.fingerprinter = None # Lazy load
        self.exporter = MarkdownExporter()

    def initialize(self, progress_callback=None):
        """
        Initializes models. If model is missing, it will be downloaded.
        """
        # 1. Initialize Transcriber (Whisper)
        if progress_callback:
            progress_callback(0.05, "Checking Whisper model...")
        
        self.transcriber.download_model(
            self.config['asr']['model_size'], 
            progress_callback=progress_callback
        )
        
        if progress_callback:
            progress_callback(0.40, "Loading Whisper model...")
        self.transcriber.initialize_model()

        # 2. Voice Fingerprinting / Diarization (if enabled)
        if self.config.get('voice_fingerprint', {}).get('enabled', True):
            if progress_callback:
                progress_callback(0.70, "Initializing Diarization engine...")
            self.fingerprinter = VoiceFingerprint(
                db_path=self.config['voice_fingerprint'].get('storage_path', 'app/voice_db'),
                threshold=self.config['voice_fingerprint']['threshold'],
                device=self.config['asr']['device'],
                max_speakers=self.config['diarization'].get('max_speakers', 10)
            )

        # 3. Semantic Clustering (if enabled)
        if self.config.get('semantic_clustering', {}).get('enabled', True):
            if progress_callback:
                progress_callback(0.90, "Loading Semantic engine...")
            
            self.semantic_engine = SemanticEngine(
                model_name=self.config['semantic_clustering']['model']
            )
        
        if progress_callback:
            progress_callback(1.0, "Ready")

    def change_model(self, model_size, progress_callback=None):
        """Changes the whisper model size and re-initializes."""
        if model_size == self.transcriber.model_size:
            return
            
        self.config['asr']['model_size'] = model_size
        self.transcriber.update_config(model_size=model_size)
        
        # Ensure download and re-init
        self.initialize(progress_callback=progress_callback)

    def process_file(self, input_path):
        """
        Processes a single file with advanced diarization and high-quality transcription.
        """
        import gc
        
        try:
            # 1. Extract & Normalize Audio
            full_wav_path = self.audio_processor.extract_audio(input_path)
            
            # Check duration
            duration = self.audio_processor.get_duration(full_wav_path)
            chunk_duration = self.config['paths'].get('chunk_duration', 1800)
            
            all_segments = []
            
            # Context prompt
            initial_prompt = f"Transcription of {os.path.basename(input_path)}. High quality voice transcription."

            # High-quality settings
            transcribe_kwargs = {
                "beam_size": 12,
                "best_of": 7,
                "vad_filter": True,
                "initial_prompt": initial_prompt,
                "vad_parameters": dict(min_silence_duration_ms=700, speech_pad_ms=400)
            }

            if duration > chunk_duration:
                self.logger.info(f"Long file detected ({duration:.1f}s), processing in chunks")
                chunks = self.audio_processor.split_audio(full_wav_path, chunk_duration)
                for i, chunk_path in enumerate(chunks):
                    chunk_start = i * chunk_duration
                    chunk_segments = self.transcriber.transcribe(chunk_path, **transcribe_kwargs)
                    for seg in chunk_segments:
                        seg["start"] += chunk_start
                        seg["end"] += chunk_start
                        if "words" in seg:
                            for w in seg["words"]:
                                w["start"] += chunk_start
                                w["end"] += chunk_start
                        all_segments.append(seg)
                    self.audio_processor.cleanup(chunk_path)
                    # Clear memory after each chunk
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
            else:
                all_segments = self.transcriber.transcribe(full_wav_path, **transcribe_kwargs)

            # 4. Adaptive Chunking & AHC Diarization
            if self.fingerprinter and all_segments:
                self.logger.info("Starting Utterance-Chunked AHC Diarization")
                
                # Extract all words
                words = []
                for seg in all_segments:
                    if "words" in seg:
                        words.extend(seg["words"])
                    else:
                        # Fallback if no word timestamps
                        words.append({"word": seg["text"], "start": seg["start"], "end": seg["end"]})

                # Build adaptive chunks
                chunks = []
                current_words = []
                chunk_start = None
                
                for w in words:
                    if not current_words:
                        current_words.append(w)
                        chunk_start = w["start"]
                    else:
                        prev_w = current_words[-1]
                        pause = w["start"] - prev_w["end"]
                        duration = w["end"] - chunk_start
                        
                        if pause > 0.4 or duration > 3.0:
                            if duration >= 0.25: # minimum duration for embedding
                                chunks.append({"words": current_words, "start": chunk_start, "end": current_words[-1]["end"]})
                                current_words = [w]
                                chunk_start = w["start"]
                            else:
                                current_words.append(w)
                        else:
                            current_words.append(w)
                            
                if current_words:
                    duration = current_words[-1]["end"] - chunk_start
                    if duration >= 0.25 or not chunks:
                        chunks.append({"words": current_words, "start": chunk_start, "end": current_words[-1]["end"]})
                    else:
                        chunks[-1]["words"].extend(current_words)
                        chunks[-1]["end"] = current_words[-1]["end"]

                # Extract embeddings
                import numpy as np
                from scipy.spatial.distance import cosine
                from sklearn.cluster import AgglomerativeClustering
                
                valid_chunks = []
                invalid_chunks = []
                chunk_embeddings = []
                for chunk in chunks:
                    emb = self.fingerprinter.extract_embedding(full_wav_path, chunk["start"], chunk["end"])
                    if emb is not None:
                        valid_chunks.append(chunk)
                        chunk_embeddings.append(emb)
                    else:
                        chunk["speaker_id"] = "unknown"
                        chunk["uncertain"] = True
                        invalid_chunks.append(chunk)
                        
                if valid_chunks and len(valid_chunks) > 0:
                    X = np.array(chunk_embeddings)
                    dist_threshold = 1.0 - self.fingerprinter.threshold
                    
                    if len(X) > 1:
                        clustering = AgglomerativeClustering(
                            n_clusters=None,
                            metric="cosine",
                            linkage="average",
                            distance_threshold=dist_threshold
                        )
                        labels = clustering.fit_predict(X)
                    else:
                        labels = np.array([0])
                        
                    # Map clusters to Voice DB
                    cluster_speaker_ids = {}
                    cluster_centroids = {}
                    
                    for label in np.unique(labels):
                        cluster_indices = np.where(labels == label)[0]
                        cluster_embs = X[cluster_indices]
                        centroid = np.mean(cluster_embs, axis=0)
                        
                        # Normalize centroid
                        norm = np.linalg.norm(centroid)
                        if norm > 0:
                            centroid = centroid / norm
                            
                        speaker_id = self.fingerprinter.identify_speaker(centroid)
                        cluster_centroids[label] = centroid
                        cluster_speaker_ids[label] = speaker_id
                        
                    # Assign speakers to chunks and assess confidence
                    for i, chunk in enumerate(valid_chunks):
                        label = labels[i]
                        emb = X[i]
                        centroid = cluster_centroids[label]
                        dist = cosine(emb, centroid)
                        
                        chunk["speaker_id"] = cluster_speaker_ids[label]
                        chunk["uncertain"] = dist > (dist_threshold * 0.8)

                    # Reconstruct segments from all chunks (valid + invalid)
                    all_chunks = valid_chunks + invalid_chunks
                    all_chunks.sort(key=lambda x: x["start"])
                    
                    final_segments = []
                    current_seg = None
                    
                    for chunk in all_chunks:
                        # Clean words
                        text = "".join([w["word"] if w["word"].startswith((" ", ".", ",", "?", "!")) else " " + w["word"] for w in chunk["words"]]).strip()
                        if not text:
                            continue
                            
                        if chunk["uncertain"]:
                            text = f"[(?)] {text}"
                            
                        if current_seg is None:
                            current_seg = {
                                "start": chunk["start"],
                                "end": chunk["end"],
                                "text": text,
                                "speaker_id": chunk["speaker_id"],
                                "uncertain": chunk["uncertain"]
                            }
                        else:
                            pause = chunk["start"] - current_seg["end"]
                            # Merge if same speaker, no uncertainty, and small pause
                            if chunk["speaker_id"] == current_seg["speaker_id"] and pause < 1.0 and not chunk["uncertain"] and not current_seg.get("uncertain", False):
                                current_seg["end"] = chunk["end"]
                                current_seg["text"] += " " + text
                            else:
                                final_segments.append(current_seg)
                                current_seg = {
                                    "start": chunk["start"],
                                    "end": chunk["end"],
                                    "text": text,
                                    "speaker_id": chunk["speaker_id"],
                                    "uncertain": chunk["uncertain"]
                                }
                    if current_seg:
                        final_segments.append(current_seg)
                        
                    all_segments = final_segments
                else:
                    self.logger.warning("No valid chunks extracted for diarization, fallback to unknown.")
                    # Fallback to invalid chunks if everything failed
                    all_chunks = invalid_chunks
                    all_chunks.sort(key=lambda x: x["start"])
                    final_segments = []
                    current_seg = None
                    for chunk in all_chunks:
                        text = "".join([w["word"] if w["word"].startswith((" ", ".", ",", "?", "!")) else " " + w["word"] for w in chunk["words"]]).strip()
                        if not text: continue
                        text = f"[(?)] {text}"
                        if current_seg is None:
                            current_seg = {"start": chunk["start"], "end": chunk["end"], "text": text, "speaker_id": "unknown", "uncertain": True}
                        else:
                            pause = chunk["start"] - current_seg["end"]
                            if pause < 1.0:
                                current_seg["end"] = chunk["end"]
                                current_seg["text"] += " " + text
                            else:
                                final_segments.append(current_seg)
                                current_seg = {"start": chunk["start"], "end": chunk["end"], "text": text, "speaker_id": "unknown", "uncertain": True}
                    if current_seg: final_segments.append(current_seg)
                    all_segments = final_segments

            # 5. Advanced Speaker Smoothing
            if len(all_segments) >= 3:
                for _ in range(2):
                    for i in range(1, len(all_segments) - 1):
                        prev = all_segments[i-1]["speaker_id"]
                        curr = all_segments[i]["speaker_id"]
                        next = all_segments[i+1]["speaker_id"]
                        if curr == "unknown":
                            all_segments[i]["speaker_id"] = prev if prev != "unknown" else next
                        if curr != prev and prev == next and (all_segments[i]["end"] - all_segments[i]["start"] < 1.5):
                            all_segments[i]["speaker_id"] = prev

            # 6. Semantic Clustering
            if self.config.get('semantic_clustering', {}).get('enabled', True) and self.semantic_engine and all_segments:
                all_segments = self.semantic_engine.cluster_segments(
                    all_segments, 
                    max_clusters=self.config['semantic_clustering']['max_clusters']
                )
            
            # 7. Export
            export_path = self.exporter.export(all_segments, input_path)
            
            # 8. Cleanup
            self.audio_processor.cleanup(full_wav_path)
            
            # Final memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            return all_segments, export_path
        except Exception as e:
            self.logger.error(f"Error processing file {input_path}: {e}")
            # Try to cleanup
            try: self.audio_processor.cleanup(full_wav_path)
            except: pass
            raise
