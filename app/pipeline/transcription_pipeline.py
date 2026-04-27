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
                        all_segments.append(seg)
                    self.audio_processor.cleanup(chunk_path)
                    # Clear memory after each chunk
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
            else:
                all_segments = self.transcriber.transcribe(full_wav_path, **transcribe_kwargs)

            # 4. Advanced Window-based Diarization
            if self.fingerprinter and all_segments:
                self.logger.info(f"Starting diarization for {len(all_segments)} segments")
                final_segments = []
                for seg in all_segments:
                    seg_duration = seg["end"] - seg["start"]
                    
                    # If segment is short, do single identification
                    if seg_duration < 4.0:
                        embedding = self.fingerprinter.extract_embedding(full_wav_path, seg["start"], seg["end"])
                        seg["speaker_id"] = self.fingerprinter.identify_speaker(embedding)
                        final_segments.append(seg)
                    else:
                        # Multi-window analysis
                        windows = []
                        window_size = 2.0
                        step = 1.0
                        
                        curr = seg["start"]
                        while curr + window_size <= seg["end"]:
                            emb = self.fingerprinter.extract_embedding(full_wav_path, curr, curr + window_size)
                            windows.append(self.fingerprinter.identify_speaker(emb))
                            curr += step
                        
                        if not windows:
                            embedding = self.fingerprinter.extract_embedding(full_wav_path, seg["start"], seg["end"])
                            seg["speaker_id"] = self.fingerprinter.identify_speaker(embedding)
                            final_segments.append(seg)
                        else:
                            from collections import Counter
                            counts = Counter(windows)
                            most_common, freq = counts.most_common(1)[0]
                            seg["speaker_id"] = most_common
                            if freq / len(windows) <= 0.7:
                                seg["text"] = f"[(?)] {seg['text']}"
                            final_segments.append(seg)
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
