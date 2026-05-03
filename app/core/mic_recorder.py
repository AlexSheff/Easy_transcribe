import queue
import time
import collections
import os
import sounddevice as sd
import soundfile as sf
import webrtcvad
import numpy as np
from PySide6.QtCore import QThread, Signal

class MicRecorder(QThread):
    chunk_ready = Signal(str)
    progress = Signal(str)
    error = Signal(str)
    
    def __init__(self, output_dir, sample_rate=16000, frame_duration_ms=30, padding_ms=300, silence_duration_ms=600):
        super().__init__()
        self.output_dir = output_dir
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.padding_ms = padding_ms
        self.silence_duration_ms = silence_duration_ms
        
        self.frame_size = int(self.sample_rate * (self.frame_duration_ms / 1000.0))
        self.chunk_count = 0
        self.is_recording = False
        
        self.vad = webrtcvad.Vad(3) # Aggressiveness 3 (highest)
        self.audio_queue = queue.Queue()
        
    def audio_callback(self, indata, frames, time_info, status):
        if status:
            pass # Handle status if needed (e.g. overflow)
        self.audio_queue.put(bytes(indata))

    def run(self):
        self.is_recording = True
        
        # webrtcvad needs 16-bit PCM. sounddevice 'int16' format gives us this directly.
        try:
            stream = sd.RawInputStream(
                samplerate=self.sample_rate, 
                blocksize=self.frame_size,
                device=None, # Default device
                channels=1, 
                dtype='int16',
                callback=self.audio_callback
            )
        except Exception as e:
            self.error.emit(f"Microphone initialization failed: {str(e)}")
            return

        num_padding_frames = int(self.padding_ms / self.frame_duration_ms)
        num_silence_frames = int(self.silence_duration_ms / self.frame_duration_ms)
        
        ring_buffer = collections.deque(maxlen=num_padding_frames)
        triggered = False
        voiced_frames = []
        silence_counter = 0

        self.progress.emit("Recording started...")

        with stream:
            while self.is_recording:
                try:
                    frame = self.audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # frame should be self.frame_size * 2 bytes long (16-bit = 2 bytes)
                if len(frame) < self.frame_size * 2:
                    continue

                is_speech = self.vad.is_speech(frame, self.sample_rate)

                if not triggered:
                    ring_buffer.append(frame)
                    if is_speech:
                        triggered = True
                        voiced_frames.extend(ring_buffer)
                        ring_buffer.clear()
                else:
                    voiced_frames.append(frame)
                    if is_speech:
                        silence_counter = 0
                    else:
                        silence_counter += 1
                        
                    if silence_counter > num_silence_frames:
                        # Phrase complete, save it
                        self.save_chunk(b''.join(voiced_frames))
                        
                        # Reset for next phrase
                        triggered = False
                        voiced_frames = []
                        silence_counter = 0

            # On stop, save any remaining voiced frames
            if voiced_frames:
                self.save_chunk(b''.join(voiced_frames))
                
        self.progress.emit("Recording stopped.")

    def save_chunk(self, audio_bytes):
        if len(audio_bytes) < self.sample_rate * 2 * 0.5:
            # Ignore chunks shorter than 0.5 seconds to prevent junk
            return
            
        self.chunk_count += 1
        filename = f"chunk_{self.chunk_count:04d}.wav"
        filepath = os.path.join(self.output_dir, filename)
        
        # Convert bytes to numpy array
        audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
        
        sf.write(filepath, audio_data, self.sample_rate)
        self.chunk_ready.emit(filepath)

    def stop(self):
        self.is_recording = False
