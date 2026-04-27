import ffmpeg
import os
import logging
from pathlib import Path

class AudioProcessor:
    def __init__(self, output_dir="app/temp"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def get_duration(self, file_path):
        """Returns duration of audio file in seconds."""
        try:
            probe = ffmpeg.probe(str(file_path))
            return float(probe['format']['duration'])
        except Exception as e:
            self.logger.error(f"Error probing duration: {e}")
            return 0

    def split_audio(self, input_path, chunk_duration):
        """
        Splits audio file into smaller chunks.
        """
        input_file = Path(input_path)
        chunks = []
        duration = self.get_duration(input_path)
        
        if duration <= chunk_duration:
            return [str(input_path)]

        self.logger.info(f"Splitting {input_path} into {chunk_duration}s chunks")
        
        num_chunks = int(duration // chunk_duration) + (1 if duration % chunk_duration > 0 else 0)
        
        for i in range(num_chunks):
            start_time = i * chunk_duration
            chunk_file = self.output_dir / f"{input_file.stem}_part{i:03d}.wav"
            
            try:
                # Add a small 1s overlap to avoid cutting words at boundaries
                stream = ffmpeg.input(str(input_path), ss=start_time, t=chunk_duration + 1)
                stream = ffmpeg.output(stream, str(chunk_file), acodec='pcm_s16le', ac=1, ar='16k', loglevel='quiet')
                ffmpeg.run(stream, overwrite_output=True)
                chunks.append(str(chunk_file))
            except ffmpeg.Error as e:
                self.logger.error(f"Error creating chunk {i}: {e}")
        
        return chunks

    def extract_audio(self, input_path):
        """
        Extracts audio from video file and converts to normalized 16kHz mono WAV.
        """
        input_file = Path(input_path)
        output_file = self.output_dir / f"{input_file.stem}.wav"

        self.logger.info(f"Extracting & Normalizing audio: {input_path}")
        
        try:
            # We use absolute paths to avoid issues with different drives
            abs_input = str(Path(input_path).absolute())
            abs_output = str(output_file.absolute())

            stream = ffmpeg.input(abs_input)
            stream = ffmpeg.output(
                stream, 
                abs_output, 
                acodec='pcm_s16le', 
                ac=1, 
                ar='16k',
                af='loudnorm',
                loglevel='quiet'
            )
            ffmpeg.run(stream, overwrite_output=True)
            return abs_output
        except ffmpeg.Error as e:
            err_msg = e.stderr.decode() if e.stderr else str(e)
            self.logger.error(f"FFmpeg error: {err_msg}")
            raise RuntimeError(f"FFmpeg failed: {err_msg}")
        except Exception as e:
            self.logger.error(f"Generic extraction error: {e}")
            raise

    @staticmethod
    def cleanup(file_path):
        """Removes temporary audio files."""
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Cleanup error: {e}")
