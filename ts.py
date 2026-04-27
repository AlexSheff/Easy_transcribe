import os
import ffmpeg
from faster_whisper import WhisperModel
from tqdm import tqdm

INPUT_DIR = "input_mp4"
OUTPUT_DIR = "output_txt"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# модель
model = WhisperModel("small", device="cpu", compute_type="int8")

def extract_audio(video_path):
    audio_path = video_path.replace(".mp4", ".mp3")

    if not os.path.exists(audio_path):
        (
            ffmpeg
            .input(video_path)
            .output(audio_path, ac=1, ar=16000)
            .run(overwrite_output=True, quiet=True)
        )
    return audio_path

def transcribe(audio_path):
    segments, _ = model.transcribe(audio_path, beam_size=5)
    text = ""
    for seg in segments:
        text += f"[{seg.start:.2f}-{seg.end:.2f}] {seg.text}\n"
    return text

def process_folder():
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".mp4")]

    for file in tqdm(files):
        video_path = os.path.join(INPUT_DIR, file)
        audio_path = extract_audio(video_path)

        result = transcribe(audio_path)

        out_file = os.path.join(OUTPUT_DIR, file.replace(".mp4", ".txt"))
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(result)

if __name__ == "__main__":
    process_folder()