from app.ai.transcription import transcribe
from app.config import settings
import os

# Path to the uploaded file
audio_path = os.path.join(settings.UPLOAD_DIR, "download.mp3")

print("Testing transcription...")
print("Audio file:", audio_path)

segments = transcribe(audio_path)

print("\nTranscription segments:\n")

for seg in segments:
    start = seg["start"]
    end = seg["end"]
    text = seg["text"]

    print(f"[{start:.1f}s - {end:.1f}s] {text}")

print(f"\nTotal segments: {len(segments)}")