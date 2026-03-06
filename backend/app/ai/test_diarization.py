from app.ai.diarization import diarize

segments = diarize("uploads/download.wav", num_speakers=2)

for seg in segments:
    print(seg)