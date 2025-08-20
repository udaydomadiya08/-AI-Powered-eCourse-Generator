import whisperx

# Load your audio file (supports .mp3, .wav, .m4a, etc.)
audio_path = "/Users/uday/Downloads/Part 2 - WBS Network Diagram CPM Slack Times.mp3"

# Load model (medium is a good balance between speed and accuracy)
model = whisperx.load_model("medium", device="cpu")

# Transcribe the audio
result = model.transcribe(audio_path)

# Print full transcript
print("TRANSCRIPT:")
print(result["text"])

# Optional: Word-level timestamps using alignment
model_a, metadata = whisperx.load_align_model(language_code=result["language"], device="cpu")
aligned = whisperx.align(result["segments"], model_a, metadata, audio_path, device="cpu")

# Print with word-level timestamps
for word in aligned["word_segments"]:
    print(f"{word['word']} ({word['start']:.2f}s - {word['end']:.2f}s)")
