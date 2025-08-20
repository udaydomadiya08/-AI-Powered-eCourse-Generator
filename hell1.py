import whisperx

# Path to audio
audio_path = "/Users/uday/Downloads/Part 2 - WBS Network Diagram CPM Slack Times.mp3"

# Load ASR model (tiny.en is fast and works well on CPU)
model = whisperx.load_model("tiny.en", device="cpu", compute_type="int8")

# Transcribe the audio
result = model.transcribe(audio_path)

# Try to get full transcript
print("TRANSCRIPT:")
if "text" in result:
    print(result["text"])
else:
    # Fallback: Construct from segments
    full_text = " ".join([seg["text"] for seg in result.get("segments", [])])
    print(full_text)

# Load alignment model (for word-level timestamps)
model_a, metadata = whisperx.load_align_model(language_code=result["language"], device="cpu")

# Align transcript to audio
aligned = whisperx.align(result["segments"], model_a, metadata, audio_path, device="cpu")

# Print word-level timestamps
print("\nWORD TIMESTAMPS:")
for word in aligned["word_segments"]:
    print(f"{word['word']} ({word['start']:.2f}s - {word['end']:.2f}s)")
