"""
Verify edited audio by transcribing it and checking for swear words.
This is the ground truth test - if transcription shows swear words, they're still there!
"""
import whisperx
import torch
import json

device = "cpu"
compute_type = "float32"

print("="*70)
print("VERIFICATION: Transcribing edited audio to check for swear words")
print("="*70)

# Load model
print("\nLoading WhisperX model...")
model = whisperx.load_model("base", device, compute_type=compute_type)

# Transcribe edited audio
print("\nTranscribing edited_audio_wordlevel.wav...")
audio = whisperx.load_audio("edited_audio_wordlevel.wav")
result = model.transcribe(audio, batch_size=16)

print("\nAligning words...")
model_a, metadata = whisperx.load_align_model(language_code="en", device=device)
result = whisperx.align(result["segments"], model_a, metadata, audio, device)

# Load swear words
with open("word_lists.json", "r") as f:
    swear_words = json.load(f)['swear_words']

print("\n" + "="*70)
print("TRANSCRIPTION OF EDITED AUDIO")
print("="*70)

found_swears = []
total_words = 0

for segment in result["segments"]:
    print(f"\n[{segment['start']:.2f} - {segment['end']:.2f}]: {segment['text']}")

    if "words" in segment:
        for word_info in segment["words"]:
            word = word_info["word"].lower().strip()
            total_words += 1

            # Check if this is a swear word
            if any(swear in word for swear in swear_words):
                found_swears.append({
                    "word": word,
                    "time": word_info["start"],
                    "segment": segment['text']
                })
                print(f"  ⚠️  SWEAR WORD FOUND: '{word}' at {word_info['start']:.2f}s")

print("\n" + "="*70)
print("VERIFICATION RESULTS")
print("="*70)

if found_swears:
    print(f"\n❌ FAILED: Found {len(found_swears)} swear word(s) in edited audio!")
    print("\nSwear words detected:")
    for swear in found_swears:
        print(f"  - '{swear['word']}' at {swear['time']:.2f}s in: \"{swear['segment']}\"")
    print("\n⚠️  The replacements are NOT working correctly!")
else:
    print(f"\n✅ SUCCESS: No swear words found in edited audio!")
    print(f"   Transcribed {total_words} total words, all clean.")
    print("\n🎉 The replacements are working correctly!")

# Save edited transcription
with open("edited_transcript.txt", "w") as f:
    f.write("Edited Audio Transcription (Post-Processing):\n")
    f.write("="*70 + "\n\n")
    for segment in result["segments"]:
        f.write(f"[{segment['start']:.2f} - {segment['end']:.2f}]: {segment['text']}\n")

print(f"\n✓ Saved edited transcription to edited_transcript.txt")

# Compare word counts
print("\n" + "="*70)
print("COMPARISON WITH ORIGINAL")
print("="*70)

# Load original transcript
with open("transcript.txt", "r") as f:
    orig_lines = f.readlines()

orig_segments = []
for line in orig_lines:
    if "[" in line and "]" in line:
        orig_segments.append(line.strip())

print(f"\nOriginal segments: {len(orig_segments)}")
print(f"Edited segments:   {len(result['segments'])}")

print("\nSample comparison:")
print(f"\nOriginal segment 2: {orig_segments[1] if len(orig_segments) > 1 else 'N/A'}")
if len(result['segments']) > 1:
    print(f"Edited segment 2:   [{result['segments'][1]['start']:.2f} - {result['segments'][1]['end']:.2f}]: {result['segments'][1]['text']}")
