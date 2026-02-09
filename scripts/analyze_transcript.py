"""
Analyze what WhisperX actually transcribed to see if it missed any profanity
"""
from pydub import AudioSegment

# Read transcript
with open("transcript.txt", "r") as f:
    lines = f.readlines()

print("="*70)
print("TRANSCRIPT ANALYSIS: 4-8 second section")
print("="*70)

# Parse transcript for the 4-8 second section
relevant_lines = []
for line in lines:
    if not line.strip() or line.startswith("Format:"):
        continue
    if "[" in line:
        # Extract timestamp
        try:
            time_part = line.split("]")[0].split("[")[1]
            start_time = float(time_part.split("-")[0].strip())
            if 3.0 <= start_time <= 9.0:  # Broader range to see context
                relevant_lines.append(line.strip())
        except:
            pass

print("\nTranscript lines around 4-8 seconds:")
for line in relevant_lines:
    print(f"  {line}")

# Check the actual audio content
print("\n" + "="*70)
print("ACTUAL AUDIO CONTENT")
print("="*70)

original = AudioSegment.from_file("audio/5h4FDy9Eqzs.wav")

# Extract the "gap" audio (5255-6920ms) that we're including in output
gap_audio = original[5255:6920]
print(f"\nAudio from 5255-6920ms (the 'gap' after buffer):")
print(f"  Duration: {len(gap_audio)}ms ({len(gap_audio)/1000:.2f}s)")
print(f"  RMS: {gap_audio.rms:.1f}")
print(f"  This is what appears in edited audio after the replacement")

# Save it for manual inspection
gap_audio.export("gap_audio_5255_6920.wav", format="wav")
print(f"  ✓ Exported to gap_audio_5255_6920.wav")

print(f"\nAccording to transcript:")
print(f"  - 'Fucking' ends at 5.055s")
print(f"  - 'hell' starts at 6.92s")
print(f"  - Gap: 5.055 to 6.92 (1.865 seconds)")
print(f"\nWe skip: 5.055 to 5.255 (200ms buffer)")
print(f"We include: 5.255 to 6.92 (1.665 seconds)")

print(f"\n{'!'*70}")
print("HYPOTHESIS:")
print("!"*70)
print("If you're hearing profanity, it could be:")
print("1. The word 'fucking' actually extends past 5.055s (transcription error)")
print("2. There's additional profanity in the 5.255-6.92s range that wasn't transcribed")
print("3. Video/audio desync in the final MP4 file")
print(f"\nPlease listen to 'gap_audio_5255_6920.wav' and tell me:")
print("  - Do you hear any profanity in this clip?")
print("  - What exactly do you hear?")
