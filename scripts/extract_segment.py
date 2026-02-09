"""
Extract specific audio segments to analyze what's actually there.
"""
from pydub import AudioSegment

audio_file = "audio/5h4FDy9Eqzs.wav"
original_audio = AudioSegment.from_file(audio_file)

# Extract key segments
print("Extracting audio segments for analysis...\n")

segments_to_extract = [
    ("fucking_word", 4795, 5055, "The transcribed 'Fucking' word"),
    ("gap_after_fucking", 5055, 6920, "Gap between 'Fucking' and 'hell'"),
    ("hell_word", 6920, 7200, "The word 'hell'"),
    ("extended_fucking", 4795, 5500, "Extended range (in case 'fucking' is longer)"),
]

for name, start_ms, end_ms, description in segments_to_extract:
    segment = original_audio[start_ms:end_ms]
    filename = f"debug_{name}.wav"
    segment.export(filename, format="wav")
    print(f"✓ {filename}: {description}")
    print(f"  Range: {start_ms}-{end_ms}ms ({end_ms-start_ms}ms duration)")
    print(f"  RMS volume: {segment.rms:.1f}")
    print()

print("=" * 70)
print("ANALYSIS:")
print("=" * 70)
print("Please listen to these files to identify what's in each segment:")
print("1. debug_fucking_word.wav - Should contain just 'Fucking'")
print("2. debug_gap_after_fucking.wav - What's in the 1.86 second gap?")
print("3. debug_extended_fucking.wav - Does 'fucking' extend past 5055ms?")
print()
print("If you hear 'fucking' continuing past 5055ms, that's the problem!")
