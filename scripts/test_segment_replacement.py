"""
Test replacing the ENTIRE SEGMENT instead of just the word.
This bypasses potentially incorrect word-level timestamps.
"""
from pydub import AudioSegment

# Load original audio
original = AudioSegment.from_file("audio/5h4FDy9Eqzs.wav")

# The problematic segment according to transcript: [4.79 - 7.20]: "Fucking hell."
segment_start_ms = int(4.79 * 1000)  # 4790ms
segment_end_ms = int(7.20 * 1000)     # 7200ms

print("="*70)
print("SEGMENT-LEVEL REPLACEMENT TEST")
print("="*70)
print(f"\nSegment: 'Fucking hell.' at {segment_start_ms}-{segment_end_ms}ms")
print(f"Duration: {segment_end_ms - segment_start_ms}ms")

# Build output using segment-level replacement
output = AudioSegment.empty()

# Add everything before the segment
output += original[0:segment_start_ms]
print(f"✓ Added audio before segment: 0-{segment_start_ms}ms ({len(output)}ms)")

# Load a replacement
replacement = AudioSegment.from_file("tts_cache/cabbage.mp3")
print(f"✓ Loaded replacement 'cabbage': {len(replacement)}ms")

# Add replacement
output += replacement
print(f"✓ Added replacement: now at {len(output)}ms")

# Add everything after the segment
output += original[segment_end_ms:]
print(f"✓ Added audio after segment: {segment_end_ms}ms to end")
print(f"✓ Final length: {len(output)}ms")

# Export
output.export("test_segment_replacement.wav", format="wav")
print(f"\n✓ Exported to test_segment_replacement.wav")

print("\n" + "="*70)
print("COMPARISON:")
print("="*70)
print(f"Original segment: {segment_start_ms}-{segment_end_ms}ms ({segment_end_ms-segment_start_ms}ms)")
print(f"  Contains: 'Fucking hell.'")
print(f"  COMPLETELY SKIPPED in this test")
print(f"\nReplacement: {len(replacement)}ms 'cabbage'")
print(f"  Replaces the ENTIRE segment")
print(f"\nIf you still hear profanity in test_segment_replacement.wav:")
print(f"  - The profanity is in a DIFFERENT segment")
print(f"  - Or WhisperX got the SEGMENT boundaries wrong too")
print(f"\nIf test_segment_replacement.wav is clean:")
print(f"  - Word-level timestamps are the problem")
print(f"  - We should use segment-level replacement for this segment")
