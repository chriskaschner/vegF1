"""
Check what's in the edited audio output at the same positions.
"""
from pydub import AudioSegment

edited_audio = AudioSegment.from_file("edited_audio.wav")

print("Checking edited audio output...")
print(f"Total length: {len(edited_audio)}ms\n")

# According to replacement log, the second segment should be at:
# [4747-4968ms] turnip replacement
# Then audio continues from original 5055ms onwards

print("Expected edited audio structure:")
print("- [4747-4968ms]: 'turnip' replacement (221ms)")
print("- [4968-7113ms]: Original audio from 5055-7200ms (2145ms)")
print()

# Extract the replacement and what comes after
replacement_segment = edited_audio[4747:4968]
after_replacement = edited_audio[4968:5500]  # First 532ms after replacement

replacement_segment.export("debug_edited_replacement.wav", format="wav")
after_replacement.export("debug_edited_after_replacement.wav", format="wav")

print("Extracted from edited audio:")
print(f"✓ debug_edited_replacement.wav: Replacement at 4747-4968ms")
print(f"  RMS: {replacement_segment.rms:.1f}")
print(f"✓ debug_edited_after_replacement.wav: Audio immediately after (4968-5500ms)")
print(f"  RMS: {after_replacement.rms:.1f}")
print()

# Compare to original
from_original = AudioSegment.from_file("audio/5h4FDy9Eqzs.wav")[5055:5587]
from_original.export("debug_original_5055_5587.wav", format="wav")
print(f"✓ debug_original_5055_5587.wav: Original audio from 5055-5587ms (what SHOULD be after replacement)")
print(f"  RMS: {from_original.rms:.1f}")
print()

print("=" * 70)
print("TO VERIFY:")
print("=" * 70)
print("Listen to these files in order:")
print("1. debug_edited_replacement.wav - Should sound like 'turnip' or vegetable name")
print("2. debug_edited_after_replacement.wav - What comes right after the replacement?")
print("3. debug_original_5055_5587.wav - What SHOULD come after (from original 5055ms)")
print()
print("If #2 and #3 sound different, we have a problem!")
print("If #2 contains 'fucking' or profanity, that's the bug!")
