"""
Verify that audio after replacement matches expected original position
"""
import numpy as np
from pydub import AudioSegment
from scipy import signal

# Load audio files
original = AudioSegment.from_file("audio/5h4FDy9Eqzs.wav")
edited = AudioSegment.from_file("edited_audio.wav")

def audio_to_array(segment):
    """Convert audio segment to numpy array"""
    samples = np.array(segment.get_array_of_samples(), dtype=np.float32)
    if segment.channels == 2:
        samples = samples.reshape((-1, 2))
        samples = samples.mean(axis=1)  # Mono
    return samples

print("="*70)
print("AUDIO MATCHING VERIFICATION")
print("="*70)

# According to our logic:
# - Replacement ends at 4768ms in edited audio
# - Next audio should come from 5255ms in original (5055ms word end + 200ms buffer)

# Extract 500ms samples to compare
edited_after_repl_start = 4768
edited_after_repl = edited[edited_after_repl_start:edited_after_repl_start + 500]

# What SHOULD be there (from original 5255ms)
expected_start = 5255
expected_audio = original[expected_start:expected_start + 500]

# What SHOULD NOT be there (from original swear word area)
wrong_start_1 = 4795  # Start of "Fucking"
wrong_audio_1 = original[wrong_start_1:wrong_start_1 + 500]

wrong_start_2 = 5055  # End of "Fucking" (no buffer)
wrong_audio_2 = original[wrong_start_2:wrong_start_2 + 500]

# Convert to arrays
edited_arr = audio_to_array(edited_after_repl)
expected_arr = audio_to_array(expected_audio)
wrong_arr_1 = audio_to_array(wrong_audio_1)
wrong_arr_2 = audio_to_array(wrong_audio_2)

# Normalize for comparison
def normalize(arr):
    if len(arr) == 0 or np.std(arr) == 0:
        return arr
    return (arr - np.mean(arr)) / np.std(arr)

edited_norm = normalize(edited_arr)
expected_norm = normalize(expected_arr)
wrong_norm_1 = normalize(wrong_arr_1)
wrong_norm_2 = normalize(wrong_arr_2)

# Calculate correlation
def correlation(a, b):
    """Calculate correlation coefficient"""
    if len(a) != len(b):
        min_len = min(len(a), len(b))
        a = a[:min_len]
        b = b[:min_len]
    if len(a) == 0:
        return 0.0
    return np.corrcoef(a, b)[0, 1]

corr_expected = correlation(edited_norm, expected_norm)
corr_wrong_1 = correlation(edited_norm, wrong_norm_1)
corr_wrong_2 = correlation(edited_norm, wrong_norm_2)

print(f"\nComparing edited audio at {edited_after_repl_start}ms with original audio:")
print(f"\n1. Correlation with EXPECTED position (original 5255ms):")
print(f"   Correlation: {corr_expected:.4f}")
print(f"   {'✓ MATCH!' if corr_expected > 0.7 else '✗ NO MATCH'}")

print(f"\n2. Correlation with WRONG position #1 (original 4795ms - start of swear):")
print(f"   Correlation: {corr_wrong_1:.4f}")
print(f"   {'✗ PROBLEM! Audio matches swear word area!' if corr_wrong_1 > 0.7 else '✓ Good, no match with swear'}")

print(f"\n3. Correlation with WRONG position #2 (original 5055ms - end of swear, no buffer):")
print(f"   Correlation: {corr_wrong_2:.4f}")
print(f"   {'✗ PROBLEM! Audio matches end of swear (buffer not working)!' if corr_wrong_2 > 0.7 else '✓ Good, no match'}")

# Determine verdict
print("\n" + "="*70)
print("VERDICT:")
print("="*70)

if corr_expected > 0.7:
    print("✓ Code is working correctly!")
    print("  The audio after replacement matches the expected position (original 5255ms)")
    print("  This means the swear word AND the 200ms buffer are being skipped properly")
elif corr_wrong_1 > 0.7:
    print("✗ BUG FOUND: Audio matches position 4795ms (START of swear word)!")
    print("  This means the replacement is being INSERTED, not REPLACING")
    print("  The original swear word is still present in the output")
elif corr_wrong_2 > 0.7:
    print("✗ BUG FOUND: Audio matches position 5055ms (END of swear, no buffer)!")
    print("  This means the 200ms safety buffer is NOT being applied")
    print("  We're only skipping the exact word, not the buffer after it")
else:
    print("? UNCLEAR: Audio doesn't strongly match any expected position")
    print(f"  Expected correlation: {corr_expected:.4f}")
    print(f"  Wrong #1 correlation: {corr_wrong_1:.4f}")
    print(f"  Wrong #2 correlation: {corr_wrong_2:.4f}")
    print("  The audio might be heavily processed or from an unexpected position")

# Also check RMS levels
print(f"\nRMS levels:")
print(f"  Edited after repl: {edited_after_repl.rms:.1f}")
print(f"  Expected (orig 5255ms): {expected_audio.rms:.1f}")
print(f"  Wrong #1 (orig 4795ms): {wrong_audio_1.rms:.1f}")
print(f"  Wrong #2 (orig 5055ms): {wrong_audio_2.rms:.1f}")

# Export samples for manual listening
edited_after_repl.export("verify_edited_after_replacement.wav", format="wav")
expected_audio.export("verify_expected_audio.wav", format="wav")
wrong_audio_1.export("verify_wrong_audio_1.wav", format="wav")
wrong_audio_2.export("verify_wrong_audio_2.wav", format="wav")

print(f"\n✓ Exported audio samples for manual verification:")
print(f"  - verify_edited_after_replacement.wav (what's actually in edited)")
print(f"  - verify_expected_audio.wav (what SHOULD be there)")
print(f"  - verify_wrong_audio_1.wav (if bug: inserted instead of replaced)")
print(f"  - verify_wrong_audio_2.wav (if bug: buffer not working)")
