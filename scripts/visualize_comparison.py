"""
Visual comparison of original vs edited audio waveforms
"""
import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment

# Load both audio files
original = AudioSegment.from_file("audio/5h4FDy9Eqzs.wav")
edited = AudioSegment.from_file("edited_audio.wav")

# Extract the problematic section: 4-8 seconds in original
start_sec = 4
end_sec = 8
original_section = original[start_sec * 1000:end_sec * 1000]

# For edited audio, we need to calculate where this section appears
# Based on the replacement log:
# After first replacement at 3114-3434ms, we have drift
# The second segment starts at 4790ms original, but appears at ~4547ms in edited
# So we need to look at edited audio starting around 4.5 seconds

# From replacement log: [4795-5055ms] fucking -> [4547-4768ms] cabbage
# So original 4000-8000ms roughly corresponds to edited 3752-7752ms (accounting for ~248ms drift from first replacement)
edited_start_ms = 3752  # Approximate position in edited audio
edited_end_ms = edited_start_ms + 4000
edited_section = edited[edited_start_ms:edited_end_ms]

# Convert to numpy arrays for plotting
def audio_to_array(segment):
    samples = np.array(segment.get_array_of_samples())
    if segment.channels == 2:
        samples = samples.reshape((-1, 2))
        samples = samples.mean(axis=1)  # Convert stereo to mono
    return samples

original_samples = audio_to_array(original_section)
edited_samples = audio_to_array(edited_section)

# Time axes
original_time = np.linspace(start_sec, end_sec, len(original_samples))
edited_time = np.linspace(edited_start_ms/1000, edited_end_ms/1000, len(edited_samples))

# Create figure with subplots
fig, axes = plt.subplots(3, 1, figsize=(15, 10))

# Plot 1: Original audio
axes[0].plot(original_time, original_samples, linewidth=0.5, color='blue', alpha=0.7)
axes[0].set_title('ORIGINAL AUDIO (4-8 seconds)', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Amplitude')
axes[0].grid(True, alpha=0.3)

# Mark the word boundaries in original
# "Fucking" at 4.795-5.055, "hell" at 6.92-7.20
axes[0].axvline(x=4.795, color='red', linestyle='--', linewidth=2, label='Start "Fucking"')
axes[0].axvline(x=5.055, color='red', linestyle='--', linewidth=2, label='End "Fucking"')
axes[0].axvspan(4.795, 5.055, alpha=0.2, color='red', label='Word "Fucking"')
axes[0].axvline(x=6.92, color='orange', linestyle='--', linewidth=2, label='Start "hell"')
axes[0].axvline(x=7.20, color='orange', linestyle='--', linewidth=2, label='End "hell"')
axes[0].axvspan(6.92, 7.20, alpha=0.2, color='orange', label='Word "hell"')

# Mark what should be skipped (5.055 to 5.255 due to 200ms buffer)
axes[0].axvspan(5.055, 5.255, alpha=0.3, color='yellow', label='200ms safety buffer')

axes[0].legend(loc='upper right', fontsize=8)

# Plot 2: Edited audio
axes[1].plot(edited_time, edited_samples, linewidth=0.5, color='green', alpha=0.7)
axes[1].set_title('EDITED AUDIO (corresponding section)', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Amplitude')
axes[1].grid(True, alpha=0.3)

# Mark the replacement position in edited audio
# From log: [4547-4768ms] cabbage replacement
repl_start = 4.547
repl_end = 4.768
axes[1].axvline(x=repl_start, color='green', linestyle='--', linewidth=2, label='Start replacement')
axes[1].axvline(x=repl_end, color='green', linestyle='--', linewidth=2, label='End replacement')
axes[1].axvspan(repl_start, repl_end, alpha=0.3, color='green', label='Replacement "cabbage"')
axes[1].legend(loc='upper right', fontsize=8)

# Plot 3: Overlay comparison (zoomed to the critical area)
zoom_start = 4.5
zoom_end = 6.0
zoom_original_start_idx = int((zoom_start - start_sec) * original.frame_rate)
zoom_original_end_idx = int((zoom_end - start_sec) * original.frame_rate)
zoom_edited_start_idx = int((zoom_start - edited_start_ms/1000) * edited.frame_rate)
zoom_edited_end_idx = int((zoom_end - edited_start_ms/1000) * edited.frame_rate)

if zoom_edited_end_idx <= len(edited_samples):
    zoom_original = original_samples[zoom_original_start_idx:zoom_original_end_idx]
    zoom_edited = edited_samples[zoom_edited_start_idx:zoom_edited_end_idx]

    zoom_original_time = np.linspace(zoom_start, zoom_end, len(zoom_original))
    zoom_edited_time = np.linspace(zoom_start, zoom_end, len(zoom_edited))

    axes[2].plot(zoom_original_time, zoom_original, linewidth=0.8, color='blue', alpha=0.6, label='Original')
    axes[2].plot(zoom_edited_time, zoom_edited, linewidth=0.8, color='green', alpha=0.6, label='Edited')
    axes[2].set_title('ZOOMED COMPARISON (4.5-6.0 seconds)', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Time (seconds)')
    axes[2].set_ylabel('Amplitude')
    axes[2].grid(True, alpha=0.3)

    # Mark critical points
    axes[2].axvline(x=4.795, color='red', linestyle=':', linewidth=1, alpha=0.7, label='Original word start')
    axes[2].axvline(x=5.055, color='red', linestyle=':', linewidth=1, alpha=0.7, label='Original word end')
    axes[2].axvspan(4.795, 5.055, alpha=0.1, color='red')
    axes[2].axvspan(repl_start, repl_end, alpha=0.1, color='green')
    axes[2].legend(loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig('waveform_comparison.png', dpi=150, bbox_inches='tight')
print("✓ Waveform comparison saved to waveform_comparison.png")

# Also print detailed analysis
print("\n" + "="*70)
print("DETAILED ANALYSIS")
print("="*70)
print(f"\nOriginal audio (4-8 sec):")
print(f"  - 'Fucking' word: 4.795-5.055s (260ms)")
print(f"  - Should be REPLACED in edited audio")
print(f"  - Safety buffer: 5.055-5.255s (200ms)")
print(f"  - Should be SKIPPED in edited audio")
print(f"  - Audio from 5.255s onwards: Should appear in edited audio")
print(f"\nEdited audio:")
print(f"  - Replacement 'cabbage': 4.547-4.768s (221ms)")
print(f"  - Audio after replacement starts at: 4.768s")
print(f"  - This corresponds to original 5.255s (after word + buffer)")
print(f"\nExpected in edited audio at 4.768s onwards:")
print(f"  - Should hear audio from original 5.255s (NOT the swear word)")
print(f"\nIF you hear the swear word, check:")
print(f"  1. Is the waveform at edited 4.768s matching original 5.255s?")
print(f"  2. Or is it matching original 4.795s or earlier?")
print(f"  3. Visual comparison should reveal the mismatch")
