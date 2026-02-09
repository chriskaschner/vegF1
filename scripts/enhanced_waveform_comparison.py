"""
Enhanced waveform comparison with detailed annotations
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pydub import AudioSegment

# Load audio
original = AudioSegment.from_file("audio/5h4FDy9Eqzs.wav")
edited = AudioSegment.from_file("edited_audio.wav")

def audio_to_array(segment):
    samples = np.array(segment.get_array_of_samples(), dtype=np.float32)
    if segment.channels == 2:
        samples = samples.reshape((-1, 2))
        samples = samples.mean(axis=1)
    # Normalize
    if np.std(samples) > 0:
        samples = samples / np.max(np.abs(samples))
    return samples

# Extract sections
orig_section = original[3000:9000]  # 3-9 seconds
orig_samples = audio_to_array(orig_section)
orig_time = np.linspace(3, 9, len(orig_samples))

# For edited, account for drift (~248ms after first replacement)
edited_section = edited[2752:8752]  # Offset by ~248ms
edited_samples = audio_to_array(edited_section)
edited_time = np.linspace(3, 9, len(edited_samples))

# Create figure
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 0.3], hspace=0.4)

# Plot 1: Original Audio with detailed annotations
ax1 = fig.add_subplot(gs[0])
ax1.plot(orig_time, orig_samples, linewidth=0.3, color='#2E86AB', alpha=0.8)
ax1.set_title('ORIGINAL AUDIO (3-9 seconds)', fontsize=16, fontweight='bold', pad=20)
ax1.set_ylabel('Amplitude', fontsize=12)
ax1.set_xlim(3, 9)
ax1.set_ylim(-1.1, 1.1)
ax1.grid(True, alpha=0.2, linestyle='--')

# Mark segments
# First segment: [2.53 - 4.78]: Kevin, just fucking smash the door...
ax1.axvspan(3.114, 3.434, alpha=0.3, color='red', label='1st "fucking" (replaced)')
ax1.text(3.274, 0.9, 'fucking\n320ms', ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))

# Second segment: [4.79 - 7.20]: Fucking hell.
ax1.axvspan(4.795, 5.055, alpha=0.3, color='darkred', label='2nd "fucking" (replaced)')
ax1.axvspan(5.055, 5.255, alpha=0.3, color='yellow', label='200ms safety buffer (skipped)')
ax1.axvspan(5.255, 6.92, alpha=0.15, color='orange', label='Gap audio (INCLUDED - this is the problem!)')
ax1.axvspan(6.92, 7.20, alpha=0.3, color='green', label='"hell" (kept)')

# Add vertical markers
ax1.axvline(x=4.795, color='darkred', linestyle='--', linewidth=2, alpha=0.7)
ax1.axvline(x=5.055, color='darkred', linestyle='--', linewidth=2, alpha=0.7)
ax1.axvline(x=5.255, color='orange', linestyle='--', linewidth=2, alpha=0.7)
ax1.axvline(x=6.92, color='green', linestyle='--', linewidth=2, alpha=0.7)

# Add text annotations
ax1.text(4.795, -0.9, '4.795s\nWord\nstarts', ha='center', fontsize=8, color='darkred')
ax1.text(5.055, -0.9, '5.055s\nWord\nends', ha='center', fontsize=8, color='darkred')
ax1.text(5.255, -0.9, '5.255s\nBuffer\nends', ha='center', fontsize=8, color='orange')
ax1.text(6.92, -0.9, '6.92s\n"hell"\nstarts', ha='center', fontsize=8, color='green')

ax1.legend(loc='upper right', fontsize=9)

# Plot 2: Edited Audio with annotations
ax2 = fig.add_subplot(gs[1])
ax2.plot(edited_time, edited_samples, linewidth=0.3, color='#06A77D', alpha=0.8)
ax2.set_title('EDITED AUDIO (corresponding section)', fontsize=16, fontweight='bold', pad=20)
ax2.set_ylabel('Amplitude', fontsize=12)
ax2.set_xlim(3, 9)
ax2.set_ylim(-1.1, 1.1)
ax2.grid(True, alpha=0.2, linestyle='--')

# Mark replacements (adjusted for timing in display)
# First replacement ends around 3.386s in output (at original 3s position in our view, this would be ~3.386s)
ax2.axvspan(3.114, 3.386, alpha=0.3, color='purple', label='1st replacement "carrot" (272ms)')
ax2.text(3.25, 0.9, 'carrot\n272ms', ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='purple', alpha=0.3))

# Second replacement: 4.547-4.768 in output (but shifted in our view)
# Due to drift, this appears earlier
repl2_display_start = 4.547
repl2_display_end = 4.768
ax2.axvspan(repl2_display_start, repl2_display_end, alpha=0.3, color='purple', label='2nd replacement "cabbage" (221ms)')
ax2.text((repl2_display_start + repl2_display_end)/2, 0.9, 'cabbage\n221ms', ha='center', fontsize=9,
         bbox=dict(boxstyle='round', facecolor='purple', alpha=0.3))

# The gap audio follows immediately
ax2.axvspan(repl2_display_end, repl2_display_end + 1.665, alpha=0.15, color='orange',
            label='Gap audio from original 5.255-6.92s (1665ms)')

ax2.axvline(x=repl2_display_end, color='orange', linestyle='--', linewidth=2, alpha=0.7)
ax2.text(repl2_display_end, -0.9, f'{repl2_display_end:.3f}s\nGap audio\nstarts here',
         ha='center', fontsize=8, color='orange')

ax2.legend(loc='upper right', fontsize=9)

# Plot 3: Zoomed comparison (4.5-7.5s)
ax3 = fig.add_subplot(gs[2])

zoom_start = 4.5
zoom_end = 7.5

# Extract zoomed sections
zoom_orig = original[int(zoom_start*1000):int(zoom_end*1000)]
zoom_edit = edited[int(zoom_start*1000):int(zoom_end*1000)]

zoom_orig_samples = audio_to_array(zoom_orig)
zoom_edit_samples = audio_to_array(zoom_edit)

zoom_orig_time = np.linspace(zoom_start, zoom_end, len(zoom_orig_samples))
zoom_edit_time = np.linspace(zoom_start, zoom_end, len(zoom_edit_samples))

ax3.plot(zoom_orig_time, zoom_orig_samples, linewidth=0.5, color='#2E86AB', alpha=0.6, label='Original')
ax3.plot(zoom_edit_time, zoom_edit_samples, linewidth=0.5, color='#06A77D', alpha=0.6, label='Edited')

ax3.set_title('ZOOMED OVERLAY COMPARISON (4.5-7.5 seconds)', fontsize=16, fontweight='bold', pad=20)
ax3.set_xlabel('Time (seconds)', fontsize=12)
ax3.set_ylabel('Amplitude', fontsize=12)
ax3.set_xlim(zoom_start, zoom_end)
ax3.set_ylim(-1.1, 1.1)
ax3.grid(True, alpha=0.2, linestyle='--')

# Add critical markers
ax3.axvline(x=4.795, color='red', linestyle=':', linewidth=2, alpha=0.5)
ax3.axvline(x=5.055, color='red', linestyle=':', linewidth=2, alpha=0.5)
ax3.axvline(x=5.255, color='orange', linestyle=':', linewidth=2, alpha=0.5)
ax3.axvspan(4.795, 5.055, alpha=0.1, color='red')
ax3.axvspan(5.055, 5.255, alpha=0.1, color='yellow')
ax3.axvspan(5.255, 6.92, alpha=0.08, color='orange')

ax3.legend(loc='upper right', fontsize=10)

# Plot 4: Legend/Key
ax4 = fig.add_subplot(gs[3])
ax4.axis('off')

legend_text = """
KEY FINDINGS:
• Red shaded area (4.795-5.055s): "Fucking" word - REPLACED with "cabbage" TTS in edited audio
• Yellow area (5.055-5.255s): 200ms safety buffer - SKIPPED in edited audio
• Orange area (5.255-6.92s): 1665ms "gap" audio - INCLUDED in edited audio (RMS: 760 = SPEECH!)
• Green area (6.92-7.20s): "hell" word - KEPT in edited audio

CORRELATION TEST: 95.85% match between edited 4.768s and original 5.255s ✓ CODE WORKING!
PROBLEM: The orange "gap" area contains profanity that WhisperX failed to detect properly
"""

ax4.text(0.05, 0.5, legend_text, fontsize=11, verticalalignment='center',
         family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.savefig('enhanced_waveform_comparison.png', dpi=150, bbox_inches='tight')
print("✓ Enhanced waveform comparison saved to enhanced_waveform_comparison.png")

# Also create a simple side-by-side for the problematic section
fig2, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(16, 6))

# Left: Original 4.79-7.20 (the full segment)
seg_orig = original[4790:7200]
seg_orig_samples = audio_to_array(seg_orig)
seg_orig_time = np.linspace(4.79, 7.20, len(seg_orig_samples))

ax_l.plot(seg_orig_time, seg_orig_samples, linewidth=0.5, color='#2E86AB')
ax_l.set_title('ORIGINAL: Segment [4.79-7.20s]\n"Fucking hell."', fontsize=14, fontweight='bold')
ax_l.set_xlabel('Time (seconds)', fontsize=11)
ax_l.set_ylabel('Amplitude', fontsize=11)
ax_l.grid(True, alpha=0.3)

# Highlight the parts
ax_l.axvspan(4.795, 5.055, alpha=0.3, color='red', label='"Fucking" (replaced)')
ax_l.axvspan(5.055, 5.255, alpha=0.3, color='yellow', label='Safety buffer (skipped)')
ax_l.axvspan(5.255, 6.92, alpha=0.2, color='orange', label='Gap (included)')
ax_l.axvspan(6.92, 7.20, alpha=0.3, color='green', label='"hell" (kept)')
ax_l.legend(fontsize=9)

# Right: Edited corresponding section
# From replacement start (4.547) to equivalent end position
seg_edit = edited[4547:6713]  # 4.547 to ~6.713 (4768 + 1945ms gap audio)
seg_edit_samples = audio_to_array(seg_edit)
seg_edit_time = np.linspace(4.547, 6.713, len(seg_edit_samples))

ax_r.plot(seg_edit_time, seg_edit_samples, linewidth=0.5, color='#06A77D')
ax_r.set_title('EDITED: Corresponding section [4.547-6.713s]\n"[cabbage] + gap audio"', fontsize=14, fontweight='bold')
ax_r.set_xlabel('Time (seconds)', fontsize=11)
ax_r.set_ylabel('Amplitude', fontsize=11)
ax_r.grid(True, alpha=0.3)

# Highlight
ax_r.axvspan(4.547, 4.768, alpha=0.3, color='purple', label='"cabbage" (replacement)')
ax_r.axvspan(4.768, 6.433, alpha=0.2, color='orange', label='Gap from orig 5.255-6.92s')
ax_r.legend(fontsize=9)

plt.tight_layout()
plt.savefig('side_by_side_comparison.png', dpi=150, bbox_inches='tight')
print("✓ Side-by-side comparison saved to side_by_side_comparison.png")

print("\n" + "="*70)
print("Generated visualizations:")
print("  1. enhanced_waveform_comparison.png - Full annotated comparison")
print("  2. side_by_side_comparison.png - Direct side-by-side view")
print("="*70)
