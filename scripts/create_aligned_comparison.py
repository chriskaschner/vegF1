"""
Create aligned timeline comparisons with shared x-axis.
Shows original and edited on the same timeline for easy comparison.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pydub import AudioSegment
import json

# Load audio files
original = AudioSegment.from_file("audio/5h4FDy9Eqzs.wav")
edited = AudioSegment.from_file("edited_audio_timeline.wav")

# Load replacement log
with open("replacements_log_timeline.txt", "r") as f:
    lines = f.readlines()

replacements = []
for line in lines[3:]:  # Skip header
    if "->" in line and "[" in line:
        # Parse: [2530-4780ms] : Kevin... -> butternut squash, fenugreek, garlic (Original: 2250ms, Replacement: 2146ms)
        orig_part = line.split("]")[0].split("[")[1]
        orig_start, orig_end = map(lambda x: int(x.replace("ms", "")), orig_part.split("-"))

        text_part = line.split("]")[1].split("->")[0].strip()
        veg_part = line.split("->")[1].split("(")[0].strip()

        durations_part = line.split("(")[1].split(")")[0]
        orig_dur = int(durations_part.split("Original:")[1].split("ms")[0].strip())
        repl_dur = int(durations_part.split("Replacement:")[1].split("ms")[0].strip())

        replacements.append({
            "orig_start": orig_start,
            "orig_end": orig_end,
            "orig_duration": orig_dur,
            "text": text_part,
            "vegetables": veg_part,
            "repl_duration": repl_dur
        })

def audio_to_array(segment):
    """Convert audio to normalized array"""
    samples = np.array(segment.get_array_of_samples(), dtype=np.float32)
    if segment.channels == 2:
        samples = samples.reshape((-1, 2)).mean(axis=1)
    if np.std(samples) > 0:
        samples = samples / np.max(np.abs(samples))
    return samples

# Create figure with ALIGNED timelines (shared x-axis)
fig = plt.figure(figsize=(20, 10))
gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 0.5], hspace=0.3)

# SHARED X-AXIS: Use ORIGINAL timeline (0 to 36.5 seconds)
time_extent = (0, len(original) / 1000)

# Plot 1: ORIGINAL AUDIO
ax1 = fig.add_subplot(gs[0])
orig_samples = audio_to_array(original)
orig_time = np.linspace(0, len(original) / 1000, len(orig_samples))

ax1.plot(orig_time, orig_samples, linewidth=0.2, color='#2E86AB', alpha=0.8)
ax1.set_title('ORIGINAL AUDIO - Full Timeline', fontsize=16, fontweight='bold', pad=15)
ax1.set_ylabel('Amplitude', fontsize=12)
ax1.set_xlim(time_extent)
ax1.set_ylim(-1.1, 1.1)
ax1.grid(True, alpha=0.2, linestyle='--')

# Highlight replacements on original timeline
for i, repl in enumerate(replacements):
    start_sec = repl['orig_start'] / 1000
    end_sec = repl['orig_end'] / 1000

    # Color code by replacement type
    colors = ['#FF6B6B', '#FF8E53', '#FFA07A', '#FFB84D', '#FFCB5C', '#FFE66D', '#C7EFCF', '#A7E1AF']
    color = colors[i % len(colors)]

    ax1.axvspan(start_sec, end_sec, alpha=0.3, color=color)

    # Add label
    mid_point = (start_sec + end_sec) / 2
    if i % 2 == 0:
        y_pos = 0.85
    else:
        y_pos = -0.85

    ax1.text(mid_point, y_pos, f"R{i+1}", ha='center', fontsize=9,
             bbox=dict(boxstyle='circle', facecolor=color, alpha=0.7))

ax1.set_xlabel('Time (seconds) - Original Timeline', fontsize=11)

# Plot 2: EDITED AUDIO - ALIGNED TO ORIGINAL TIMELINE
ax2 = fig.add_subplot(gs[1], sharex=ax1)

# Build a mapping of original timeline to edited timeline
# We need to show where each part of the edited audio came from in the original

# Create edited audio representation aligned to original timeline
# Strategy: Map each millisecond of original time to edited position
orig_to_edited_map = np.zeros(len(original))  # Map: original_ms -> edited_ms
edited_to_orig_map = {}  # Map: edited_ms -> original_ms

current_edited_pos = 0
current_orig_pos = 0

# Build the timeline mapping
for i, repl in enumerate(replacements):
    # Before replacement: original audio is kept
    if repl['orig_start'] > current_orig_pos:
        keep_duration = repl['orig_start'] - current_orig_pos
        for ms in range(current_orig_pos, repl['orig_start']):
            orig_to_edited_map[ms] = current_edited_pos + (ms - current_orig_pos)
        current_edited_pos += keep_duration
        current_orig_pos = repl['orig_start']

    # Replacement: original segment maps to replacement duration
    for ms in range(repl['orig_start'], repl['orig_end']):
        # This original audio is replaced, mark as -1
        orig_to_edited_map[ms] = -1

    current_edited_pos += repl['repl_duration']
    current_orig_pos = repl['orig_end']

# After last replacement
if current_orig_pos < len(original):
    for ms in range(current_orig_pos, len(original)):
        orig_to_edited_map[ms] = current_edited_pos + (ms - current_orig_pos)

# Now plot edited audio samples, but stretched/compressed to align with original timeline
edited_samples = audio_to_array(edited)

# For visualization, we'll show edited audio in segments
# Plot each kept segment and each replacement

ax2.set_title('EDITED AUDIO - Aligned to Original Timeline', fontsize=16, fontweight='bold', pad=15)
ax2.set_ylabel('Amplitude', fontsize=12)
ax2.set_xlim(time_extent)
ax2.set_ylim(-1.1, 1.1)
ax2.grid(True, alpha=0.2, linestyle='--')

# Plot edited audio segments aligned to original timeline
current_edited_ms = 0

for i, repl in enumerate(replacements):
    # Plot kept audio before this replacement
    if i == 0:
        keep_start_orig = 0
    else:
        keep_start_orig = replacements[i-1]['orig_end']

    keep_end_orig = repl['orig_start']

    if keep_end_orig > keep_start_orig:
        # Calculate how much edited audio this represents
        keep_duration_ms = keep_end_orig - keep_start_orig
        keep_start_edited_sample = int((current_edited_ms / len(edited)) * len(edited_samples))
        keep_end_edited_sample = int(((current_edited_ms + keep_duration_ms) / len(edited)) * len(edited_samples))

        if keep_end_edited_sample > keep_start_edited_sample:
            # Extract and plot
            segment_samples = edited_samples[keep_start_edited_sample:keep_end_edited_sample]
            segment_time = np.linspace(keep_start_orig / 1000, keep_end_orig / 1000, len(segment_samples))
            ax2.plot(segment_time, segment_samples, linewidth=0.2, color='#06A77D', alpha=0.8)

        current_edited_ms += keep_duration_ms

    # Plot replacement - show on original timeline where it replaces
    colors = ['#FF6B6B', '#FF8E53', '#FFA07A', '#FFB84D', '#FFCB5C', '#FFE66D', '#C7EFCF', '#A7E1AF']
    color = colors[i % len(colors)]

    start_sec = repl['orig_start'] / 1000
    end_sec = repl['orig_end'] / 1000

    # Plot replacement audio
    repl_start_edited_sample = int((current_edited_ms / len(edited)) * len(edited_samples))
    repl_end_edited_sample = int(((current_edited_ms + repl['repl_duration']) / len(edited)) * len(edited_samples))

    if repl_end_edited_sample > repl_start_edited_sample:
        segment_samples = edited_samples[repl_start_edited_sample:repl_end_edited_sample]
        # Plot on original timeline (stretched/compressed)
        segment_time = np.linspace(start_sec, end_sec, len(segment_samples))
        ax2.plot(segment_time, segment_samples, linewidth=0.3, color=color, alpha=0.9, zorder=10)

    # Shade the replacement region
    ax2.axvspan(start_sec, end_sec, alpha=0.3, color=color, zorder=5)

    # Add label
    mid_point = (start_sec + end_sec) / 2
    if i % 2 == 0:
        y_pos = 0.85
    else:
        y_pos = -0.85

    veg_names = repl['vegetables']
    if len(veg_names) > 30:
        veg_names = veg_names[:27] + "..."

    ax2.text(mid_point, y_pos, f"R{i+1}", ha='center', fontsize=9,
             bbox=dict(boxstyle='circle', facecolor=color, alpha=0.7))

    current_edited_ms += repl['repl_duration']

# Plot remaining audio after last replacement
if replacements:
    keep_start_orig = replacements[-1]['orig_end']
    keep_end_orig = len(original)

    if keep_end_orig > keep_start_orig:
        keep_duration_ms = keep_end_orig - keep_start_orig
        keep_start_edited_sample = int((current_edited_ms / len(edited)) * len(edited_samples))
        keep_end_edited_sample = len(edited_samples)

        if keep_end_edited_sample > keep_start_edited_sample:
            segment_samples = edited_samples[keep_start_edited_sample:keep_end_edited_sample]
            segment_time = np.linspace(keep_start_orig / 1000, keep_end_orig / 1000, len(segment_samples))
            ax2.plot(segment_time, segment_samples, linewidth=0.2, color='#06A77D', alpha=0.8)

ax2.set_xlabel('Time (seconds) - Original Timeline', fontsize=11)

# Plot 3: LEGEND / KEY
ax3 = fig.add_subplot(gs[2])
ax3.axis('off')

legend_text = "REPLACEMENT KEY:\n"
for i, repl in enumerate(replacements):
    duration_change = repl['repl_duration'] - repl['orig_duration']
    sign = "+" if duration_change > 0 else ""
    legend_text += f"R{i+1}: {repl['vegetables']} ({sign}{duration_change}ms)\n"

legend_text += f"\nSUMMARY:\n"
legend_text += f"• Original duration: {len(original)/1000:.2f}s\n"
legend_text += f"• Edited duration: {len(edited)/1000:.2f}s\n"
legend_text += f"• Time saved: {(len(original)-len(edited))/1000:.2f}s ({100*(len(original)-len(edited))/len(original):.1f}% reduction)\n"
legend_text += f"• Replacements: {len(replacements)} segments\n"
legend_text += f"• Approach: Timeline-based, segment-level replacement with multiple vegetables"

ax3.text(0.05, 0.5, legend_text, fontsize=10, verticalalignment='center',
         family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))

plt.savefig('aligned_timeline_comparison.png', dpi=150, bbox_inches='tight')
print("✓ Aligned timeline comparison saved to aligned_timeline_comparison.png")

# Create zoomed view of problematic section (4-8 seconds)
fig2, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

zoom_start = 4.0
zoom_end = 8.0

# Original zoomed
zoom_orig = original[int(zoom_start*1000):int(zoom_end*1000)]
zoom_orig_samples = audio_to_array(zoom_orig)
zoom_orig_time = np.linspace(zoom_start, zoom_end, len(zoom_orig_samples))

ax_top.plot(zoom_orig_time, zoom_orig_samples, linewidth=0.4, color='#2E86AB')
ax_top.set_title(f'ORIGINAL: {zoom_start}-{zoom_end}s', fontsize=14, fontweight='bold')
ax_top.set_ylabel('Amplitude', fontsize=11)
ax_top.grid(True, alpha=0.3)

# Mark the replacement
for repl in replacements:
    if zoom_start * 1000 <= repl['orig_start'] < zoom_end * 1000:
        start_sec = repl['orig_start'] / 1000
        end_sec = repl['orig_end'] / 1000
        ax_top.axvspan(start_sec, end_sec, alpha=0.3, color='red')
        ax_top.text((start_sec + end_sec) / 2, 0.8, repl['text'][:20],
                   ha='center', fontsize=8, bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))

# Edited zoomed - need to find corresponding section
# Find where 4.0-8.0s maps to in edited audio
edited_start_ms = int(orig_to_edited_map[int(zoom_start * 1000)])
edited_end_ms = int(orig_to_edited_map[int(zoom_end * 1000) - 1])

# But we need to account for the replacement
# Actually, let's just plot the edited segment with proper alignment
for repl in replacements:
    if zoom_start * 1000 <= repl['orig_start'] < zoom_end * 1000:
        # This replacement is in our zoom window
        # Plot before replacement
        if repl['orig_start'] > zoom_start * 1000:
            # Calculate edited position
            pre_samples_start = int((edited_start_ms / len(edited)) * len(edited_samples))
            pre_duration = repl['orig_start'] - zoom_start * 1000
            pre_samples_end = int(((edited_start_ms + pre_duration) / len(edited)) * len(edited_samples))

            if pre_samples_end > pre_samples_start:
                pre_segment = edited_samples[pre_samples_start:pre_samples_end]
                pre_time = np.linspace(zoom_start, repl['orig_start'] / 1000, len(pre_segment))
                ax_bottom.plot(pre_time, pre_segment, linewidth=0.4, color='#06A77D')

        # Plot replacement
        repl_edited_start_ms = edited_start_ms + (repl['orig_start'] - zoom_start * 1000)
        repl_samples_start = int((repl_edited_start_ms / len(edited)) * len(edited_samples))
        repl_samples_end = int(((repl_edited_start_ms + repl['repl_duration']) / len(edited)) * len(edited_samples))

        if repl_samples_end > repl_samples_start:
            repl_segment = edited_samples[repl_samples_start:repl_samples_end]
            repl_time = np.linspace(repl['orig_start'] / 1000, repl['orig_end'] / 1000, len(repl_segment))
            ax_bottom.plot(repl_time, repl_segment, linewidth=0.4, color='purple')
            ax_bottom.axvspan(repl['orig_start'] / 1000, repl['orig_end'] / 1000, alpha=0.3, color='purple')
            ax_bottom.text((repl['orig_start'] + repl['orig_end']) / 2000, 0.8, repl['vegetables'],
                          ha='center', fontsize=8, bbox=dict(boxstyle='round', facecolor='purple', alpha=0.3))

ax_bottom.set_title(f'EDITED: {zoom_start}-{zoom_end}s (aligned to original timeline)', fontsize=14, fontweight='bold')
ax_bottom.set_xlabel('Time (seconds)', fontsize=11)
ax_bottom.set_ylabel('Amplitude', fontsize=11)
ax_bottom.set_xlim(zoom_start, zoom_end)
ax_bottom.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('zoomed_aligned_comparison.png', dpi=150, bbox_inches='tight')
print("✓ Zoomed aligned comparison saved to zoomed_aligned_comparison.png")

print("\n" + "="*70)
print("GENERATED VISUALIZATIONS:")
print("="*70)
print("1. aligned_timeline_comparison.png - Full timeline with aligned x-axis")
print("2. zoomed_aligned_comparison.png - Zoomed 4-8s section")
print("\nBoth use the ORIGINAL TIMELINE as the shared x-axis for easy comparison!")
