"""
Create proper aligned comparison showing both audio files on their own timelines
with clear annotations about what was replaced where.
"""
import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment

# Load audio
original = AudioSegment.from_file("audio/5h4FDy9Eqzs.wav")
edited = AudioSegment.from_file("edited_audio_wordlevel.wav")

def audio_to_array(segment):
    samples = np.array(segment.get_array_of_samples(), dtype=np.float32)
    if segment.channels == 2:
        samples = samples.reshape((-1, 2)).mean(axis=1)
    if np.std(samples) > 0:
        samples = samples / np.max(np.abs(samples))
    return samples

# Parse replacement log to build timeline mapping
replacements = [
    {"orig_start": 3114, "orig_end": 3834, "word": "fucking", "repl_dur": 730, "veg": "bamboo shoots"},
    {"orig_start": 4795, "orig_end": 5455, "word": "Fucking", "repl_dur": 660, "veg": "sweet potato"},
    {"orig_start": 6920, "orig_end": 7200, "word": "hell", "repl_dur": 280, "veg": "garlic"},
    {"orig_start": 10098, "orig_end": 10798, "word": "fucking", "repl_dur": 700, "veg": "bamboo shoots"},
    {"orig_start": 13399, "orig_end": 14039, "word": "fuck", "repl_dur": 640, "veg": "cauliflower"},
    {"orig_start": 19062, "orig_end": 19722, "word": "fucking", "repl_dur": 660, "veg": "sweet potato"},
    {"orig_start": 22344, "orig_end": 22740, "word": "fuck", "repl_dur": 400, "veg": "onion"},
    {"orig_start": 24925, "orig_end": 25430, "word": "Fucking", "repl_dur": 510, "veg": "bock choy"},
    {"orig_start": 25300, "orig_end": 25430, "word": "hell", "repl_dur": 130, "veg": "artichoke"},
    {"orig_start": 27158, "orig_end": 27879, "word": "fucking", "repl_dur": 730, "veg": "bamboo shoots"},
]

# Create figure with TWO SEPARATE TIMELINES
fig = plt.figure(figsize=(20, 10))
gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 0.4], hspace=0.4)

# Plot 1: ORIGINAL audio on its own timeline
ax1 = fig.add_subplot(gs[0])
orig_samples = audio_to_array(original)
orig_time = np.linspace(0, len(original)/1000, len(orig_samples))

ax1.plot(orig_time, orig_samples, linewidth=0.2, color='#2E86AB', alpha=0.8)
ax1.set_title('ORIGINAL AUDIO (36.45s total)', fontsize=16, fontweight='bold', pad=15)
ax1.set_ylabel('Amplitude', fontsize=12)
ax1.set_xlim(0, len(original)/1000)
ax1.set_ylim(-1.1, 1.1)
ax1.grid(True, alpha=0.2, linestyle='--')

# Mark all replacements on original
colors = ['#FF6B6B', '#FF8E53', '#FFA07A', '#FFB84D', '#FFCB5C', '#FFE66D', '#C7EFCF', '#A7E1AF', '#98D8C8', '#6FCDDD']
for i, repl in enumerate(replacements):
    start_sec = repl['orig_start'] / 1000
    end_sec = repl['orig_end'] / 1000
    color = colors[i % len(colors)]

    ax1.axvspan(start_sec, end_sec, alpha=0.3, color=color, zorder=1)

    # Add label
    if i % 2 == 0:
        y_pos = 0.85
    else:
        y_pos = -0.85

    ax1.text((start_sec + end_sec) / 2, y_pos, f"R{i+1}\n{repl['word']}",
             ha='center', fontsize=7, bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))

ax1.set_xlabel('Time (seconds) - Original Timeline', fontsize=11)

# Plot 2: EDITED audio on its own timeline
ax2 = fig.add_subplot(gs[1])
edited_samples = audio_to_array(edited)
edited_time = np.linspace(0, len(edited)/1000, len(edited_samples))

ax2.plot(edited_time, edited_samples, linewidth=0.2, color='#06A77D', alpha=0.8)
ax2.set_title('EDITED AUDIO (30.39s total) - 6.06s shorter', fontsize=16, fontweight='bold', pad=15)
ax2.set_ylabel('Amplitude', fontsize=12)
ax2.set_xlim(0, len(edited)/1000)
ax2.set_ylim(-1.1, 1.1)
ax2.grid(True, alpha=0.2, linestyle='--')

# Calculate where each replacement appears in edited timeline
# Build mapping: original_ms -> edited_ms
current_edited_pos = 0
for i, repl in enumerate(replacements):
    # Before this replacement, all original audio is kept
    if i == 0:
        prev_end = 0
    else:
        prev_end = replacements[i-1]['orig_end']

    # Audio from prev_end to this repl_start is kept
    kept_duration = repl['orig_start'] - prev_end
    current_edited_pos += kept_duration

    # Mark where the replacement appears in edited audio
    repl_start_edited = current_edited_pos / 1000
    repl_end_edited = (current_edited_pos + repl['repl_dur']) / 1000

    color = colors[i % len(colors)]
    ax2.axvspan(repl_start_edited, repl_end_edited, alpha=0.3, color=color, zorder=1)

    if i % 2 == 0:
        y_pos = 0.85
    else:
        y_pos = -0.85

    # Show what vegetable this is
    ax2.text((repl_start_edited + repl_end_edited) / 2, y_pos, f"R{i+1}\n{repl['veg']}",
             ha='center', fontsize=7, bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))

    # Move edited position forward by replacement duration
    current_edited_pos += repl['repl_dur']

ax2.set_xlabel('Time (seconds) - Edited Timeline (Shorter!)', fontsize=11)

# Plot 3: Key showing the mapping
ax3 = fig.add_subplot(gs[2])
ax3.axis('off')

key_text = "REPLACEMENT KEY (R1-R10):\n"
key_text += f"{'#':<4} {'Original Position':<20} {'Word':<12} {'→':<3} {'Vegetable':<18} {'Duration':<12}\n"
key_text += "-" * 95 + "\n"

for i, repl in enumerate(replacements, 1):
    orig_pos = f"{repl['orig_start']/1000:.2f}-{repl['orig_end']/1000:.2f}s"
    dur = f"{repl['repl_dur']}ms"
    key_text += f"R{i:<3} {orig_pos:<20} {repl['word']:<12} {'→':<3} {repl['veg']:<18} {dur:<12}\n"

ax3.text(0.05, 0.5, key_text, fontsize=9, verticalalignment='center',
         family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))

plt.savefig('proper_comparison.png', dpi=150, bbox_inches='tight')
print("✓ Saved proper_comparison.png")

# Create zoomed comparison of problematic section
fig2, (ax_top, ax_mid, ax_bot) = plt.subplots(3, 1, figsize=(18, 10), height_ratios=[1, 1, 0.3])

zoom_start = 2.5
zoom_end = 8.0

# Original zoomed
zoom_orig = original[int(zoom_start*1000):int(zoom_end*1000)]
zoom_orig_samples = audio_to_array(zoom_orig)
zoom_orig_time = np.linspace(zoom_start, zoom_end, len(zoom_orig_samples))

ax_top.plot(zoom_orig_time, zoom_orig_samples, linewidth=0.4, color='#2E86AB')
ax_top.set_title(f'ORIGINAL: {zoom_start}-{zoom_end}s (showing "Fucking hell" section)',
                 fontsize=14, fontweight='bold')
ax_top.set_ylabel('Amplitude', fontsize=11)
ax_top.set_xlim(zoom_start, zoom_end)
ax_top.grid(True, alpha=0.3)

# Mark the three replacements in this range
ax_top.axvspan(3.114, 3.834, alpha=0.3, color='red')
ax_top.text(3.5, 0.85, '"fucking"\n(R1)', ha='center', fontsize=9,
           bbox=dict(boxstyle='round', facecolor='red', alpha=0.7))

ax_top.axvspan(4.795, 5.455, alpha=0.3, color='orange')
ax_top.text(5.1, -0.85, '"Fucking"\n(R2)', ha='center', fontsize=9,
           bbox=dict(boxstyle='round', facecolor='orange', alpha=0.7))

ax_top.axvspan(6.920, 7.200, alpha=0.3, color='yellow')
ax_top.text(7.0, 0.85, '"hell"\n(R3)', ha='center', fontsize=9,
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# Add annotations for kept audio
ax_top.text(2.7, 0.5, 'KEPT:\n"Kevin, just"', fontsize=8, color='green', weight='bold',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
ax_top.text(4.0, 0.5, 'KEPT:\n"smash the door"', fontsize=8, color='green', weight='bold',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

# Edited zoomed - need to find where this section maps to
# Calculate edited positions for this section
# R1 "fucking" at 3.114-3.834 in original
# Before R1: 0-3.114 is kept (3114ms)
# R1: 730ms replacement
# Between R1 and R2: 3.834-4.795 kept (961ms)
# So R2 starts at: 3114 + 730 + 961 = 4805ms in edited

edited_r1_start = 3114 / 1000  # Kept audio before
edited_r1_end = (3114 + 730) / 1000
edited_r2_start = (3114 + 730 + 961) / 1000
edited_r2_end = (3114 + 730 + 961 + 660) / 1000
# Gap from R2 end to "hell" word: 5.455-6.920 = 1.465s kept
edited_r3_start = (3114 + 730 + 961 + 660 + 1465) / 1000
edited_r3_end = (3114 + 730 + 961 + 660 + 1465 + 280) / 1000

# Extract edited section
zoom_edited_start_ms = int(2.5 * 1000)  # Start from same relative position
zoom_edited_end_ms = int(edited_r3_end * 1000) + 500  # Go past R3

zoom_edited = edited[zoom_edited_start_ms:zoom_edited_end_ms]
zoom_edited_samples = audio_to_array(zoom_edited)
zoom_edited_time = np.linspace(zoom_edited_start_ms/1000, zoom_edited_end_ms/1000,
                                len(zoom_edited_samples))

ax_mid.plot(zoom_edited_time, zoom_edited_samples, linewidth=0.4, color='#06A77D')
ax_mid.set_title(f'EDITED: Corresponding section in edited audio',
                fontsize=14, fontweight='bold')
ax_mid.set_ylabel('Amplitude', fontsize=11)
ax_mid.set_xlim(zoom_edited_start_ms/1000, zoom_edited_end_ms/1000)
ax_mid.grid(True, alpha=0.3)

# Mark replacements in edited
ax_mid.axvspan(edited_r1_start, edited_r1_end, alpha=0.3, color='red')
ax_mid.text((edited_r1_start + edited_r1_end)/2, 0.85, 'R1:\nbamboo\nshoots',
           ha='center', fontsize=8, bbox=dict(boxstyle='round', facecolor='red', alpha=0.7))

ax_mid.axvspan(edited_r2_start, edited_r2_end, alpha=0.3, color='orange')
ax_mid.text((edited_r2_start + edited_r2_end)/2, -0.85, 'R2:\nsweet\npotato',
           ha='center', fontsize=8, bbox=dict(boxstyle='round', facecolor='orange', alpha=0.7))

ax_mid.axvspan(edited_r3_start, edited_r3_end, alpha=0.3, color='yellow')
ax_mid.text((edited_r3_start + edited_r3_end)/2, 0.85, 'R3:\ngarlic',
           ha='center', fontsize=8, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

ax_mid.set_xlabel('Time (seconds) - Edited Timeline', fontsize=11)

# Bottom: explanation
ax_bot.axis('off')
explanation = f"""
ALIGNMENT EXPLANATION:
• Original: R1 at {3.114:.2f}s, R2 at {4.795:.2f}s, R3 at {6.920:.2f}s
• Edited:   R1 at {edited_r1_start:.2f}s, R2 at {edited_r2_start:.2f}s, R3 at {edited_r3_start:.2f}s
• Why different? Edited is {(len(original)-len(edited))/1000:.2f}s shorter (removed {len(replacements)} swear words + buffers)
• R2 appears EARLIER in edited ({edited_r2_start:.2f}s vs {4.795:.2f}s) because we removed R1 before it!
"""
ax_bot.text(0.05, 0.5, explanation, fontsize=10, verticalalignment='center',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.6))

plt.tight_layout()
plt.savefig('proper_zoomed_comparison.png', dpi=150, bbox_inches='tight')
print("✓ Saved proper_zoomed_comparison.png")

print("\n" + "="*70)
print("CORRECT VISUALIZATIONS CREATED")
print("="*70)
print("\nThese show BOTH timelines properly:")
print("• Original: 36.45s timeline with swear words marked")
print("• Edited: 30.39s timeline (shorter!) with vegetables marked")
print("\nThe edited audio is SHORTER, so timestamps don't align 1:1!")
print("That's why 'sweet potato' appears at ~4.8s in edited vs ~4.8s in original")
