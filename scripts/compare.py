"""
Unified comparison tool for audio analysis and visualization.
Replaces: create_aligned_comparison.py, create_proper_comparison.py,
          create_wordlevel_comparison.py, enhanced_waveform_comparison.py
"""
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment
import sys
import os

# Add parent directory to path to import timeline_processor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def audio_to_array(segment):
    """Convert AudioSegment to normalized numpy array."""
    samples = np.array(segment.get_array_of_samples(), dtype=np.float32)
    if segment.channels == 2:
        samples = samples.reshape((-1, 2)).mean(axis=1)
    if np.std(samples) > 0:
        samples = samples / np.max(np.abs(samples))
    return samples


def create_waveform_comparison(original_path, edited_path, output_path,
                                replacements=None, zoom_start=None, zoom_end=None):
    """
    Create side-by-side waveform comparison.

    Args:
        original_path: Path to original audio
        edited_path: Path to edited audio
        output_path: Where to save the visualization
        replacements: List of replacement dicts with start/end times
        zoom_start: Optional start time (seconds) for zoomed view
        zoom_end: Optional end time (seconds) for zoomed view
    """
    original = AudioSegment.from_file(original_path)
    edited = AudioSegment.from_file(edited_path)

    # Apply zoom if requested
    if zoom_start is not None and zoom_end is not None:
        original = original[int(zoom_start*1000):int(zoom_end*1000)]
        edited = edited[int(zoom_start*1000):int(zoom_end*1000)]
        time_offset = zoom_start
        title_suffix = f" ({zoom_start:.1f}-{zoom_end:.1f}s)"
    else:
        time_offset = 0
        title_suffix = ""

    # Convert to arrays
    orig_samples = audio_to_array(original)
    edited_samples = audio_to_array(edited)
    orig_time = np.linspace(time_offset, time_offset + len(original)/1000, len(orig_samples))
    edited_time = np.linspace(time_offset, time_offset + len(edited)/1000, len(edited_samples))

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10), sharex=True)

    # Plot original
    ax1.plot(orig_time, orig_samples, linewidth=0.3, color='#2E86AB', alpha=0.8)
    ax1.set_title(f'Original Audio{title_suffix}', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Amplitude', fontsize=11)
    ax1.set_ylim(-1.1, 1.1)
    ax1.grid(True, alpha=0.2, linestyle='--')

    # Mark replacements on original
    if replacements:
        colors = ['#FF6B6B', '#FF8E53', '#FFA07A', '#FFB84D', '#FFCB5C',
                  '#FFE66D', '#C7EFCF', '#A7E1AF', '#98D8C8', '#6FCDDD']
        for i, repl in enumerate(replacements):
            start_sec = repl['original_start'] / 1000
            end_sec = repl['original_end'] / 1000

            # Only show if in zoom range
            if zoom_start is None or (start_sec >= zoom_start and end_sec <= zoom_end):
                color = colors[i % len(colors)]
                ax1.axvspan(start_sec, end_sec, alpha=0.3, color=color, zorder=1)

                y_pos = 0.85 if i % 2 == 0 else -0.85
                ax1.text((start_sec + end_sec) / 2, y_pos, f"R{i+1}",
                        ha='center', fontsize=9,
                        bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))

    # Plot edited
    ax2.plot(edited_time, edited_samples, linewidth=0.3, color='#06A77D', alpha=0.8)
    ax2.set_title(f'Edited Audio{title_suffix}', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Amplitude', fontsize=11)
    ax2.set_xlabel('Time (seconds)', fontsize=11)
    ax2.set_ylim(-1.1, 1.1)
    ax2.grid(True, alpha=0.2, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved {output_path}")


def create_transcript_comparison(original_transcript_path, edited_transcript_path,
                                 output_path, replacements=None):
    """Create side-by-side transcript comparison."""
    with open(original_transcript_path, 'r') as f:
        orig_lines = f.readlines()

    with open(edited_transcript_path, 'r') as f:
        edited_lines = f.readlines()

    with open(output_path, 'w') as f:
        f.write("TRANSCRIPT COMPARISON\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"{'Time':<15} {'Original':<50} {'Edited':<50}\n")
        f.write("-" * 100 + "\n\n")

        # Parse and compare segments
        for orig_line, edited_line in zip(orig_lines, edited_lines):
            if "[" in orig_line and "]" in orig_line:
                f.write(f"{orig_line.strip():<65} → {edited_line.strip()}\n")

    print(f"✓ Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Unified audio comparison tool')
    parser.add_argument('--original', required=True, help='Original audio file')
    parser.add_argument('--edited', required=True, help='Edited audio file')
    parser.add_argument('--output', default='../outputs/comparison.png',
                       help='Output visualization path')
    parser.add_argument('--zoom-start', type=float, help='Zoom start time (seconds)')
    parser.add_argument('--zoom-end', type=float, help='Zoom end time (seconds)')
    parser.add_argument('--transcript-original', help='Original transcript file')
    parser.add_argument('--transcript-edited', help='Edited transcript file')
    parser.add_argument('--transcript-output', default='../outputs/transcript_comparison.txt',
                       help='Transcript comparison output')

    args = parser.parse_args()

    # Create waveform comparison
    create_waveform_comparison(
        args.original,
        args.edited,
        args.output,
        zoom_start=args.zoom_start,
        zoom_end=args.zoom_end
    )

    # Create transcript comparison if requested
    if args.transcript_original and args.transcript_edited:
        create_transcript_comparison(
            args.transcript_original,
            args.transcript_edited,
            args.transcript_output
        )


if __name__ == "__main__":
    main()
