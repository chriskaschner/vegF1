"""
Unified video creation tool.
Processes audio with timeline processor and merges with video.
Replaces: create_final_video.py
"""
import argparse
import json
import subprocess
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from timeline_processor import process_audio_with_timeline


def create_video(
    video_path,
    audio_path,
    segments,
    swear_words,
    output_path="final_output.mp4",
    debug=False
):
    """
    Process audio and merge with video.

    Args:
        video_path: Path to original video
        audio_path: Path to original audio
        segments: Transcription segments with word-level timing
        swear_words: List of swear words to replace
        output_path: Output video path
        debug: Enable debug output

    Returns:
        Path to created video
    """
    print("="*70)
    print("CREATING VIDEO WITH TIMELINE PROCESSOR")
    print("="*70)

    # Process audio
    print("\nStep 1: Processing audio with timeline processor...")
    print(f"  Segments: {len(segments)}")
    print(f"  Swear words to replace: {len(swear_words)}")

    edited_audio_path = output_path.replace(".mp4", "_audio.wav")

    output_audio_path, replacement_log = process_audio_with_timeline(
        audio_path=audio_path,
        segments=segments,
        swear_words=swear_words,
        output_path=edited_audio_path,
        debug=debug
    )

    print(f"\n✓ Audio processed: {output_audio_path}")
    print(f"  Replacements: {len(replacement_log)}")

    # Save replacement log
    log_path = output_path.replace(".mp4", "_replacements.log")
    with open(log_path, "w") as f:
        f.write("Video Replacement Log\n")
        f.write("="*70 + "\n\n")
        for i, repl in enumerate(replacement_log, 1):
            veg_list = ", ".join(repl['vegetables'])
            f.write(f"{i}. [{repl['original_start']}-{repl['original_end']}ms] "
                   f"\"{repl['original_text']}\" -> {veg_list}\n")
            f.write(f"   Duration: {repl['original_duration']}ms -> "
                   f"{repl['replacement_duration']}ms\n\n")

    print(f"✓ Saved {log_path}")

    # Merge audio and video
    print("\nStep 2: Merging audio with video...")

    temp_output = output_path.replace(".mp4", "_temp.mp4")

    # First pass: merge video and audio
    cmd1 = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", output_audio_path,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "h264",
        "-profile:v", "high",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-ac", "2",
        "-ar", "44100",
        "-shortest",
        "-movflags", "+faststart",
        temp_output
    ]

    print("  Running ffmpeg merge...")
    result = subprocess.run(cmd1, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ⚠️  FFmpeg error: {result.stderr}")
        return None
    print("  ✓ First pass complete")

    # Second pass: optimize
    cmd2 = [
        "ffmpeg", "-y",
        "-i", temp_output,
        "-c", "copy",
        "-movflags", "+faststart",
        "-f", "mp4",
        output_path
    ]

    print("  Optimizing output...")
    result = subprocess.run(cmd2, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ⚠️  FFmpeg error: {result.stderr}")
        return None
    print("  ✓ Second pass complete")

    # Cleanup temp file
    Path(temp_output).unlink(missing_ok=True)

    print("\n" + "="*70)
    print("✅ VIDEO CREATED!")
    print("="*70)
    print(f"\n📹 Output: {output_path}")
    print(f"🎵 Audio: {output_audio_path}")
    print(f"📝 Log: {log_path}")

    print(f"\n📊 Summary:")
    print(f"  - Replaced {len(replacement_log)} profanity phrases")
    print(f"  - Used phrase-level replacement (groups nearby swears)")
    print(f"  - Handles punctuation and variants")
    print(f"  - All non-profane words preserved")

    return output_path


def main():
    parser = argparse.ArgumentParser(description='Create video with censored audio')
    parser.add_argument('--video', required=True, help='Input video file')
    parser.add_argument('--audio', required=True, help='Input audio file')
    parser.add_argument('--segments', required=True, help='Segments JSON file')
    parser.add_argument('--swear-words', default='../word_lists.json',
                       help='Swear words JSON')
    parser.add_argument('--output', default='../outputs/final_output.mp4',
                       help='Output video path')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')

    args = parser.parse_args()

    # Load segments and swear words
    with open(args.segments, 'r') as f:
        segments = json.load(f)

    with open(args.swear_words, 'r') as f:
        swear_words = json.load(f)['swear_words']

    # Create video
    result = create_video(
        args.video,
        args.audio,
        segments,
        swear_words,
        args.output,
        args.debug
    )

    sys.exit(0 if result else 1)


if __name__ == "__main__":
    main()
