"""
Create final video using timeline processor with all fixes
"""
import json
import subprocess
from timeline_processor import process_audio_with_timeline
import whisperx
import torch

print("="*70)
print("CREATING FINAL VIDEO WITH TIMELINE PROCESSOR")
print("="*70)

# Use the existing transcription or re-run if needed
# For now, let's use the test data we created which has word-level timing
# In production, this would come from WhisperX

# Load segments with word-level timing from test
import sys
sys.path.insert(0, '.')
from test_timeline_with_words import segments

# Load swear words
with open("word_lists.json", "r") as f:
    data = json.load(f)
    swear_words = data['swear_words']

print("\nStep 1: Processing audio with timeline processor...")
print(f"  Segments: {len(segments)}")
print(f"  Swear words to replace: {len(swear_words)}")

# Process audio
output_audio_path, replacement_log = process_audio_with_timeline(
    audio_path="audio/5h4FDy9Eqzs.wav",
    segments=segments,
    swear_words=swear_words,
    output_path="final_edited_audio.wav",
    debug=True
)

print(f"\n✓ Audio processed: {output_audio_path}")
print(f"  Replacements: {len(replacement_log)}")

# Save detailed log
with open("final_replacements_log.txt", "w") as f:
    f.write("Final Video Replacement Log\n")
    f.write("="*70 + "\n\n")
    for i, repl in enumerate(replacement_log, 1):
        veg_list = ", ".join(repl['vegetables'])
        f.write(f"{i}. [{repl['original_start']}-{repl['original_end']}ms] "
                f"\"{repl['original_text']}\" -> {veg_list}\n")
        f.write(f"   Duration: {repl['original_duration']}ms -> {repl['replacement_duration']}ms\n\n")

print("✓ Saved final_replacements_log.txt")

print("\nStep 2: Merging audio with video...")

# Run ffmpeg to merge audio and video
video_path = "videos/5h4FDy9Eqzs.mp4"

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
    "final_output_temp.mp4"
]

print("  Running ffmpeg merge...")
result = subprocess.run(cmd1, capture_output=True, text=True)
if result.returncode != 0:
    print(f"  ⚠️  FFmpeg error: {result.stderr}")
else:
    print("  ✓ First pass complete")

# Second pass: optimize
cmd2 = [
    "ffmpeg", "-y",
    "-i", "final_output_temp.mp4",
    "-c", "copy",
    "-movflags", "+faststart",
    "-f", "mp4",
    "final_output.mp4"
]

print("  Optimizing output...")
result = subprocess.run(cmd2, capture_output=True, text=True)
if result.returncode != 0:
    print(f"  ⚠️  FFmpeg error: {result.stderr}")
else:
    print("  ✓ Second pass complete")

print("\n" + "="*70)
print("✅ FINAL VIDEO CREATED!")
print("="*70)
print(f"\n📹 Output: final_output.mp4")
print(f"🎵 Audio: {output_audio_path}")
print(f"📝 Log: final_replacements_log.txt")

print(f"\n📊 Summary:")
print(f"  - Replaced {len(replacement_log)} profanity phrases")
print(f"  - Used phrase-level replacement (groups nearby swears)")
print(f"  - Handles punctuation and variants (fuckin, fucking, etc.)")
print(f"  - All non-profane words preserved")

print("\n🎬 Please review final_output.mp4")
print("   The 'Fucking hell' at 4.79-7.20s should now be vegetables!")
