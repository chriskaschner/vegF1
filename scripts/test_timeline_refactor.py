"""
Test the timeline-based refactor and generate aligned comparisons
"""
import json
from timeline_processor import process_audio_with_timeline
from pydub import AudioSegment

# Load transcription
with open("transcript.txt", "r") as f:
    lines = f.readlines()

# Parse transcript into segments format
segments = []
for line in lines:
    if "[" in line and "]" in line and "-" in line.split("]")[0]:
        try:
            time_part = line.split("]")[0].split("[")[1]
            start = float(time_part.split("-")[0].strip())
            end = float(time_part.split("-")[1].strip())
            text = line.split("]")[1].strip()
            segments.append({"start": start, "end": end, "text": text})
        except:
            pass

# Load swear words
with open("word_lists.json", "r") as f:
    data = json.load(f)
    swear_words = data['swear_words']

print("Testing timeline-based audio processor...")
print(f"Segments: {len(segments)}")
print(f"Swear words: {len(swear_words)}")
print()

# Process audio
output_path, replacement_log = process_audio_with_timeline(
    audio_path="audio/5h4FDy9Eqzs.wav",
    segments=segments,
    swear_words=swear_words,
    output_path="edited_audio_timeline.wav",
    debug=True
)

# Save replacement log
print("\n" + "="*70)
print("REPLACEMENT LOG")
print("="*70)

with open("replacements_log_timeline.txt", "w") as f:
    f.write("Timeline-Based Replacement Log\n")
    f.write("Format: [Original Start-End] Original Text -> Vegetables (Durations)\n\n")

    for i, repl in enumerate(replacement_log, 1):
        veg_list = ", ".join(repl['vegetables'])
        log_line = (f"[{repl['original_start']}-{repl['original_end']}ms] "
                   f"{repl['original_text']} -> {veg_list} "
                   f"(Original: {repl['original_duration']}ms, "
                   f"Replacement: {repl['replacement_duration']}ms)")
        f.write(log_line + "\n")
        print(f"{i}. {log_line}")

print(f"\n✓ Log saved to replacements_log_timeline.txt")
print(f"✓ Audio saved to {output_path}")

# Load both audio files for comparison
original = AudioSegment.from_file("audio/5h4FDy9Eqzs.wav")
edited = AudioSegment.from_file(output_path)

print(f"\nComparison:")
print(f"  Original: {len(original)}ms ({len(original)/1000:.2f}s)")
print(f"  Edited:   {len(edited)}ms ({len(edited)/1000:.2f}s)")
print(f"  Saved:    {len(original) - len(edited)}ms ({(len(original)-len(edited))/1000:.2f}s)")
print(f"  Reduction: {100 * (len(original) - len(edited)) / len(original):.1f}%")
