"""
Test timeline processor with word-level timing data
"""
import json
from timeline_processor import process_audio_with_timeline
from pydub import AudioSegment

# Create segments with word-level timing based on the original transcript
# (reconstructed from our earlier debugging)
segments = [
    {
        "start": 1.53,
        "end": 1.99,
        "text": "What was that?",
        "words": [
            {"word": "What", "start": 1.53, "end": 1.65},
            {"word": "was", "start": 1.67, "end": 1.80},
            {"word": "that", "start": 1.82, "end": 1.99}
        ]
    },
    {
        "start": 2.53,
        "end": 4.78,
        "text": "Kevin, just fucking smash the door, that's not acceptable.",
        "words": [
            {"word": "Kevin", "start": 2.53, "end": 2.80},
            {"word": "just", "start": 2.90, "end": 3.11},
            {"word": "fucking", "start": 3.114, "end": 3.434},
            {"word": "smash", "start": 3.50, "end": 3.75},
            {"word": "the", "start": 3.77, "end": 3.85},
            {"word": "door", "start": 3.87, "end": 4.10},
            {"word": "that's", "start": 4.20, "end": 4.40},
            {"word": "not", "start": 4.42, "end": 4.52},
            {"word": "acceptable", "start": 4.54, "end": 4.78}
        ]
    },
    {
        "start": 4.79,
        "end": 7.20,
        "text": "Fucking hell.",
        "words": [
            {"word": "Fucking", "start": 4.795, "end": 5.055},
            {"word": "hell", "start": 6.92, "end": 7.20}
        ]
    },
    {
        "start": 9.66,
        "end": 11.46,
        "text": "He's not fucking doing that to me, you know?",
        "words": [
            {"word": "He's", "start": 9.66, "end": 9.85},
            {"word": "not", "start": 9.87, "end": 10.00},
            {"word": "fucking", "start": 10.098, "end": 10.398},
            {"word": "doing", "start": 10.50, "end": 10.75},
            {"word": "that", "start": 10.77, "end": 10.90},
            {"word": "to", "start": 10.92, "end": 10.98},
            {"word": "me", "start": 11.00, "end": 11.15},
            {"word": "you", "start": 11.20, "end": 11.32},
            {"word": "know", "start": 11.34, "end": 11.46}
        ]
    },
    {
        "start": 12.64,
        "end": 14.78,
        "text": "He does not fuck smash my door, tell him that.",
        "words": [
            {"word": "He", "start": 12.64, "end": 12.75},
            {"word": "does", "start": 12.77, "end": 12.95},
            {"word": "not", "start": 12.97, "end": 13.15},
            {"word": "fuck", "start": 13.399, "end": 13.639},
            {"word": "smash", "start": 13.75, "end": 13.98},
            {"word": "my", "start": 14.00, "end": 14.12},
            {"word": "door", "start": 14.14, "end": 14.35},
            {"word": "tell", "start": 14.40, "end": 14.55},
            {"word": "him", "start": 14.57, "end": 14.68},
            {"word": "that", "start": 14.70, "end": 14.78}
        ]
    },
    {
        "start": 15.10,
        "end": 17.08,
        "text": "If he doesn't want to come back, he better tell him now.",
        "words": []  # No profanity, simplified
    },
    {
        "start": 18.38,
        "end": 19.88,
        "text": "He smash my fucking office door.",
        "words": [
            {"word": "He", "start": 18.38, "end": 18.50},
            {"word": "smash", "start": 18.52, "end": 18.75},
            {"word": "my", "start": 18.77, "end": 18.88},
            {"word": "fucking", "start": 19.062, "end": 19.322},
            {"word": "office", "start": 19.40, "end": 19.60},
            {"word": "door", "start": 19.65, "end": 19.88}
        ]
    },
    {
        "start": 21.00,
        "end": 22.74,
        "text": "I don't know where he is, but he can fuck off.",
        "words": [
            {"word": "I", "start": 21.00, "end": 21.08},
            {"word": "don't", "start": 21.10, "end": 21.30},
            {"word": "know", "start": 21.32, "end": 21.50},
            {"word": "where", "start": 21.52, "end": 21.70},
            {"word": "he", "start": 21.72, "end": 21.80},
            {"word": "is", "start": 21.82, "end": 21.95},
            {"word": "but", "start": 22.00, "end": 22.15},
            {"word": "he", "start": 22.17, "end": 22.25},
            {"word": "can", "start": 22.27, "end": 22.34},
            {"word": "fuck", "start": 22.344, "end": 22.564},
            {"word": "off", "start": 22.60, "end": 22.74}
        ]
    },
    {
        "start": 24.93,
        "end": 25.43,
        "text": "Fucking hell.",
        "words": [
            {"word": "Fucking", "start": 24.925, "end": 25.186},
            {"word": "hell", "start": 25.30, "end": 25.43}
        ]
    },
    {
        "start": 26.56,
        "end": 28.52,
        "text": "We have got too fucking idiot striving for us.",
        "words": [
            {"word": "We", "start": 26.56, "end": 26.68},
            {"word": "have", "start": 26.70, "end": 26.85},
            {"word": "got", "start": 26.87, "end": 27.00},
            {"word": "too", "start": 27.02, "end": 27.15},
            {"word": "fucking", "start": 27.158, "end": 27.479},
            {"word": "idiot", "start": 27.55, "end": 27.80},
            {"word": "striving", "start": 27.85, "end": 28.20},
            {"word": "for", "start": 28.25, "end": 28.38},
            {"word": "us", "start": 28.40, "end": 28.52}
        ]
    }
]

# Load swear words
with open("word_lists.json", "r") as f:
    data = json.load(f)
    swear_words = data['swear_words']

print("Testing timeline-based processor with WORD-LEVEL timing...")
print(f"Segments: {len(segments)}")
print(f"Swear words: {len(swear_words)}")
print()

# Process audio
output_path, replacement_log = process_audio_with_timeline(
    audio_path="audio/5h4FDy9Eqzs.wav",
    segments=segments,
    swear_words=swear_words,
    output_path="edited_audio_wordlevel.wav",
    debug=True
)

print("\n" + "="*70)
print("REPLACEMENT LOG (Word-Level)")
print("="*70)

with open("replacements_log_wordlevel.txt", "w") as f:
    f.write("Word-Level Timeline Replacement Log\n")
    f.write("Format: [Original Start-End] Original Text -> Vegetables (Durations)\n\n")

    for i, repl in enumerate(replacement_log, 1):
        veg_list = ", ".join(repl['vegetables'])
        log_line = (f"[{repl['original_start']}-{repl['original_end']}ms] "
                   f"\"{repl['original_text']}\" -> {veg_list} "
                   f"(Original: {repl['original_duration']}ms, "
                   f"Replacement: {repl['replacement_duration']}ms)")
        f.write(log_line + "\n")
        print(f"{i}. {log_line}")

print(f"\n✓ Log saved to replacements_log_wordlevel.txt")
print(f"✓ Audio saved to {output_path}")

# Compare
original = AudioSegment.from_file("audio/5h4FDy9Eqzs.wav")
edited = AudioSegment.from_file(output_path)

print(f"\nComparison:")
print(f"  Original: {len(original)}ms ({len(original)/1000:.2f}s)")
print(f"  Edited:   {len(edited)}ms ({len(edited)/1000:.2f}s)")
print(f"  Saved:    {len(original) - len(edited)}ms ({(len(original)-len(edited))/1000:.2f}s)")
print(f"  Reduction: {100 * (len(original) - len(edited)) / len(original):.1f}%")

print("\n" + "="*70)
print("KEY DIFFERENCE:")
print("="*70)
print("This version replaces ONLY the swear words (+ 400ms safety buffer)")
print("All other words in the sentence are kept!")
print("\nExample:")
print("  'Kevin, just FUCKING smash the door' →")
print("  'Kevin, just [vegetables] smash the door'")
print("\nNOT:")
print("  '[vegetables]' (which is what segment-level does)")
