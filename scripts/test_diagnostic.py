"""
Diagnostic test to trace exactly what audio is being added.
"""
from pydub import AudioSegment
import json

# Load the actual audio and transcription
audio_file = "audio/5h4FDy9Eqzs.wav"
original_audio = AudioSegment.from_file(audio_file)

# Focus on the problematic segment
test_segment = {
    "start": 4.79,
    "end": 7.20,
    "text": "Fucking hell.",
    "words": [
        {"word": "Fucking", "start": 4.795, "end": 5.055},
        {"word": "hell", "start": 6.92, "end": 7.20}
    ]
}

print("=" * 70)
print("DIAGNOSTIC: Tracing audio additions for segment 'Fucking hell'")
print("=" * 70)
print(f"\nSegment: {test_segment['start']}-{test_segment['end']} ({test_segment['text']})")
print(f"Words:")
for w in test_segment['words']:
    print(f"  - '{w['word']}' at {w['start']}-{w['end']}")

# Simulate processing
last_end = 4780  # End of previous segment
output_audio = AudioSegment.silent(duration=4780)  # Simulate output up to previous segment
swear_words = ["fucking"]

print(f"\n{'Step':<6} {'Action':<40} {'Original Range':<20} {'Output Pos':<15} {'Duration':<10}")
print("-" * 95)

step = 1
for word_info in test_segment["words"]:
    word = word_info["word"].lower()
    word_start_ms = int(word_info["start"] * 1000)
    word_end_ms = int(word_info["end"] * 1000)

    if word in swear_words:
        # Step 1: Add audio before replacement
        replacement_start = word_start_ms
        if replacement_start > last_end:
            gap_audio = original_audio[last_end:replacement_start]
            output_before = len(output_audio)
            output_audio += gap_audio
            output_after = len(output_audio)
            print(f"{step:<6} {'Add original (before replacement)':<40} {f'{last_end}-{replacement_start}ms':<20} {f'{output_before}-{output_after}ms':<15} {len(gap_audio)}ms")
            step += 1

        # Step 2: Add replacement
        replacement_duration = 221  # Simulated
        replacement_audio = AudioSegment.silent(duration=replacement_duration)
        output_before = len(output_audio)
        output_audio += replacement_audio
        output_after = len(output_audio)
        print(f"{step:<6} {'Add REPLACEMENT (turnip)':<40} {f'[replaces {word_start_ms}-{word_end_ms}]':<20} {f'{output_before}-{output_after}ms':<15} {replacement_duration}ms")
        step += 1

        # Update last_end to skip original swear word
        print(f"{step:<6} {'Set last_end (skip original swear)':<40} {f'-> {word_end_ms}ms':<20} {'':<15}")
        last_end = word_end_ms
        step += 1
    else:
        print(f"{step:<6} {f'Skip non-swear: {word}':<40} {f'{word_start_ms}-{word_end_ms}ms':<20} {'':<15} {word_end_ms - word_start_ms}ms")
        step += 1

# Step 3: Add remaining audio after all words
end_ms = int(test_segment["end"] * 1000)
if end_ms > last_end:
    remaining_audio = original_audio[last_end:end_ms]
    output_before = len(output_audio)
    output_audio += remaining_audio
    output_after = len(output_audio)
    print(f"{step:<6} {'Add remaining (after all words)':<40} {f'{last_end}-{end_ms}ms':<20} {f'{output_before}-{output_after}ms':<15} {len(remaining_audio)}ms")

    # Check what's actually in this audio
    print(f"\n{'!':<6} {f'CRITICAL: Audio from {last_end}-{end_ms}ms contains:':<40}")
    print(f"{'!':<6}   - Word 'hell' at 6920-7200ms")
    print(f"{'!':<6}   - But also audio from {last_end}ms to 6920ms ({6920-last_end}ms)")
    print(f"{'!':<6}   - This gap audio might contain the tail end of 'fucking'!")

print("\n" + "=" * 70)
print("CONCLUSION:")
print("=" * 70)
print(f"Original swear word 'fucking': {word_start_ms}-{word_end_ms}ms (should be SKIPPED)")
print(f"Replacement 'turnip': Added to output")
print(f"last_end after replacement: {last_end}ms")
print(f"Audio added from {last_end}ms onwards: Contains everything after {last_end}ms")
print(f"\nThe original swear from {word_start_ms}-{word_end_ms}ms should NOT be in output.")
print(f"Next audio starts from {last_end}ms, which is AFTER the swear word ends.")
print("\nIF you're still hearing the swear word, possible causes:")
print("1. Transcription timestamps are wrong (swear extends past 5055ms)")
print("2. Video/audio desync during ffmpeg merge")
print("3. The edited_audio.wav is correct but video has wrong audio track")
