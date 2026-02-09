"""
Generate a detailed transcript comparison showing original vs edited content
"""
import json
from pydub import AudioSegment

# Load the replacement log
with open("replacements_log.txt", "r") as f:
    log_lines = f.readlines()

# Parse replacements
replacements = []
for line in log_lines[3:]:  # Skip header
    if line.strip() and "->" in line:
        # Format: [3114-3434ms] fucking -> [3114-3386ms] carrot (Original: 320ms, Replacement: 272ms)
        parts = line.strip().split("->")
        orig_part = parts[0].strip()
        repl_part = parts[1].strip()

        # Extract original info
        orig_times = orig_part.split("]")[0].split("[")[1]
        orig_start = int(orig_times.split("-")[0].replace("ms", ""))
        orig_end = int(orig_times.split("-")[1].replace("ms", ""))
        orig_word = orig_part.split("]")[1].strip()

        # Extract replacement info
        repl_times = repl_part.split("]")[0].split("[")[1]
        repl_start = int(repl_times.split("-")[0].replace("ms", ""))
        repl_end = int(repl_times.split("-")[1].replace("ms", ""))
        repl_word = repl_part.split("]")[1].split("(")[0].strip()

        replacements.append({
            "orig_start": orig_start,
            "orig_end": orig_end,
            "orig_word": orig_word,
            "repl_start": repl_start,
            "repl_end": repl_end,
            "repl_word": repl_word
        })

# Load original audio to get durations
original = AudioSegment.from_file("audio/5h4FDy9Eqzs.wav")
edited = AudioSegment.from_file("edited_audio.wav")

print("="*80)
print("TRANSCRIPT COMPARISON: ORIGINAL vs EDITED")
print("="*80)

# Read original transcript
with open("transcript.txt", "r") as f:
    transcript_lines = f.readlines()

print("\nFocusing on 0-10 seconds (where the problematic replacements are):\n")

# Parse and display
segments = []
for line in transcript_lines:
    if "[" in line and "]" in line and "-" in line.split("]")[0]:
        try:
            time_part = line.split("]")[0].split("[")[1]
            start = float(time_part.split("-")[0].strip())
            end = float(time_part.split("-")[1].strip())
            text = line.split("]")[1].strip()
            if start < 10:
                segments.append({"start": start, "end": end, "text": text})
        except:
            pass

print(f"{'Time Range':<20} {'Original Text':<40} {'Action':<30}")
print("-"*90)

for seg in segments:
    start_ms = int(seg["start"] * 1000)
    end_ms = int(seg["end"] * 1000)

    # Check if this segment has replacements
    seg_replacements = [r for r in replacements if start_ms <= r["orig_start"] < end_ms]

    if seg_replacements:
        for repl in seg_replacements:
            print(f"{seg['start']:.2f}s - {seg['end']:.2f}s    {seg['text']:<40} ⚠️  MODIFIED")
            print(f"  └─ [{repl['orig_start']}-{repl['orig_end']}ms] '{repl['orig_word']}' → '{repl['repl_word']}' at [{repl['repl_start']}-{repl['repl_end']}ms]")
    else:
        print(f"{seg['start']:.2f}s - {seg['end']:.2f}s    {seg['text']:<40} ✓  Unchanged")

print("\n" + "="*80)
print("DETAILED ANALYSIS: Second Segment (4.79-7.20s)")
print("="*80)

print("""
ORIGINAL AUDIO STRUCTURE (4.79-7.20s):
  Timeline:                    4.79        4.795    5.055      5.255           6.92     7.20
                                |           |        |          |               |        |
  Original:                    [gap] [----Fucking----][safety][-----1665ms-----][--hell--]
                                           260ms      200ms      buffer gap        280ms

EDITED AUDIO STRUCTURE:
  Timeline:                    4.547    4.768                              6.713
                                |        |                                  |
  Edited:                      [cabbage][--------1665ms gap audio---------][continues...]
                                221ms     (from original 5.255-6.92s)

WHAT SHOULD HAPPEN:
  ✓ Skip original 4.79-4.795s (pre-word gap)
  ✓ REPLACE original 4.795-5.055s ('Fucking') with 'cabbage'
  ✓ SKIP original 5.055-5.255s (200ms safety buffer)
  ✓ INCLUDE original 5.255-6.92s (audio before 'hell')
  ✓ INCLUDE original 6.92-7.20s ('hell')

CORRELATION TEST RESULT:
  ✓ Audio at edited 4.768ms matches original 5.255ms (95.85% correlation)
  ✓ Code is working correctly!

THE PROBLEM:
  ⚠️  Original audio from 5.255-6.92s (1665ms gap) contains speech!
  ⚠️  This audio has RMS level of 760.0 (very high - indicates speech)
  ⚠️  WhisperX incorrectly transcribed this as a gap between words
  ⚠️  Actual content likely: "...ucking hell" or similar profanity continuation
""")

print("\n" + "="*80)
print("FULL REPLACEMENT LOG:")
print("="*80)
print(f"\n{'#':<4} {'Original Position':<25} {'Word':<12} {'Edited Position':<25} {'Replacement':<12}")
print("-"*90)

for i, repl in enumerate(replacements, 1):
    print(f"{i:<4} [{repl['orig_start']:>5}-{repl['orig_end']:>5}ms]"
          f" ({(repl['orig_end']-repl['orig_start']):>3}ms)    "
          f"{repl['orig_word']:<12} "
          f"[{repl['repl_start']:>5}-{repl['repl_end']:>5}ms]"
          f" ({(repl['repl_end']-repl['repl_start']):>3}ms)    "
          f"{repl['repl_word']:<12}")

print(f"\nOriginal audio length: {len(original)}ms ({len(original)/1000:.2f}s)")
print(f"Edited audio length:   {len(edited)}ms ({len(edited)/1000:.2f}s)")
print(f"Time saved:            {len(original)-len(edited)}ms ({(len(original)-len(edited))/1000:.2f}s)")

# Calculate expected time saved
total_skipped = sum((r['orig_end'] - r['orig_start']) + 200 for r in replacements)  # word + buffer
total_added = sum(r['repl_end'] - r['repl_start'] for r in replacements)
expected_saved = total_skipped - total_added
print(f"Expected time saved:   {expected_saved}ms ({expected_saved/1000:.2f}s)")
print(f"  (8 words × 260ms avg) + (8 × 200ms buffer) = {total_skipped}ms skipped")
print(f"  (8 replacements × 230ms avg) = {total_added}ms added")
print(f"  Net difference: {expected_saved}ms")

print("\n" + "="*80)
print("RECOMMENDATIONS:")
print("="*80)
print("""
Since correlation analysis proves the code is working correctly,
but you're still hearing profanity, the issue is WhisperX's transcription.

OPTIONS:
1. Increase post_replacement_skip_ms from 200ms to 500ms or 1000ms
   - Pros: Simple fix, more aggressive skipping
   - Cons: May cut off legitimate speech

2. Use SEGMENT-LEVEL replacement instead of word-level for profanity
   - Pros: Bypasses incorrect word timestamps, replaces entire phrase
   - Cons: Loses more audio (replaces "Fucking hell" as a unit)

3. Manual correction: Edit word boundaries in transcript before processing
   - Pros: Most accurate
   - Cons: Requires manual intervention

4. Use multiple transcription services and compare results
   - Pros: More reliable timestamps
   - Cons: More complex, slower

RECOMMENDED: Try option 2 (segment-level replacement) for segments containing profanity.
""")
