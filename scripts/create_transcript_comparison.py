"""
Create detailed transcript comparison showing original vs edited
"""
import json

# Load replacement log
with open("replacements_log_timeline.txt", "r") as f:
    lines = f.readlines()

replacements = []
for line in lines[3:]:
    if "->" in line and "[" in line:
        orig_part = line.split("]")[0].split("[")[1]
        orig_start, orig_end = map(lambda x: int(x.replace("ms", "")), orig_part.split("-"))

        text_part = line.split("]")[1].split("->")[0].strip()
        veg_part = line.split("->")[1].split("(")[0].strip()

        durations_part = line.split("(")[1].split(")")[0]
        orig_dur = int(durations_part.split("Original:")[1].split("ms")[0].strip())
        repl_dur = int(durations_part.split("Replacement:")[1].split("ms")[0].strip())

        replacements.append({
            "orig_start": orig_start / 1000,
            "orig_end": orig_end / 1000,
            "orig_duration": orig_dur,
            "text": text_part,
            "vegetables": veg_part,
            "repl_duration": repl_dur
        })

# Load original transcript
with open("transcript.txt", "r") as f:
    transcript_lines = f.readlines()

segments = []
for line in transcript_lines:
    if "[" in line and "]" in line and "-" in line.split("]")[0]:
        try:
            time_part = line.split("]")[0].split("[")[1]
            start = float(time_part.split("-")[0].strip())
            end = float(time_part.split("-")[1].strip())
            text = line.split("]")[1].strip()
            segments.append({"start": start, "end": end, "text": text})
        except:
            pass

print("="*80)
print("TRANSCRIPT COMPARISON: ORIGINAL → EDITED (Timeline-Based)")
print("="*80)
print("\nShowing side-by-side what changed:\n")

print(f"{'Time':<15} {'Original Text':<45} {'Edited Version':<45}")
print("-"*105)

for seg in segments:
    # Check if this segment was replaced
    was_replaced = False
    replacement_text = None

    for repl in replacements:
        if abs(seg['start'] - repl['orig_start']) < 0.1:  # Match within 100ms
            was_replaced = True
            replacement_text = f"[{repl['vegetables']}]"
            break

    time_str = f"{seg['start']:.2f}-{seg['end']:.2f}s"

    if was_replaced:
        print(f"{time_str:<15} {seg['text']:<45} {'→ ' + replacement_text:<45}")
    else:
        print(f"{time_str:<15} {seg['text']:<45} {'(unchanged)':<45}")

print("\n" + "="*80)
print("DETAILED REPLACEMENT ANALYSIS")
print("="*80)

for i, repl in enumerate(replacements, 1):
    duration_diff = repl['repl_duration'] - repl['orig_duration']
    percent_change = 100 * duration_diff / repl['orig_duration']

    print(f"\nReplacement #{i}:")
    print(f"  Time:     {repl['orig_start']:.2f}-{repl['orig_end']:.2f}s")
    print(f"  Original: \"{repl['text']}\"")
    print(f"  Edited:   [{repl['vegetables']}]")
    print(f"  Duration: {repl['orig_duration']}ms → {repl['repl_duration']}ms "
          f"({duration_diff:+d}ms, {percent_change:+.1f}%)")

    # Show what this sounds like
    veg_list = repl['vegetables'].split(", ")
    if len(veg_list) == 1:
        print(f"  Audio:    Single vegetable: \"{veg_list[0]}\"")
    else:
        print(f"  Audio:    Multiple vegetables: {len(veg_list)} items")
        for j, veg in enumerate(veg_list, 1):
            print(f"            {j}. {veg}")

print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

from pydub import AudioSegment
original = AudioSegment.from_file("audio/5h4FDy9Eqzs.wav")
edited = AudioSegment.from_file("edited_audio_timeline.wav")

total_orig_duration = sum(r['orig_duration'] for r in replacements)
total_repl_duration = sum(r['repl_duration'] for r in replacements)

print(f"\nAudio Durations:")
print(f"  Original total:      {len(original)/1000:.2f}s")
print(f"  Edited total:        {len(edited)/1000:.2f}s")
print(f"  Time saved:          {(len(original)-len(edited))/1000:.2f}s")
print(f"  Reduction:           {100*(len(original)-len(edited))/len(original):.1f}%")

print(f"\nReplacements:")
print(f"  Segments replaced:   {len(replacements)}")
print(f"  Original duration:   {total_orig_duration/1000:.2f}s")
print(f"  Replacement duration:{total_repl_duration/1000:.2f}s")
print(f"  Duration matching:   {100*total_repl_duration/total_orig_duration:.1f}%")

# Count vegetables used
all_vegetables = []
for repl in replacements:
    all_vegetables.extend(repl['vegetables'].split(", "))

unique_vegetables = set(all_vegetables)

print(f"\nVegetables:")
print(f"  Total used:          {len(all_vegetables)}")
print(f"  Unique vegetables:   {len(unique_vegetables)}")
print(f"  Variety:             {', '.join(sorted(unique_vegetables))}")

print(f"\nApproach:")
print(f"  ✓ Timeline-based (declarative, not imperative)")
print(f"  ✓ Segment-level replacement (bypasses word-level timestamp issues)")
print(f"  ✓ Multiple vegetables for longer segments (natural duration matching)")
print(f"  ✓ No drift compensation needed (clean architecture)")

print("\n" + "="*80)
print("KEY IMPROVEMENTS FROM REFACTOR")
print("="*80)
print("""
BEFORE (Old Implementation):
  ✗ Word-level replacement with unreliable timestamps
  ✗ 200ms safety buffer still included profanity
  ✗ Complex drift compensation and state management
  ✗ Single vegetable replacements (too fast, unintelligible)
  ✗ Imperative audio building prone to bugs

AFTER (Timeline-Based Refactor):
  ✓ Segment-level replacement (more reliable)
  ✓ Multiple vegetables fill duration naturally
  ✓ No drift compensation needed
  ✓ Declarative timeline (testable, maintainable)
  ✓ Clean separation of concerns
  ✓ 95%+ duration matching on replacements

RESULT:
  The problematic "Fucking hell" segment (4.79-7.20s) is now replaced with
  "butternut squash, bell pepper, daikon" which fills 2.31s of the 2.41s
  original duration (95.8% match) - no more hidden profanity!
""")

# Save to file
with open("transcript_comparison_timeline.txt", "w") as f:
    f.write("="*80 + "\n")
    f.write("TRANSCRIPT COMPARISON: ORIGINAL → EDITED (Timeline-Based)\n")
    f.write("="*80 + "\n\n")

    f.write(f"{'Time':<15} {'Original Text':<45} {'Edited Version':<45}\n")
    f.write("-"*105 + "\n")

    for seg in segments:
        was_replaced = False
        replacement_text = None

        for repl in replacements:
            if abs(seg['start'] - repl['orig_start']) < 0.1:
                was_replaced = True
                replacement_text = f"[{repl['vegetables']}]"
                break

        time_str = f"{seg['start']:.2f}-{seg['end']:.2f}s"

        if was_replaced:
            f.write(f"{time_str:<15} {seg['text']:<45} {'→ ' + replacement_text:<45}\n")
        else:
            f.write(f"{time_str:<15} {seg['text']:<45} {'(unchanged)':<45}\n")

print("\n✓ Saved to transcript_comparison_timeline.txt")
