"""
Timeline-based audio processor - clean refactor with declarative approach.

Instead of imperative audio building with complex state management,
this uses a declarative timeline where each segment is independent.
"""
import json
import random
from typing import List, Dict, Optional
from pydub import AudioSegment
import librosa
import numpy as np


class TimelineSegment:
    """Represents a segment in the audio timeline"""
    def __init__(self, start_ms: int, end_ms: int, segment_type: str, **kwargs):
        self.start_ms = start_ms
        self.end_ms = end_ms
        self.duration_ms = end_ms - start_ms
        self.type = segment_type  # "keep" or "replace"
        self.metadata = kwargs

    def __repr__(self):
        if self.type == "keep":
            return f"Keep[{self.start_ms}-{self.end_ms}ms]"
        else:
            return f"Replace[{self.start_ms}-{self.end_ms}ms with {self.metadata.get('replacements', [])}]"


def create_timeline(segments: List[Dict], swear_words: List[str],
                   safety_buffer_ms: int = 100) -> List[TimelineSegment]:
    """
    Create a declarative timeline from transcription segments.

    Uses word-level timestamps to replace ONLY swear words (plus safety buffer),
    keeping all other words in the segment.

    Args:
        segments: WhisperX transcription segments
        swear_words: List of words to replace
        safety_buffer_ms: Extra milliseconds to skip after each swear word

    Returns:
        List of TimelineSegment objects defining what to do with each time range
    """
    timeline = []
    last_end = 0

    for segment in segments:
        start_ms = int(segment.get("start", 0) * 1000)
        end_ms = int(segment.get("end", 0) * 1000)

        # Add any gap before this segment
        if start_ms > last_end:
            timeline.append(TimelineSegment(last_end, start_ms, "keep"))

        # Check if segment has word-level timing
        if "words" in segment and segment["words"]:
            # First pass: identify all swear words and group into phrases
            swear_positions = []
            for word_info in segment["words"]:
                word = word_info["word"].lower().strip()
                word_clean = word.strip('.,!?;:"\'-')
                is_swear = word_clean in swear_words or any(
                    word_clean in [s.rstrip('g'), s + "'", s.rstrip('g') + "'"]
                    for s in swear_words if s.endswith('ing')
                )
                if is_swear:
                    swear_positions.append({
                        "start": int(word_info["start"] * 1000),
                        "end": int(word_info["end"] * 1000),
                        "word": word_clean
                    })

            # Group swears that are close together (< 2000ms apart) into phrases
            # BUT only if there are no non-swear words in between
            swear_phrases = []
            if swear_positions:
                current_phrase = [swear_positions[0]]
                for swear in swear_positions[1:]:
                    gap_ms = swear["start"] - current_phrase[-1]["end"]

                    if gap_ms < 2000:
                        # Check if there are non-swear words in the gap
                        has_non_swear_in_gap = False
                        for word_info in segment["words"]:
                            word_start_ms = int(word_info["start"] * 1000)
                            word_end_ms = int(word_info["end"] * 1000)
                            word_clean = word_info["word"].lower().strip().strip('.,!?;:"\'-')

                            # Check if this word is in the gap between swears
                            if current_phrase[-1]["end"] < word_start_ms < swear["start"]:
                                # This word is in the gap - is it a swear?
                                is_swear = word_clean in swear_words or any(
                                    word_clean in [s.rstrip('g'), s + "'", s.rstrip('g') + "'"]
                                    for s in swear_words if s.endswith('ing')
                                )
                                if not is_swear:
                                    has_non_swear_in_gap = True
                                    break

                        # Only group if no non-swear words in between
                        if not has_non_swear_in_gap:
                            current_phrase.append(swear)
                        else:
                            # Non-swear in between, start new phrase
                            swear_phrases.append(current_phrase)
                            current_phrase = [swear]
                    else:
                        # Too far apart, start new phrase
                        swear_phrases.append(current_phrase)
                        current_phrase = [swear]
                swear_phrases.append(current_phrase)

            # Convert phrases to replacement spans
            replacement_spans = []
            for phrase in swear_phrases:
                phrase_start = phrase[0]["start"]
                phrase_end = phrase[-1]["end"] + safety_buffer_ms

                # Don't extend buffer past the next word's start
                last_swear_end = phrase[-1]["end"]
                for word_info in segment["words"]:
                    word_start = int(word_info["start"] * 1000)
                    if word_start > last_swear_end:
                        # This is the next word after the swear
                        phrase_end = min(phrase_end, word_start)
                        break

                # Also don't extend past segment end
                phrase_end = min(phrase_end, end_ms)
                replacement_spans.append({
                    "start": phrase_start,
                    "end": phrase_end,
                    "words": [s["word"] for s in phrase]
                })

            # Process word by word, skipping entire phrases
            segment_last_end = start_ms

            # Track which replacement spans we've processed
            processed_spans = set()

            for word_info in segment["words"]:
                word_start_ms = int(word_info["start"] * 1000)
                word_end_ms = int(word_info["end"] * 1000)

                # Check if this word is part of a replacement span
                in_replacement_span = False
                for i, span in enumerate(replacement_spans):
                    if span["start"] <= word_start_ms < span["end"]:
                        in_replacement_span = True

                        # If we haven't processed this span yet, do it now
                        if i not in processed_spans:
                            # Keep audio before this span
                            if span["start"] > segment_last_end:
                                timeline.append(TimelineSegment(segment_last_end, span["start"], "keep"))

                            # Replace entire span (phrase)
                            phrase_text = " ".join(span["words"])
                            timeline.append(TimelineSegment(
                                span["start"], span["end"], "replace",
                                original_text=phrase_text,
                                segment_duration=span["end"] - span["start"]
                            ))
                            segment_last_end = span["end"]
                            processed_spans.add(i)
                        break

                if not in_replacement_span:
                    # Keep audio before this word
                    if word_start_ms > segment_last_end:
                        timeline.append(TimelineSegment(segment_last_end, word_start_ms, "keep"))

                    # Keep this word
                    timeline.append(TimelineSegment(word_start_ms, word_end_ms, "keep"))
                    segment_last_end = word_end_ms

            # Keep any remaining audio in segment
            if end_ms > segment_last_end:
                timeline.append(TimelineSegment(segment_last_end, end_ms, "keep"))

        else:
            # No word-level timing - fall back to segment-level if contains profanity
            text = segment.get("text", "").lower()
            contains_profanity = any(swear in text for swear in swear_words)

            if contains_profanity:
                # Replace entire segment (fallback)
                original_text = segment.get("text", "")
                timeline.append(TimelineSegment(
                    start_ms, end_ms, "replace",
                    original_text=original_text,
                    segment_duration=end_ms - start_ms
                ))
            else:
                # Keep original audio
                timeline.append(TimelineSegment(start_ms, end_ms, "keep"))

        last_end = end_ms

    return timeline


def select_vegetables_for_duration(
    target_duration_ms: int,
    original_text: str,
    word_lists_file: str = "word_lists.json",
    min_stretch: float = 0.5,
    max_stretch: float = 1.8,
    recently_used: Optional[List[str]] = None
) -> List[Dict]:
    """
    Select one or more vegetables to fill the target duration.

    Strategy:
    - If target < 1000ms: Use single vegetable with time-stretching
    - If target >= 1000ms: Use multiple vegetables to fill duration naturally

    Args:
        target_duration_ms: Duration to fill
        original_text: Original profanity text (for syllable matching)
        word_lists_file: Path to word lists JSON
        min_stretch: Minimum stretch ratio (0.5 = half speed)
        max_stretch: Maximum stretch ratio (1.8 = 1.8x speed)

    Returns:
        List of vegetable dicts with word, file_path, base_duration, target_duration
    """
    with open(word_lists_file, 'r') as f:
        data = json.load(f)

    vegetables = data['vegetables']
    recently_used = recently_used or []

    # Single vegetable approach for shorter segments
    if target_duration_ms < 1000:
        # Find vegetable with duration closest to target
        candidates = []
        for veg in vegetables:
            # Skip recently used vegetables (last 5)
            if veg['word'] in recently_used[-5:]:
                continue

            base_duration = veg.get('duration', 400)
            stretch_ratio = target_duration_ms / base_duration
            if min_stretch <= stretch_ratio <= max_stretch:
                candidates.append({
                    'word': veg['word'],
                    'file_path': veg['file_path'],
                    'base_duration': base_duration,
                    'target_duration': target_duration_ms,
                    'stretch_ratio': stretch_ratio
                })

        if not candidates:
            # Fallback: pick any vegetable (even if recently used)
            veg = random.choice([v for v in vegetables if v['word'] not in recently_used[-3:]] or vegetables)
            return [{
                'word': veg['word'],
                'file_path': veg['file_path'],
                'base_duration': veg.get('duration', 400),
                'target_duration': target_duration_ms,
                'stretch_ratio': target_duration_ms / veg.get('duration', 400)
            }]

        # Sort by stretch ratio closest to 1.0 (least stretching)
        candidates.sort(key=lambda x: abs(x['stretch_ratio'] - 1.0))
        # Add variety: pick randomly from top 5 candidates instead of always the best
        return [random.choice(candidates[:min(5, len(candidates))])]

    # Multiple vegetables approach for longer segments
    else:
        selected = []
        remaining_duration = target_duration_ms
        pause_between_ms = 100  # Pause between vegetables

        # Keep selecting vegetables until we fill the duration
        attempts = 0
        max_attempts = 10

        while remaining_duration > 300 and attempts < max_attempts:
            attempts += 1

            # Target duration for this vegetable (aim for 40-60% of remaining)
            veg_target = int(remaining_duration * random.uniform(0.4, 0.6))
            veg_target = max(300, min(veg_target, 1200))  # Clamp to reasonable range

            # Find suitable vegetable
            candidates = []
            for veg in vegetables:
                base_duration = veg.get('duration', 400)
                stretch_ratio = veg_target / base_duration
                if min_stretch <= stretch_ratio <= max_stretch:
                    # Avoid recently used vegetables (last 2 in this phrase + last 5 globally)
                    recently_in_phrase = [s['word'] for s in selected[-2:]]
                    if veg['word'] not in recently_in_phrase and veg['word'] not in recently_used[-5:]:
                        candidates.append({
                            'word': veg['word'],
                            'file_path': veg['file_path'],
                            'base_duration': base_duration,
                            'target_duration': veg_target,
                            'stretch_ratio': stretch_ratio
                        })

            if not candidates:
                break

            # Pick random from top candidates
            candidates.sort(key=lambda x: abs(x['stretch_ratio'] - 1.0))
            chosen = random.choice(candidates[:5])
            selected.append(chosen)

            remaining_duration -= (chosen['target_duration'] + pause_between_ms)

        # If we still have significant time left, stretch the last vegetable
        if selected and remaining_duration > 0:
            last = selected[-1]
            last['target_duration'] += remaining_duration
            last['stretch_ratio'] = last['target_duration'] / last['base_duration']

        return selected if selected else select_vegetables_for_duration(
            target_duration_ms, original_text, word_lists_file,
            min_stretch=0.3, max_stretch=2.5  # Relax constraints
        )


def time_stretch_audio(audio_segment: AudioSegment, target_duration_ms: int) -> AudioSegment:
    """Time-stretch audio using librosa while preserving pitch"""
    samples = np.array(audio_segment.get_array_of_samples()).astype(np.float32)
    sr = audio_segment.frame_rate

    if audio_segment.channels > 1:
        samples = samples.reshape((-1, audio_segment.channels)).T

    current_duration_sec = len(audio_segment) / 1000.0
    target_duration_sec = target_duration_ms / 1000.0
    rate = current_duration_sec / target_duration_sec

    if audio_segment.channels == 1:
        stretched = librosa.effects.time_stretch(y=samples, rate=rate)
    else:
        stretched_channels = [
            librosa.effects.time_stretch(y=channel, rate=rate)
            for channel in samples
        ]
        stretched = np.stack(stretched_channels).T.flatten()

    if audio_segment.channels == 1:
        stretched = stretched.flatten()

    stretched = np.clip(stretched, -32768, 32767).astype(np.int16)

    return AudioSegment(
        stretched.tobytes(),
        frame_rate=sr,
        sample_width=audio_segment.sample_width,
        channels=audio_segment.channels
    )


def build_audio_from_timeline(
    original_audio: AudioSegment,
    timeline: List[TimelineSegment],
    word_lists_file: str = "word_lists.json",
    debug: bool = False
) -> tuple[AudioSegment, List[Dict]]:
    """
    Build output audio by following the timeline.
    Pure function - no state management, just concatenate segments.

    Args:
        original_audio: Original audio segment
        timeline: List of TimelineSegment objects
        word_lists_file: Path to word lists JSON
        debug: Print debug info

    Returns:
        (output_audio, replacement_log)
    """
    output = AudioSegment.empty()
    replacement_log = []
    recently_used_vegetables = []  # Track across all replacements

    for segment in timeline:
        if segment.type == "keep":
            # Keep original audio
            chunk = original_audio[segment.start_ms:segment.end_ms]
            output += chunk

            if debug:
                print(f"[KEEP] {segment.start_ms}-{segment.end_ms}ms ({len(chunk)}ms)")

        elif segment.type == "replace":
            # Select vegetables to fill this duration
            vegetables = select_vegetables_for_duration(
                segment.duration_ms,
                segment.metadata.get('original_text', ''),
                word_lists_file,
                recently_used=recently_used_vegetables
            )

            # Update recently used list
            for veg in vegetables:
                recently_used_vegetables.append(veg['word'])

            if debug:
                veg_names = ", ".join(v['word'] for v in vegetables)
                print(f"[REPLACE] {segment.start_ms}-{segment.end_ms}ms ({segment.duration_ms}ms)")
                print(f"  Original: '{segment.metadata.get('original_text', '')}'")
                print(f"  Vegetables: {veg_names}")

            replacement_start_in_output = len(output)

            # Build replacement audio
            for i, veg in enumerate(vegetables):
                # Load vegetable TTS
                veg_audio = AudioSegment.from_file(veg['file_path'])

                # Time-stretch if needed
                if abs(len(veg_audio) - veg['target_duration']) > 10:
                    veg_audio = time_stretch_audio(veg_audio, veg['target_duration'])
                    if debug:
                        print(f"    - {veg['word']}: {veg['base_duration']}ms → {len(veg_audio)}ms (stretch: {veg['stretch_ratio']:.2f}x)")
                else:
                    if debug:
                        print(f"    - {veg['word']}: {len(veg_audio)}ms (no stretch)")

                # Add small fade for smoothness
                fade_ms = min(30, len(veg_audio) // 4)
                veg_audio = veg_audio.fade_in(fade_ms).fade_out(fade_ms)

                output += veg_audio

                # Add small pause between vegetables (except last one)
                if i < len(vegetables) - 1:
                    pause = AudioSegment.silent(duration=100)
                    output += pause

            replacement_end_in_output = len(output)

            # Log replacement
            replacement_log.append({
                'original_start': segment.start_ms,
                'original_end': segment.end_ms,
                'original_duration': segment.duration_ms,
                'original_text': segment.metadata.get('original_text', ''),
                'replacement_start': replacement_start_in_output,
                'replacement_end': replacement_end_in_output,
                'replacement_duration': replacement_end_in_output - replacement_start_in_output,
                'vegetables': [v['word'] for v in vegetables]
            })

    return output, replacement_log


def process_audio_with_timeline(
    audio_path: str,
    segments: List[Dict],
    swear_words: List[str],
    output_path: str = "edited_audio_timeline.wav",
    word_lists_file: str = "word_lists.json",
    debug: bool = False
) -> tuple[str, List[Dict]]:
    """
    Main entry point for timeline-based audio processing.

    Args:
        audio_path: Path to original audio
        segments: WhisperX transcription segments
        swear_words: List of words to replace
        output_path: Path for output audio
        word_lists_file: Path to word lists JSON
        debug: Print debug info

    Returns:
        (output_path, replacement_log)
    """
    if debug:
        print("="*70)
        print("TIMELINE-BASED AUDIO PROCESSING")
        print("="*70)

    # Load original audio
    original_audio = AudioSegment.from_file(audio_path)

    if debug:
        print(f"\nOriginal audio: {len(original_audio)}ms ({len(original_audio)/1000:.2f}s)")

    # Create timeline
    timeline = create_timeline(segments, swear_words)

    if debug:
        print(f"\nTimeline: {len(timeline)} segments")
        replace_count = sum(1 for s in timeline if s.type == "replace")
        keep_count = sum(1 for s in timeline if s.type == "keep")
        print(f"  - Replace: {replace_count}")
        print(f"  - Keep: {keep_count}")
        print()

    # Build output audio
    output_audio, replacement_log = build_audio_from_timeline(
        original_audio, timeline, word_lists_file, debug
    )

    # Export
    output_audio.export(output_path, format="wav")

    if debug:
        print(f"\nOutput audio: {len(output_audio)}ms ({len(output_audio)/1000:.2f}s)")
        print(f"Time saved: {len(original_audio) - len(output_audio)}ms")
        print(f"Replacements: {len(replacement_log)}")
        print(f"Exported to: {output_path}")

    return output_path, replacement_log
