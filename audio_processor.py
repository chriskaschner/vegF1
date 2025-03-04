import os
import tempfile
from pydub import AudioSegment
import librosa
import numpy as np
import subprocess
import json
from typing import Dict, List, Optional

def time_stretch_audio(audio_segment, target_duration):
    """
    Time-stretch the given pydub AudioSegment to match the target duration (in milliseconds)
    using librosa, preserving pitch.
    """
    # Convert AudioSegment to numpy array
    samples = np.array(audio_segment.get_array_of_samples()).astype(np.float32)
    sr = audio_segment.frame_rate

    # If stereo, reshape samples (pydub interleaves channels)
    if audio_segment.channels > 1:
        samples = samples.reshape((-1, audio_segment.channels)).T  # shape: (channels, samples)

    current_duration_sec = len(audio_segment) / 1000.0
    target_duration_sec = target_duration / 1000.0

    # Calculate rate factor (rate > 1 speeds up, rate < 1 slows down)
    rate = current_duration_sec / target_duration_sec

    # Use keyword arguments to pass the rate
    if audio_segment.channels == 1:
        stretched = librosa.effects.time_stretch(y=samples, rate=rate)
    else:
        stretched_channels = []
        for channel in samples:
            stretched_channel = librosa.effects.time_stretch(y=channel, rate=rate)
            stretched_channels.append(stretched_channel)
        stretched = np.stack(stretched_channels)

    # If stereo, re-interleave channels
    if audio_segment.channels > 1:
        stretched = stretched.T.flatten()
    else:
        stretched = stretched.flatten()

    # Convert back to 16-bit PCM
    stretched = np.clip(stretched, -32768, 32767).astype(np.int16)

    new_segment = AudioSegment(
        stretched.tobytes(),
        frame_rate=sr,
        sample_width=audio_segment.sample_width,
        channels=audio_segment.channels
    )

    # Verify the duration is correct (within 5ms tolerance)
    if abs(len(new_segment) - target_duration) > 5:
        print(f"[WARNING] Time-stretched audio duration ({len(new_segment)}ms) doesn't match target ({target_duration}ms)")

    return new_segment

def detect_silence(audio_segment, silence_threshold=-50.0, min_silence_len=50):
    """
    Detect silence at the beginning and end of an audio segment.
    Returns the start and end positions of non-silent audio.
    
    Args:
        audio_segment: The AudioSegment to analyze
        silence_threshold: The threshold (in dB) below which is considered silence
        min_silence_len: Minimum length (in ms) of silence to detect
    
    Returns:
        tuple: (start_pos, end_pos) in milliseconds
    """
    def is_silent(segment):
        return segment.dBFS < silence_threshold

    chunk_length = 10  # ms
    chunks = [audio_segment[i:i+chunk_length] for i in range(0, len(audio_segment), chunk_length)]
    
    # Find start position (first non-silent chunk)
    start_pos = 0
    for i, chunk in enumerate(chunks):
        if not is_silent(chunk):
            start_pos = i * chunk_length
            break
    
    # Find end position (last non-silent chunk)
    end_pos = len(audio_segment)
    for i in range(len(chunks)-1, -1, -1):
        if not is_silent(chunks[i]):
            end_pos = (i + 1) * chunk_length
            break
    
    return start_pos, end_pos

def replace_audio_segments(audio_path, segments, swear_words, get_replacement_fn, 
                         apply_stretch=True, pre_replacement_offset_ms=75, 
                         end_trim_ms=50, fixed_start_offset_ms=50, crossfade_ms=50, debug=False):
    """
    Replace segments of audio containing swear words with TTS-generated vegetable names.
    The replacement audio files are pre-trimmed (silence removed) during generation.
    
    Args:
        audio_path (str): Path to the original audio file
        segments (list): List of transcription segments with word-level timing
        swear_words (list): List of words to replace
        get_replacement_fn (callable): Function that returns replacement word data
        apply_stretch (bool): Whether to time-stretch replacements to match original
        pre_replacement_offset_ms (int): Milliseconds to offset replacement start
        end_trim_ms (int): Milliseconds to trim from end of replacements
        fixed_start_offset_ms (int): Fixed duration in ms to start replacement before swear word (default: 50ms)
        crossfade_ms (int): Duration of crossfade between original and replacement audio (default: 50ms)
        debug (bool): Whether to print debug information
    
    Returns:
        tuple: (output_path, replacements_log)
    """
    # Load the original audio file
    original_audio = AudioSegment.from_file(audio_path)
    
    # Initialize the output audio and replacements log
    output_audio = AudioSegment.empty()
    last_end = 0
    replacements_log = []
    
    for segment in segments:
        if "words" not in segment:
            continue
            
        for word_info in segment["words"]:
            word = word_info["word"].lower()
            start_ms = int(word_info["start"] * 1000)
            end_ms = int(word_info["end"] * 1000)
            
            if word in swear_words:
                # Calculate where to start the replacement
                replacement_start = max(0, start_ms - fixed_start_offset_ms)
                
                # Add audio up to the crossfade point
                if replacement_start > last_end:
                    crossfade_point = max(last_end, replacement_start - crossfade_ms)
                    output_audio += original_audio[last_end:crossfade_point]
                
                # Get replacement word data
                # We'll use a target duration that's shorter than the gap to reduce stretching
                available_duration = end_ms - replacement_start
                target_duration = min(available_duration, available_duration * 0.8)  # Use 80% of available time
                replacement_data = get_replacement_fn(target_duration)
                
                if not replacement_data:
                    if debug:
                        print(f"[WARNING] No replacement found for '{word}' ({target_duration}ms)")
                    continue
                
                # Load replacement audio (already trimmed)
                try:
                    replacement_audio = AudioSegment.from_file(replacement_data["file"])
                    
                    # Apply time stretching if needed, but only if it won't distort too much
                    if apply_stretch:
                        current_duration = len(replacement_audio)
                        stretch_ratio = target_duration / current_duration
                        
                        # Only stretch if the ratio is reasonable (between 0.5 and 1.5)
                        if 0.5 <= stretch_ratio <= 1.5:
                            replacement_audio = time_stretch_audio(replacement_audio, target_duration)
                        elif debug:
                            print(f"[INFO] Skipping time-stretch for '{word}' (ratio {stretch_ratio:.2f} too extreme)")
                    
                    # Apply fade in/out to smooth transitions
                    fade_duration = min(crossfade_ms, len(replacement_audio) // 4)
                    replacement_audio = replacement_audio.fade_in(fade_duration).fade_out(fade_duration)
                    
                    # Create crossfade with original audio at the start
                    if replacement_start > last_end:
                        crossfade_segment = original_audio[replacement_start - crossfade_ms:replacement_start]
                        if len(crossfade_segment) >= crossfade_ms:
                            # Crossfade the beginning
                            output_audio = output_audio.append(replacement_audio, 
                                                            crossfade=min(crossfade_ms, 
                                                                        len(crossfade_segment),
                                                                        len(replacement_audio)))
                        else:
                            # If not enough audio for crossfade, just append
                            output_audio += replacement_audio
                    else:
                        # If we're starting at the beginning or overlapping, just append
                        output_audio += replacement_audio
                    
                    # Log the replacement
                    replacements_log.append({
                        "original_word": word,
                        "replacement_word": replacement_data["word"],
                        "original_start": start_ms,
                        "original_end": end_ms,
                        "original_duration": end_ms - start_ms,
                        "replacement_start": len(output_audio) - len(replacement_audio),
                        "replacement_end": len(output_audio),
                        "replacement_duration": len(replacement_audio),
                        "available_duration": available_duration,
                        "target_duration": target_duration,
                        "crossfade_duration": fade_duration
                    })
                    
                    if debug:
                        print(f"[DEBUG] Replaced '{word}' with '{replacement_data['word']}'")
                        print(f"[DEBUG] Original duration: {end_ms - start_ms}ms")
                        print(f"[DEBUG] Available duration: {available_duration}ms")
                        print(f"[DEBUG] Target duration: {target_duration}ms")
                        print(f"[DEBUG] Final duration: {len(replacement_audio)}ms")
                        print(f"[DEBUG] Crossfade duration: {fade_duration}ms")
                    
                except Exception as e:
                    print(f"[ERROR] Failed to process replacement for '{word}': {e}")
                    # In case of error, keep the original audio
                    output_audio += original_audio[start_ms:end_ms]
                
                last_end = end_ms
            
    # Add any remaining audio
    if last_end < len(original_audio):
        output_audio += original_audio[last_end:]
    
    # Create final output file with AIFF format (native to macOS)
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.aiff')
    
    # Export directly to AIFF with high quality settings
    output_audio.export(
        temp_output.name,
        format='aiff',
        parameters=[
            "-ar", "44100",  # Sample rate
            "-ac", "2",      # Stereo
            "-acodec", "pcm_s16be"  # 16-bit PCM big-endian (standard for AIFF)
        ]
    )
    
    if debug:
        print(f"[DEBUG] Exported audio to AIFF: {temp_output.name}")
    
    return temp_output.name, replacements_log

def update_word_lists_with_durations(word_lists_file: str = 'word_lists.json') -> None:
    """
    Update word_lists.json with duration information from the TTS cache.
    This ensures we store the duration information persistently.
    """
    # Load existing word lists
    with open(word_lists_file, 'r') as f:
        word_lists = json.load(f)

    # Load TTS cache
    cache_file = os.path.join("tts_cache", "tts_cache.json")
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            tts_cache = json.load(f)
    else:
        return  # No cache to update from

    # Update vegetables with duration information
    updated_vegetables = []
    for veg_entry in word_lists['vegetables']:
        word = veg_entry['word']
        if word in tts_cache:
            # Get the actual content duration (excluding silence)
            audio_path = tts_cache[word]['file']
            if os.path.exists(audio_path):
                try:
                    audio = AudioSegment.from_file(audio_path)
                    start, end = detect_silence(audio)
                    content_duration = end - start
                    
                    # Update the entry with duration information
                    updated_vegetables.append({
                        'word': word,
                        'syllables': veg_entry['syllables'],
                        'duration': content_duration
                    })
                    continue
                except Exception as e:
                    print(f"[WARNING] Failed to process duration for {word}: {e}")
        
        # If we couldn't get duration, keep the original entry
        updated_vegetables.append(veg_entry)

    # Update the word lists file
    word_lists['vegetables'] = updated_vegetables
    with open(word_lists_file, 'w') as f:
        json.dump(word_lists, f, indent=2)

def get_vegetable_replacement(target_duration: int, max_ratio: float = 1.6, recent_used_limit: int = 5) -> Optional[Dict]:
    """
    Select a vegetable replacement based on duration matching and variety.
    Uses the duration information stored in word_lists.json for efficient lookup.
    
    Args:
        target_duration: Target duration in milliseconds
        max_ratio: Maximum allowed ratio between target and replacement duration
        recent_used_limit: Number of recently used words to avoid
    
    Returns:
        Dict with word, duration, and file path information, or None if no suitable match
    """
    # Load word lists with duration information
    with open('word_lists.json', 'r') as f:
        word_lists = json.load(f)
    
    # Initialize recent used tracking if needed
    if not hasattr(get_vegetable_replacement, "recent_used"):
        get_vegetable_replacement.recent_used = []

    # Filter vegetables by duration ratio and exclude recently used
    candidates = []
    for veg in word_lists['vegetables']:
        if veg.get('duration') is None or veg['word'] in get_vegetable_replacement.recent_used:
            continue
            
        duration = veg['duration']
        ratio = max(duration / target_duration, target_duration / duration)
        
        if ratio <= max_ratio:
            candidates.append({
                'word': veg['word'],
                'duration': duration,
                'ratio': ratio,
                'file_path': veg['file_path']
            })
    
    if not candidates:
        # If no candidates, clear recent used and try again without that restriction
        if get_vegetable_replacement.recent_used:
            get_vegetable_replacement.recent_used = []
            return get_vegetable_replacement(target_duration, max_ratio)
        return None

    # Sort candidates by ratio (closest to 1.0 first)
    candidates.sort(key=lambda x: x['ratio'])
    
    # Select from top 3 candidates randomly to add variety
    import random
    selected = random.choice(candidates[:3])
    
    # Update recently used list
    get_vegetable_replacement.recent_used.append(selected['word'])
    if len(get_vegetable_replacement.recent_used) > recent_used_limit:
        get_vegetable_replacement.recent_used.pop(0)
    
    return {
        "word": selected['word'],
        "length_ms": selected['duration'],
        "file": selected['file_path']
    } 