import os
import tempfile
from pydub import AudioSegment
import librosa
import numpy as np
import subprocess

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

    # Calculate rate factor
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
                         end_trim_ms=50, debug=False):
    """
    Replace segments of audio containing swear words with TTS-generated vegetable names.
    
    Args:
        audio_path (str): Path to the original audio file
        segments (list): List of transcription segments with word-level timing
        swear_words (list): List of words to replace
        get_replacement_fn (callable): Function that returns replacement word data
        apply_stretch (bool): Whether to time-stretch replacements to match original
        pre_replacement_offset_ms (int): Milliseconds to offset replacement start
        end_trim_ms (int): Milliseconds to trim from end of replacements
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
                # Add audio up to this word, considering the pre-replacement offset
                adjusted_start = max(0, start_ms - pre_replacement_offset_ms)
                if adjusted_start > last_end:
                    output_audio += original_audio[last_end:adjusted_start]
                
                # Get replacement word data
                replacement_data = get_replacement_fn(end_ms - start_ms)
                if not replacement_data:
                    if debug:
                        print(f"[WARNING] No replacement found for '{word}' ({end_ms - start_ms}ms)")
                    continue
                
                # Load replacement audio
                try:
                    replacement_audio = AudioSegment.from_file(replacement_data["file"])
                    
                    # Detect actual content boundaries (removing silence)
                    content_start, content_end = detect_silence(replacement_audio)
                    replacement_audio = replacement_audio[content_start:content_end]
                    
                    # Apply time stretching if needed
                    if apply_stretch:
                        target_duration = end_ms - start_ms
                        replacement_audio = time_stretch_audio(replacement_audio, target_duration)
                    
                    # Apply fade in/out to smooth transitions
                    fade_duration = min(20, len(replacement_audio) // 4)  # 20ms or quarter of duration
                    replacement_audio = replacement_audio.fade_in(fade_duration).fade_out(fade_duration)
                    
                    # Trim end if specified (after time stretching)
                    if end_trim_ms > 0:
                        trim_amount = min(end_trim_ms, len(replacement_audio) // 3)  # Don't trim more than 1/3
                        replacement_audio = replacement_audio[:-trim_amount]
                    
                    # Add the replacement audio
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
                        "content_trim": {
                            "start_trim": content_start,
                            "end_trim": len(replacement_audio) - content_end
                        }
                    })
                    
                    if debug:
                        print(f"[DEBUG] Replaced '{word}' with '{replacement_data['word']}'")
                        print(f"[DEBUG] Original duration: {end_ms - start_ms}ms")
                        print(f"[DEBUG] Replacement duration: {len(replacement_audio)}ms")
                        print(f"[DEBUG] Content trimmed: start={content_start}ms, end={len(replacement_audio) - content_end}ms")
                    
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