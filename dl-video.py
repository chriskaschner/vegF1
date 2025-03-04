import yt_dlp
import subprocess
import os
import json
import re
import syllapy
from urllib.parse import urlparse, parse_qs
import whisperx
import random
import tempfile
import openai
from audio_processor import replace_audio_segments
import argparse


# Set OpenAI API Key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

CACHE_DIR = "tts_cache"
CACHE_FILE = os.path.join(CACHE_DIR, "tts_cache.json")

# Load the cached TTS metadata
def load_tts_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    return {}

TTS_CACHE = load_tts_cache()

# =========================
# Video & Audio Processing
# =========================

def get_video_id(url):
    # Extract the 'v' parameter from the URL query string.
    parsed_url = urlparse(url)
    video_id = parse_qs(parsed_url.query).get('v', [None])[0]
    return video_id

def download_video(url, output_dir='videos', archive_file='download_archive.txt'):
    """
    Downloads a YouTube video in a format that is compatible with QuickTime.
    Ensures H.264 video + AAC audio in an MP4 container to avoid re-encoding.
    """
    video_id = get_video_id(url)
    if video_id is None:
        raise ValueError("Couldn't extract video ID from URL")

    os.makedirs(output_dir, exist_ok=True)
    output_template = os.path.join(output_dir, f"{video_id}.mp4")

    # Check if file already exists
    if os.path.exists(output_template):
        print(f"Video file {output_template} already exists. Skipping download.")
        return output_template

    ydl_opts = {
        'format': 'bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]',  # Ensures H.264 + AAC
        'outtmpl': output_template,  # Output filename template
        'merge_output_format': 'mp4',  # Ensure final file is MP4
        'download_archive': archive_file,  # Avoid re-downloading
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    if os.path.exists(output_template):
        print(f"[INFO] Downloaded and saved video: {output_template}")
        return output_template

    raise FileNotFoundError(f"[ERROR] Failed to download video {video_id} in MP4 format.")


def extract_audio(video_file, audio_dir='audio'):
    os.makedirs(audio_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(video_file))[0]
    audio_file = os.path.join(audio_dir, f"{base_name}.wav")
    
    # Skip extraction if audio file already exists.
    if os.path.exists(audio_file):
        print(f"{audio_file} already exists. Skipping audio extraction.")
        return audio_file

    cmd = ['ffmpeg', '-i', video_file, '-q:a', '0', '-map', 'a', audio_file]
    subprocess.run(cmd, check=True)
    print("Audio extraction complete.")
    return audio_file

def transcribe_audio(audio_file, device="cpu", language="en"):
    """
    Transcribe the audio file using WhisperX to get word-level timestamps.
    Ensures that English ("en") is explicitly specified to prevent auto-detection.
    """
    # Force compute_type to "float32" on CPU
    compute_type = "float32" if device == "cpu" else "float16"

    # Load WhisperX model with the specified compute type and language
    model = whisperx.load_model("base", device=device, compute_type=compute_type)

    # Transcribe audio (explicitly specifying English)
    result = model.transcribe(audio_file, language=language)

    # Explicitly specify the language for alignment model
    align_model, metadata = whisperx.load_align_model(language_code=language, device=device)

    # Align words with timestamps (forcing English)
    result_aligned = whisperx.align(result["segments"], align_model, metadata, audio_file, device=device)

    return result_aligned["segments"]

def save_transcript(segments, filename="transcript.txt", replacement_log="replacements_log.txt", replacements=[]):
    """
    Save:
    - A clean transcript from the transcription segments.
    - A replacement log showing swear word substitutions with timestamps and durations.

    Args:
        segments (list): Transcription segments from WhisperX.
        filename (str): Standard transcript file.
        replacement_log (str): File with replacement details.
        replacements (list): List of tuples (start_ms, end_ms, original_word, replacement_word, replacement_start, replacement_end).
    """
    # Save the standard transcript
    with open(filename, "w", encoding="utf-8") as f:
        for segment in segments:
            start = segment.get("start", 0)
            end = segment.get("end", 0)
            text = segment.get("text", "")
            f.write(f"[{start:.2f} - {end:.2f}]: {text}\n")

    print(f"Transcript saved to {filename}")
    
    # Generate properly synced audio with playback speed adjustment and offset
    synced_audio, replacements_log = replace_audio_segments(
        audio_path,
        segments,
        swear_words=SWEAR_WORDS,
        get_replacement_fn=get_vegetable_replacement,
        apply_stretch=True,
        pre_replacement_offset_ms=75,
        end_trim_ms=50,
        debug=True
    )
    
    # Save the replacements log to a file
    with open("replacements_log.txt", "w", encoding="utf-8") as log_file:
        log_file.write("Original Word Replacements Log:\n")
        log_file.write("Format: [Original Start-End] Original -> [Replacement Start-End] Replacement (Durations)\n\n")

        for log in replacements_log:
            log_file.write(f"[{log['original_start']}-{log['original_end']}ms] {log['original_word']} "
                        f"-> [{log['replacement_start']}-{log['replacement_end']}ms] {log['replacement_word']} "
                        f"(Original: {log['original_duration']}ms, Replacement: {log['replacement_duration']}ms)\n")
    print("[INFO] Replacements log saved to replacements_log.txt")


# ========================
# Text Processing Section
# ========================

def load_word_lists_with_syllables(filename='word_lists.json'):
    """
    Load swear words and vegetables (with syllable counts) from a JSON file.
    """
    with open(filename, 'r') as f:
        data = json.load(f)
    swear_words = data.get('swear_words', [])
    # Create a dictionary mapping each vegetable to its syllable count.
    vegetable_syllables = {item['word']: item['syllables'] for item in data.get('vegetables', [])}
    return swear_words, vegetable_syllables

# Load the word lists from the JSON file.
SWEAR_WORDS, VEGETABLE_SYLLABLES = load_word_lists_with_syllables()

def get_syllable_count(word):
    """For swear words, we calculate the syllable count on the fly using syllapy."""
    return syllapy.count(word)

def get_vegetable_replacement(target_duration, cache_file="tts_cache/tts_cache.json", recent_used_limit=5):
    """
    Selects a cached TTS vegetable with improved variety and matching.
    
    - Considers both duration and syllable count for better matches
    - Avoids recently used vegetables
    - Picks from a larger pool of candidates
    
    Args:
        target_duration (int): Duration (in ms) of the original word to replace.
        cache_file (str): Path to the TTS cache JSON.
        recent_used_limit (int): Number of recently used words to avoid

    Returns:
        dict: {"word": selected_word, "length_ms": duration, "file": path_to_audio}
    """
    if not os.path.exists(cache_file):
        raise FileNotFoundError(f"[ERROR] TTS cache file {cache_file} not found!")

    with open(cache_file, "r") as f:
        cache = json.load(f)

    # Ensure target_duration is an integer
    try:
        target_duration = int(target_duration) 
    except ValueError:
        print(f"[ERROR] Invalid target duration: {target_duration} (expected an integer)")
        return None

    # Keep track of recently used vegetables (initialize if not exists)
    if not hasattr(get_vegetable_replacement, "recent_used"):
        get_vegetable_replacement.recent_used = []

    # Extract words and durations, excluding recently used ones
    candidates = []
    for word, data in cache.items():
        if word in get_vegetable_replacement.recent_used:
            continue
        
        word_duration = int(data["length_ms"])
        # Get syllable count from VEGETABLE_SYLLABLES
        syllables = VEGETABLE_SYLLABLES.get(word, 1)
        candidates.append((word, word_duration, syllables, data["file"]))

    if not candidates:
        # If all words were recently used, clear history and try again
        get_vegetable_replacement.recent_used = []
        return get_vegetable_replacement(target_duration, cache_file)

    # Score candidates based on duration difference and syllable count
    # We'll get the syllable count of the original word from the cache
    target_syllables = 2  # Default to 2 syllables if unknown
    
    scored_candidates = []
    for word, duration, syllables, file in candidates:
        # Calculate duration score (0 to 1, where 1 is perfect match)
        duration_diff = abs(duration - target_duration)
        max_diff = max(target_duration, 1000)  # Cap at 1 second difference
        duration_score = 1 - (duration_diff / max_diff)
        
        # Calculate syllable score (0 to 1, where 1 is perfect match)
        syllable_diff = abs(syllables - target_syllables)
        syllable_score = 1 - (syllable_diff / 4)  # Assume max syllable difference of 4
        
        # Combined score (weight duration more heavily)
        total_score = (duration_score * 0.7) + (syllable_score * 0.3)
        
        scored_candidates.append((word, duration, file, total_score))

    # Sort by score and take top 5 candidates
    scored_candidates.sort(key=lambda x: x[3], reverse=True)
    best_matches = scored_candidates[:5]

    # Randomly choose from the best matches
    chosen_word, chosen_duration, chosen_file, _ = random.choice(best_matches)

    # Update recently used list
    get_vegetable_replacement.recent_used.append(chosen_word)
    if len(get_vegetable_replacement.recent_used) > recent_used_limit:
        get_vegetable_replacement.recent_used.pop(0)

    return {
        "word": chosen_word,
        "length_ms": chosen_duration,
        "file": chosen_file
    }


def extract_swear_durations(segments):
    """
    Extracts the duration of each swear word in milliseconds based on WhisperX timestamps.

    Args:
        segments (list): List of transcription segments from WhisperX.

    Returns:
        dict: {swear_word: duration_in_ms}
    """
    swear_durations = {}

    for segment in segments:
        if "words" in segment:
            for word_info in segment["words"]:
                word = word_info["word"].lower()
                if word in SWEAR_WORDS:
                    start_ms = int(word_info["start"] * 1000)
                    end_ms = int(word_info["end"] * 1000)
                    duration = end_ms - start_ms
                    swear_durations[word] = duration

    return swear_durations

def replace_swears_in_segment(segment_text, swear_durations):
    """
    Replace swear words in the text with vegetable names while preserving case.
    
    - Uses cached TTS data for replacements.
    - Matches durations from the swear word list.

    Args:
        segment_text (str): Original text containing swear words.
        swear_durations (dict): Mapping of swear words to their detected durations (ms).
    
    Returns:
        str: Text with swear words replaced.
    """
    pattern = re.compile("|".join(map(re.escape, SWEAR_WORDS)), re.IGNORECASE)

    def replacer(match):
        swear = match.group(0)
        
        # Lookup duration of the swear word
        target_duration = swear_durations.get(swear.lower(), 300)  # Default to 300ms if missing
        
        # Get a replacement vegetable from cache
        replacement_data = get_vegetable_replacement(target_duration)

        replacement_word = replacement_data["word"]
        
        # Preserve case of original word
        if swear.isupper():
            return replacement_word.upper()
        elif swear[0].isupper():
            return replacement_word.capitalize()
        else:
            return replacement_word

    return pattern.sub(replacer, segment_text)


# ========================
# Synthesize new audio Section
# ========================

def synthesize_tts(text, voice="alloy", debug=False):
    """
    Uses OpenAI's TTS API to generate high-quality, natural-sounding speech.

    Parameters:
        - text (str): The text to be spoken.
        - voice (str): OpenAI TTS voice ("alloy", "echo", "fable", "onyx", "nova", "shimmer").
        - debug (bool): If True, saves the generated audio for inspection.

    Returns:
        - temp_file.name (str): Path to the generated audio file, or None if an error occurs.
    """

    # Validate API key
    if not openai.api_key:
        raise ValueError("[ERROR] Missing OpenAI API key. Set the 'OPENAI_API_KEY' environment variable.")

    try:
        # Ensure `text` is a string before making the API call
        if not isinstance(text, str):
            raise ValueError(f"[ERROR] TTS input must be a string, got {type(text)}: {text}")

        # Generate speech using OpenAI TTS
        response = openai.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text  # ✅ Only pass the replacement word as a string
        )

        # Create a temporary MP3 file to store the audio
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")

        # Properly extract the binary audio content
        audio_data = response.read()

        # Save the response audio to the temporary file
        with open(temp_file.name, "wb") as f:
            f.write(audio_data)

        if debug:
            print(f"[DEBUG] Saved OpenAI TTS audio: {temp_file.name}")

        return temp_file.name

    except Exception as e:
        print(f"[ERROR] OpenAI TTS generation failed: {e}")
        return None


# def time_stretch_audio(audio_segment, target_duration):
#     """
#     Time-stretch the given pydub AudioSegment to match the target duration (in milliseconds)
#     using librosa, preserving pitch.
#     """
#     # Convert AudioSegment to numpy array
#     samples = np.array(audio_segment.get_array_of_samples()).astype(np.float32)
#     sr = audio_segment.frame_rate

#     # If stereo, reshape samples (pydub interleaves channels)
#     if audio_segment.channels > 1:
#         samples = samples.reshape((-1, audio_segment.channels)).T  # shape: (channels, samples)

#     current_duration_sec = len(audio_segment) / 1000.0
#     target_duration_sec = target_duration / 1000.0

#     # Calculate rate factor.
#     rate = current_duration_sec / target_duration_sec

#     # Use keyword arguments to pass the rate.
#     if audio_segment.channels == 1:
#         stretched = librosa.effects.time_stretch(y=samples, rate=rate)
#     else:
#         stretched_channels = []
#         for channel in samples:
#             stretched_channel = librosa.effects.time_stretch(y=channel, rate=rate)
#             stretched_channels.append(stretched_channel)
#         stretched = np.stack(stretched_channels)

#     # If stereo, re-interleave channels.
#     if audio_segment.channels > 1:
#         stretched = stretched.T.flatten()
#     else:
#         stretched = stretched.flatten()

#     # Convert back to 16-bit PCM.
#     stretched = np.clip(stretched, -32768, 32767).astype(np.int16)

#     new_segment = AudioSegment(
#         stretched.tobytes(),
#         frame_rate=sr,
#         sample_width=audio_segment.sample_width,
#         channels=audio_segment.channels
#     )
#     return new_segment

def merge_audio_video(video_file, new_audio_file, output_video='final_output.mp4'):
    """
    Merge the new audio file with the original video file using FFmpeg.
    Ensures the output file is compatible with macOS QuickTime.

    Parameters:
      - video_file: Path to the original video file.
      - new_audio_file: Path to the newly generated audio file.
      - output_video: Path to the final MP4 output.

    Returns:
      - The path to the generated QuickTime-compatible MP4 file.
    """
    import subprocess

    # Create a temporary file for the intermediate conversion
    temp_output = os.path.splitext(output_video)[0] + "_temp.mp4"

    # First command: Create a QuickTime-compatible MP4 with original video and new audio
    cmd = [
        'ffmpeg',
        '-y',  # Overwrite without asking
        '-i', video_file,  # Input: Original video
        '-i', new_audio_file,  # Input: New audio
        '-map', '0:v:0',  # Take video from first input
        '-map', '1:a:0',  # Take audio from second input
        '-c:v', 'h264',  # Force H.264 codec for video
        '-profile:v', 'high',  # High profile H.264
        '-pix_fmt', 'yuv420p',  # Standard pixel format
        '-c:a', 'aac',  # AAC audio codec
        '-b:a', '192k',  # Audio bitrate
        '-ac', '2',  # Stereo audio
        '-ar', '44100',  # Standard audio sample rate
        '-shortest',  # Duration of shortest input
        '-movflags', '+faststart',  # Enable fast start for streaming
        temp_output  # Temporary output file
    ]

    print(f"[INFO] Running first FFmpeg command:\n{' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] FFmpeg error: {e.stderr.decode()}")
        raise

    # Second command: Copy to final output ensuring container compatibility
    cmd2 = [
        'ffmpeg',
        '-y',
        '-i', temp_output,
        '-c', 'copy',  # Just copy streams without re-encoding
        '-movflags', '+faststart',  # Ensure QuickTime compatibility
        '-f', 'mp4',  # Force MP4 format
        output_video
    ]

    print(f"[INFO] Running second FFmpeg command:\n{' '.join(cmd2)}")
    try:
        subprocess.run(cmd2, check=True, capture_output=True)
        # Clean up temporary file
        os.unlink(temp_output)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] FFmpeg error: {e.stderr.decode()}")
        # Clean up temporary file even if there's an error
        if os.path.exists(temp_output):
            os.unlink(temp_output)
        raise

    print(f"[INFO] Final video saved as: {output_video}")
    return output_video

# ========================
# Main Execution Section
# ========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download and process YouTube video')
    parser.add_argument('url', help='YouTube video URL')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    args = parser.parse_args()

    video_path = download_video(args.url)
    audio_path = extract_audio(video_path)
    segments = transcribe_audio(audio_path)

    save_transcript(segments, "transcript.txt")

    print("[DEBUG] Checking TTS Cache Integrity...")
    for word, data in TTS_CACHE.items():
        if not os.path.exists(data["file"]):
            print(f"[ERROR] Cached TTS file missing for {word}: {data['file']}")
        else:
            print(f"[DEBUG] TTS Cache OK: {word} -> {data['file']} ({data['length_ms']} ms)")

    # Extract swear word durations
    swear_durations = extract_swear_durations(segments)

    # Properly synced audio with playback speed adjustment
    synced_audio, replacements_log = replace_audio_segments(
        audio_path,
        segments,
        swear_words=SWEAR_WORDS,
        get_replacement_fn=get_vegetable_replacement,
        apply_stretch=True,
        pre_replacement_offset_ms=75,
        end_trim_ms=50,
        debug=True
    )

    # Debugging: Verify the generated audio file exists
    if not os.path.exists(synced_audio):
        raise FileNotFoundError(f"[ERROR] Edited audio file '{synced_audio}' was not created!")

    print(f"[DEBUG] Verified edited audio exists: {synced_audio}")
    
    final_video_synced = merge_audio_video(video_path, synced_audio, output_video='final_output_synced.mp4')

