import json
import os
import argparse
import openai
from pydub import AudioSegment
import syllapy

CACHE_DIR = "tts_cache"
CACHE_FILE = os.path.join(CACHE_DIR, "tts_cache.json")

def load_existing_word_lists(filename='word_lists.json'):
    """Load existing swear words and vegetable words from the JSON file."""
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        return data.get('swear_words', []), data.get('vegetables', [])
    return [], []

def get_input_words(prompt):
    """Prompt user to input words, allowing comma-separated input or one per line."""
    print(prompt)
    words = []
    while True:
        line = input("> ").strip()
        if line == "":
            break
        words.extend(word.strip() for word in line.split(",") if word.strip())
    return words

def generate_word_lists(filename='word_lists.json'):
    """Update and save word lists while preserving existing words."""
    print("Let's update your word lists!")

    # Load existing word lists
    existing_swears, existing_vegs = load_existing_word_lists(filename)
    
    if existing_swears:
        print("Existing swear words:", existing_swears)
    new_swears = get_input_words("Enter additional swear words (or press Enter to keep existing):")
    combined_swears = list(set(existing_swears + new_swears))  # Remove duplicates

    # Process vegetables
    existing_veg_words = [item['word'] for item in existing_vegs] if existing_vegs else []
    if existing_veg_words:
        print("Existing vegetable words:", existing_veg_words)
    new_vegs = get_input_words("Enter additional vegetable words (or press Enter to keep existing):")
    combined_veg_words = list(set(existing_veg_words + new_vegs))

    # Recalculate syllable counts
    vegetables = [{"word": veg, "syllables": syllapy.count(veg)} for veg in combined_veg_words]

    # Save updated lists
    updated_word_lists = {"swear_words": combined_swears, "vegetables": vegetables}
    with open(filename, "w") as f:
        json.dump(updated_word_lists, f, indent=2)
    print(f"Word lists saved to {filename}")
    
    return updated_word_lists

def generate_tts_cache(word_list_file='word_lists.json', voice="alloy", regen_tts=False):
    """
    Generates or loads cached TTS files.
    - If `regen_tts=True`, regenerates all TTS.
    - If `regen_tts=False`, only generates missing TTS.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Load word lists
    with open(word_list_file, 'r') as f:
        data = json.load(f)
    vegetables = data.get('vegetables', [])

    # Load existing cache
    existing_cache = {}
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            existing_cache = json.load(f)

    cache = {}

    for veg_data in vegetables:
        veg = veg_data['word']
        syllables = veg_data['syllables']
        tts_file = os.path.join(CACHE_DIR, f"{veg}.mp3")

        # Skip already cached files if not forcing regen
        if not regen_tts and os.path.exists(tts_file):
            cache[veg] = existing_cache.get(veg, {"file": tts_file})
            print(f"[CACHE] Using cached TTS for '{veg}'")
            continue

        print(f"[INFO] Generating TTS for '{veg}'...")

        # Generate TTS audio with OpenAI
        try:
            response = openai.audio.speech.create(
                model="tts-1",
                voice=voice,
                input=veg
            )

            # Save the audio file
            with open(tts_file, "wb") as f:
                f.write(response.read())

            # Load audio to get duration
            audio = AudioSegment.from_file(tts_file)
            length_ms = len(audio)

            # Save metadata
            cache[veg] = {
                "syllables": syllables,
                "length_ms": length_ms,
                "file": tts_file
            }
            print(f"[INFO] Saved {veg} TTS ({length_ms} ms)")

        except Exception as e:
            print(f"[ERROR] Failed to generate TTS for '{veg}': {e}")

    # Save updated cache
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)
    print(f"[INFO] TTS cache updated at {CACHE_FILE}")

def main():
    parser = argparse.ArgumentParser(description="Generate or update word lists and TTS cache.")
    parser.add_argument("--regen-tts", action="store_true", help="Regenerate all TTS audio instead of using cache.")
    args = parser.parse_args()

    # Update word lists
    generate_word_lists("word_lists.json")

    # Generate TTS cache (only regen if flag is provided)
    generate_tts_cache(regen_tts=args.regen_tts)

if __name__ == "__main__":
    main()