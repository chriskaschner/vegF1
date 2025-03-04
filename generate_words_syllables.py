import json
import os
import argparse
import openai
from pydub import AudioSegment
import syllapy
from audio_processor import detect_silence

CACHE_DIR = "tts_cache"

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

    # Create or update vegetable entries
    vegetables = []
    for veg in combined_veg_words:
        # Find existing entry to preserve any existing data
        existing_entry = next((item for item in existing_vegs if item['word'] == veg), None)
        if existing_entry:
            # Ensure file_path exists in existing entries
            if 'file_path' not in existing_entry:
                existing_entry['file_path'] = os.path.join(CACHE_DIR, f"{veg}.mp3")
            vegetables.append(existing_entry)
        else:
            vegetables.append({
                "word": veg,
                "syllables": syllapy.count(veg),
                "duration": None,  # Will be set when TTS is generated
                "file_path": os.path.join(CACHE_DIR, f"{veg}.mp3")
            })

    # Save updated lists
    updated_word_lists = {"swear_words": combined_swears, "vegetables": vegetables}
    with open(filename, "w") as f:
        json.dump(updated_word_lists, f, indent=2)
    print(f"Word lists saved to {filename}")
    
    return updated_word_lists

def generate_tts(word_list_file='word_lists.json', voice="alloy", regen_tts=False):
    """
    Generates TTS files and updates word_lists.json with duration information.
    - If `regen_tts=True`, regenerates all TTS.
    - If `regen_tts=False`, only generates missing TTS.
    - Trims silence from audio files and stores trimmed durations.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Load word lists
    with open(word_list_file, 'r') as f:
        data = json.load(f)
    vegetables = data['vegetables']

    # Count how many TTS files need to be generated
    to_generate = []
    for veg_entry in vegetables:
        veg = veg_entry['word']
        tts_file = veg_entry['file_path']
        if regen_tts or not os.path.exists(tts_file) or veg_entry.get('duration') is None:
            to_generate.append(veg)

    if not to_generate:
        print("[INFO] No TTS files need to be generated.")
        return

    # Show confirmation with cost estimate (approximately $0.015 per 1K characters)
    total_chars = sum(len(veg) for veg in to_generate)
    estimated_cost = (total_chars / 1000) * 0.015
    
    print("\nTTS Generation Summary:")
    print(f"- Words to generate: {len(to_generate)}")
    print(f"- Total characters: {total_chars}")
    print(f"- Estimated cost: ${estimated_cost:.3f}")
    print("\nWords that will be generated:")
    for veg in to_generate:
        print(f"- {veg}")
    
    confirm = input("\nWould you like to proceed? (y/N): ").lower().strip()
    if confirm != 'y':
        print("Operation cancelled.")
        return

    # Process each vegetable
    for veg_entry in vegetables:
        veg = veg_entry['word']
        tts_file = veg_entry['file_path']
        temp_file = os.path.join(CACHE_DIR, f"{veg}_temp.mp3")

        # Skip if TTS exists and not forcing regeneration
        if not regen_tts and os.path.exists(tts_file) and veg_entry.get('duration') is not None:
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

            # Save to temporary file first
            with open(temp_file, "wb") as f:
                f.write(response.read())

            # Load audio, detect silence, and trim
            audio = AudioSegment.from_file(temp_file)
            start, end = detect_silence(audio)
            trimmed_audio = audio[start:end]

            # Save the trimmed version as the final file
            trimmed_audio.export(tts_file, format="mp3")

            # Update entry with trimmed duration
            veg_entry['duration'] = len(trimmed_audio)
            print(f"[INFO] Saved {veg} TTS (trimmed duration: {len(trimmed_audio)}ms)")

            # Clean up temporary file
            os.remove(temp_file)

        except Exception as e:
            print(f"[ERROR] Failed to generate TTS for '{veg}': {e}")
            if os.path.exists(temp_file):
                os.remove(temp_file)

    # Save updated word lists
    with open(word_list_file, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[INFO] Word lists updated with TTS information")

def main():
    parser = argparse.ArgumentParser(description="Generate or update word lists and TTS files.")
    parser.add_argument("--regen-tts", action="store_true", help="Regenerate all TTS audio instead of using cache.")
    args = parser.parse_args()

    # Update word lists
    generate_word_lists("word_lists.json")

    # Generate TTS and update durations
    generate_tts(regen_tts=args.regen_tts)

if __name__ == "__main__":
    main()