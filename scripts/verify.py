"""
Unified verification tool for edited audio.
Transcribes audio and checks for remaining swear words.
Replaces: verify_with_transcription.py, check_edited.py
"""
import argparse
import whisperx
import torch
import json
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def verify_audio(audio_path, swear_words_path, device="cpu", compute_type="float32"):
    """
    Transcribe audio and check for swear words.

    Returns:
        tuple: (transcription result, list of found swears)
    """
    print("="*70)
    print("VERIFICATION: Transcribing audio to check for swear words")
    print("="*70)

    # Load swear words
    with open(swear_words_path, "r") as f:
        swear_words = json.load(f)['swear_words']

    # Load model
    print("\nLoading WhisperX model...")
    model = whisperx.load_model("base", device, compute_type=compute_type)

    # Transcribe
    print(f"\nTranscribing {audio_path}...")
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=16)

    print("\nAligning words...")
    model_a, metadata = whisperx.load_align_model(language_code="en", device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device)

    # Check for swear words
    print("\n" + "="*70)
    print("TRANSCRIPTION RESULTS")
    print("="*70)

    found_swears = []
    total_words = 0

    for segment in result["segments"]:
        print(f"\n[{segment['start']:.2f} - {segment['end']:.2f}]: {segment['text']}")

        if "words" in segment:
            for word_info in segment["words"]:
                word = word_info["word"].lower().strip()
                total_words += 1

                # Check if this is a swear word
                if any(swear in word for swear in swear_words):
                    found_swears.append({
                        "word": word,
                        "time": word_info["start"],
                        "segment": segment['text']
                    })
                    print(f"  ⚠️  SWEAR WORD FOUND: '{word}' at {word_info['start']:.2f}s")

    # Print results
    print("\n" + "="*70)
    print("VERIFICATION RESULTS")
    print("="*70)

    if found_swears:
        print(f"\n❌ FAILED: Found {len(found_swears)} swear word(s) in audio!")
        print("\nSwear words detected:")
        for swear in found_swears:
            print(f"  - '{swear['word']}' at {swear['time']:.2f}s in: \"{swear['segment']}\"")
        print("\n⚠️  The replacements are NOT working correctly!")
        return result, found_swears, False
    else:
        print(f"\n✅ SUCCESS: No swear words found in audio!")
        print(f"   Transcribed {total_words} total words, all clean.")
        print("\n🎉 The replacements are working correctly!")
        return result, found_swears, True


def main():
    parser = argparse.ArgumentParser(description='Verify edited audio for swear words')
    parser.add_argument('--audio', required=True, help='Audio file to verify')
    parser.add_argument('--swear-words', default='../word_lists.json',
                       help='Path to swear words JSON')
    parser.add_argument('--output', help='Save transcription to file')
    parser.add_argument('--device', default='cpu', help='Device (cpu/cuda)')

    args = parser.parse_args()

    result, found_swears, success = verify_audio(
        args.audio,
        args.swear_words,
        device=args.device
    )

    # Save transcription if requested
    if args.output:
        with open(args.output, "w") as f:
            f.write("Transcription Results:\n")
            f.write("="*70 + "\n\n")
            for segment in result["segments"]:
                f.write(f"[{segment['start']:.2f} - {segment['end']:.2f}]: {segment['text']}\n")
        print(f"\n✓ Saved transcription to {args.output}")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
