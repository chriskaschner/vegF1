"""
Simple test to verify that word replacement is actually working.
Creates a minimal test case to isolate the replacement logic.
"""
import json
from pydub import AudioSegment
from audio_processor import replace_audio_segments

def test_basic_replacement():
    """Test that a swear word gets replaced in the audio."""

    # Load the actual audio and transcription
    audio_file = "audio/5h4FDy9Eqzs.wav"

    # Create a minimal segment with just one swear word
    # Using the "Fucking hell" segment from 4.79 - 7.20
    test_segments = [{
        "start": 4.79,
        "end": 7.20,
        "text": "Fucking hell.",
        "words": [
            {"word": "Fucking", "start": 4.795, "end": 5.055},
            {"word": "hell", "start": 6.92, "end": 7.20}
        ]
    }]

    # Simple replacement function that just returns a fixed vegetable
    def get_test_replacement(duration, word=None):
        return {
            "word": "TEST_CARROT",
            "length_ms": 300,
            "file": "tts_cache/carrot.mp3"
        }

    print("=" * 60)
    print("BASIC REPLACEMENT TEST")
    print("=" * 60)
    print(f"Input segment: {test_segments[0]['text']}")
    print(f"Swear word: 'Fucking' at {test_segments[0]['words'][0]['start']:.3f}s")
    print()

    # Run the replacement
    output_file, replacements_log = replace_audio_segments(
        audio_path=audio_file,
        segments=test_segments,
        swear_words=["fucking"],
        get_replacement_fn=get_test_replacement,
        apply_stretch=True,
        debug=True
    )

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    if replacements_log:
        print(f"✓ Replacement occurred!")
        for log in replacements_log:
            print(f"  Original: '{log['original_word']}' at {log['original_start']}-{log['original_end']}ms")
            print(f"  Replaced with: '{log['replacement_word']}' at {log['replacement_start']}-{log['replacement_end']}ms")
            print(f"  Duration: {log['original_duration']}ms -> {log['replacement_duration']}ms")
    else:
        print("✗ NO REPLACEMENTS - Something is broken!")

    # Check the output audio length
    if output_file:
        output_audio = AudioSegment.from_file(output_file)
        print(f"\n✓ Output audio created: {len(output_audio)}ms")

        # Expected: roughly 2.41 seconds (7.20 - 4.79)
        expected_duration = (7.20 - 4.79) * 1000
        print(f"  Expected ~{expected_duration:.0f}ms, got {len(output_audio)}ms")

        if abs(len(output_audio) - expected_duration) > 500:
            print(f"  ⚠ WARNING: Duration mismatch > 500ms")

    print()
    return replacements_log


if __name__ == "__main__":
    try:
        replacements = test_basic_replacement()

        print("\n" + "=" * 60)
        if replacements:
            print("✓ TEST PASSED: Basic replacement is working")
        else:
            print("✗ TEST FAILED: No replacements occurred")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ TEST FAILED WITH ERROR:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
