from pydub import AudioSegment
import os
from audio_processor import detect_silence

# List of common vegetables (we can expand this)
VEGETABLES = [
    "lettuce", "carrot", "celery", "broccoli", "spinach", "kale", "cucumber",
    "radish", "turnip", "beet", "pea", "bean", "squash", "yam", "potato",
    "onion", "garlic", "leek", "shallot", "parsnip", "rutabaga", "asparagus",
    "zucchini", "eggplant", "pepper", "tomato", "okra", "artichoke"
]

def analyze_audio_duration(file_path):
    """Analyze the actual content duration of an audio file (excluding silence)."""
    audio = AudioSegment.from_file(file_path)
    start, end = detect_silence(audio, silence_threshold=-50.0, min_silence_len=10)
    content_duration = end - start
    return content_duration

def find_suitable_replacements(target_duration, max_ratio=1.6):
    """Find vegetables whose TTS duration is within the acceptable ratio of target duration."""
    suitable_replacements = []
    
    for veg in VEGETABLES:
        tts_path = f"tts_cache/{veg}.mp3"
        if not os.path.exists(tts_path):
            continue
            
        duration = analyze_audio_duration(tts_path)
        ratio = max(duration / target_duration, target_duration / duration)
        
        if ratio <= max_ratio:
            suitable_replacements.append({
                'word': veg,
                'duration': duration,
                'ratio': ratio
            })
    
    # Sort by ratio (closest to 1.0 first)
    return sorted(suitable_replacements, key=lambda x: x['ratio'])

# Example target duration (from our previous test)
target_duration = 260  # ms

print(f"\nAnalyzing replacements for target duration: {target_duration}ms")
print(f"Maximum duration ratio: 1.6")
print("\nSuitable replacements:")
print("-" * 60)
print(f"{'Word':<15} {'Duration':<10} {'Ratio':<10}")
print("-" * 60)

suitable = find_suitable_replacements(target_duration)
for replacement in suitable:
    print(f"{replacement['word']:<15} {replacement['duration']:<10.0f} {replacement['ratio']:<10.2f}")

if not suitable:
    print("No suitable replacements found. You may need to generate TTS files first.") 