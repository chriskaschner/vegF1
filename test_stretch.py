from pydub import AudioSegment
from audio_processor import time_stretch_audio, detect_silence

# Load the original lettuce audio
original = AudioSegment.from_file("tts_cache/lettuce.mp3")

# Create time-stretched version (target 260ms to match the example)
stretched = time_stretch_audio(original, 260)

# Detect silence in both versions
original_start, original_end = detect_silence(original, silence_threshold=-50.0, min_silence_len=10)
stretched_start, stretched_end = detect_silence(stretched, silence_threshold=-50.0, min_silence_len=10)

# Trim the silence from stretched version
stretched_trimmed = stretched[stretched_start:stretched_end]

# Save all versions for comparison
original.export("lettuce_original.mp3", format="mp3")
stretched.export("lettuce_stretched.mp3", format="mp3")
stretched_trimmed.export("lettuce_stretched_trimmed.mp3", format="mp3")

# Print detailed information
print("\nOriginal audio:")
print(f"- Total duration: {len(original)}ms")
print(f"- Content from {original_start}ms to {original_end}ms")
print(f"- Content duration: {original_end - original_start}ms")

print("\nStretched audio:")
print(f"- Total duration: {len(stretched)}ms")
print(f"- Content from {stretched_start}ms to {stretched_end}ms")
print(f"- Content duration: {stretched_end - stretched_start}ms")

print("\nTrimmed stretched audio:")
print(f"- Final duration: {len(stretched_trimmed)}ms") 