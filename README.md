# vegF1 - Formula 1 Video Profanity Replacement

Automatically replaces profanity in F1 videos with vegetable names using AI speech synthesis.

## Origin Story

This project was inspired by a comment in the [Missed Apex podcast](https://www.youtube.com/watch?v=NHZGBrNTMRM&t=1562s):

> "Surely AI can replace the words with random fruits and vegetables"
>
> *— Missed Apex Podcast, 26:02*

This project takes a random comment and thru the power of laziness and delayed project time makes it real: automatically replace profanity with vegetable names using AI.

The solution: replace every swear word with a random vegetable name, spoken in a natural voice. "What the fuck?" becomes "What the butternut squash?" The result is absurd and family-friendly?

It's terrible, but it's our terrible.

## Examples

**Try it yourself with these F1 team radio clips:**

```bash
# Example 1: Kevin Magnussen door incident (22 replacements)
uv run python dl-video.py "https://www.youtube.com/watch?v=tnY65NRvwUQ"

# Example 2: Classic team radio moments
uv run python dl-video.py "https://www.youtube.com/watch?v=5h4FDy9Eqzs"
```

**Example Replacements:**
- "fucking blind asshole" → "broccoli... **blind**... kale" *(preserves non-swears!)*
- "fucking motherfucker" → "bamboo shoots, fenugreek, chard"
- "fucking dickhead" → "collard greens, garlic"
- "fuck sake" → "rhubarb **sake**" *(preserves "sake"!)*

## Quick Start

```bash
# Download and process a video
uv run python dl-video.py [VIDEO_URL]

# The script will:
# 1. Download video and extract audio
# 2. Transcribe with WhisperX (word-level timing)
# 3. Replace profanity with vegetable names (TTS)
# 4. Merge edited audio back with video
# 5. Output: final_output.mp4
```

## Project Structure

```
vegF1/
├── dl-video.py              # Main entry point
├── timeline_processor.py    # Core audio processing library
├── word_lists.json          # Swear words and vegetable names
├── final_output.mp4         # Latest generated video
│
├── scripts/                 # Utility scripts
│   ├── compare.py          # Unified comparison tool
│   ├── verify.py           # Verification tool (transcribe + check)
│   ├── create_video.py     # Video creation tool
│   └── test_*.py           # Test scripts
│
├── outputs/                 # All generated files
│   ├── *.wav               # Audio files
│   ├── *.png               # Visualizations
│   ├── *.log               # Replacement logs
│   └── *.txt               # Transcripts
│
├── docs/                    # Documentation
│   ├── REFACTOR_SUMMARY.md
│   └── improvements_summary.md
│
├── archive/                 # Old/deprecated files
│   └── audio_processor.py  # Original processor
│
├── audio/                   # Downloaded audio files
├── videos/                  # Downloaded video files
└── tts_cache/              # Cached TTS audio
```

## Consolidated Tools

### 1. Compare Audio/Visualizations

**Replaces:** `create_aligned_comparison.py`, `create_proper_comparison.py`, `create_wordlevel_comparison.py`, `enhanced_waveform_comparison.py`

```bash
cd scripts

# Full comparison
uv run python compare.py \
  --original ../audio/original.wav \
  --edited ../outputs/edited_audio.wav \
  --output ../outputs/comparison.png

# Zoomed comparison (specific time range)
uv run python compare.py \
  --original ../audio/original.wav \
  --edited ../outputs/edited_audio.wav \
  --output ../outputs/comparison_zoomed.png \
  --zoom-start 4.5 \
  --zoom-end 8.0

# With transcript comparison
uv run python compare.py \
  --original ../audio/original.wav \
  --edited ../outputs/edited_audio.wav \
  --transcript-original ../outputs/transcript.txt \
  --transcript-edited ../outputs/edited_transcript.txt \
  --transcript-output ../outputs/transcript_comparison.txt
```

### 2. Verify Output

**Replaces:** `verify_with_transcription.py`, `check_edited.py`

```bash
cd scripts

# Verify edited audio has no swear words
uv run python verify.py \
  --audio ../outputs/edited_audio.wav \
  --swear-words ../word_lists.json \
  --output ../outputs/verification_transcript.txt

# Exit code 0 = success (no swears found)
# Exit code 1 = failure (swears found)
```

### 3. Create Video

**Replaces:** `create_final_video.py`

```bash
cd scripts

# Create video from processed audio
uv run python create_video.py \
  --video ../videos/video.mp4 \
  --audio ../audio/audio.wav \
  --segments segments.json \
  --output ../outputs/final_output.mp4 \
  --debug
```

## How It Works

### Timeline-Based Processing

The core of this project is a **declarative timeline-based architecture** that:

1. **Phrase Detection**: Groups swear words within 2000ms into phrases
   - Example: "Fucking hell" with 1.5s gap → single phrase replacement

2. **Word-Level Replacement**: Only replaces profanity + 400ms buffer
   - "Kevin, just **fucking** smash the door" → "Kevin, just **[vegetables]** smash the door"
   - NOT: "**[vegetables]**" (which segment-level would do)

3. **Multiple Vegetables**: Fills longer phrases with multiple vegetables
   - "Fucking hell" (2.4s) → "butternut squash, broccolini, turnip" (2.3s)
   - Natural timing with 95%+ duration matching

4. **Punctuation & Variants**: Handles "hell." vs "hell", "fuckin" vs "fucking"

### Processing Pipeline

```
Input Video
    ↓
Extract Audio (WAV)
    ↓
Transcribe with WhisperX (word-level timing)
    ↓
Create Timeline (keep/replace segments)
    ↓
Generate TTS for vegetables
    ↓
Build edited audio from timeline
    ↓
Merge audio + video with ffmpeg
    ↓
Output Video
```

## Configuration

### word_lists.json

```json
{
  "swear_words": ["fuck", "shit", "hell", ...],
  "vegetable_names": ["bamboo shoots", "brussels sprouts", ...]
}
```

### Timeline Processor Settings

See `timeline_processor.py`:
- `safety_buffer_ms`: 400ms (buffer after each swear)
- `phrase_gap_threshold`: 2000ms (max gap to group swears)
- `stretch_limit`: 0.7-1.5x (TTS speed adjustment range)

## Development

### Running Tests

```bash
cd scripts
uv run python test_timeline_with_words.py
```

### Adding New Vegetables

Edit `word_lists.json` and add to `vegetable_names` array.

### Debugging

1. Enable debug mode: `--debug` flag
2. Check outputs in `outputs/` folder
3. Review replacement log: `outputs/*_replacements.log`
4. Verify with: `scripts/verify.py --audio outputs/edited_audio.wav`

## Results

- **10+ replacements** per typical F1 video
- **95%+ duration matching** (sounds natural)
- **0 swear words** in output (verified by transcription)
- **22% time saved** (removed profanity + buffers)

## License

MIT
