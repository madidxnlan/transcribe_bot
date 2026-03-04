How to run: cd "/Volumes/Crucial X10/podbud/transcribe_bot"
python3 video_transcriber.py "Episode17.mp4" --speakers "Rob,Ryan,Anna,Alayna"


# Video Transcription Bot for Podcast Editing

Transcribe your QuickTime podcast videos and automatically find the timestamps where specific articles are mentioned. Perfect for making your editing workflow faster!

## What This Does

1. **Transcribes** your podcast videos (even 45+ minutes long)
2. **Finds timestamps** where you discuss specific articles
3. **Shows which articles WEREN'T mentioned** (so you know what to cut or reschedule)
4. **Gives you jump points** so you can quickly navigate to each article mention for editing

No more scrubbing through 45 minutes of footage looking for that one article discussion!

## Quick Setup

### 1. Install FFmpeg (required)

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
Download from https://ffmpeg.org/download.html and add to PATH

**Linux:**
```bash
sudo apt update
sudo apt install ffmpeg
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Basic: Transcribe + Find Articles

```bash
python video_transcriber.py podcast_episode1.mov --articles my_articles.txt
```

This creates:
- Full transcript
- SRT subtitles file
- **Article timestamp report** showing what WAS and WASN'T mentioned

### Create Your Articles File

Just paste your article URLs (one per line). The format you get them in works perfectly:

**my_articles.txt:**
```
https://llm-politics.foaster.ai
* https://www.theinformation.com/articles/openai-ceo-declares-code-red
* https://www.nytimes.com/2025/11/25/us/politics/ai-super-pac-anthropic.html
* https://techcrunch.com/2025/12/02/amazon-releases-ai-chip
```

The bot automatically extracts titles from URLs, so just paste your list and go!

### Example Output

When you run it, you'll get a report like this:

```
✅ MENTIONED:
   📰 openai ceo declares code red combat threats chatgpt
      03:45, 28:12
   
   📰 amazon releases impressive new ai chip teases nvidia friendly roadmap
      12:30, 15:45, 32:10

❌ NOT MENTIONED (2 articles):
   • llm politics foaster
   • ai super pac anthropic
```

Perfect for:
- Jumping to mentioned articles in your editor
- Knowing which articles to cut from your notes
- Planning follow-up episodes with unused material

## Advanced Usage

### Better Accuracy (for important episodes)

```bash
python video_transcriber.py podcast.mov --articles articles.txt --model medium
```

Models from fastest to most accurate:
- `tiny` - Super fast, less accurate
- `base` - **Default**, good balance
- `small` - Better accuracy
- `medium` - High accuracy (recommended for final edits)
- `large` - Best accuracy (slow, needs good GPU)

### Process Multiple Videos

```bash
python video_transcriber.py /path/to/videos --folder --articles articles.txt
```

### Specify Output Location

```bash
python video_transcriber.py podcast.mov --articles articles.txt --output ./transcripts
```

### Just Transcribe (No Article Finding)

```bash
python video_transcriber.py podcast.mov
```

## Output Files

For each video, you get:

1. **`*_transcript_*.txt`** - Full transcription in plain text
2. **`*_transcript_*.srt`** - Timestamped subtitles (can import into video editors)
3. **`*_transcript_*.json`** - Complete data with all timestamps
4. **`*_article_timestamps_*.txt`** - **YOUR EDITING GUIDE** with:
   - Articles mentioned with timestamps
   - Articles NOT mentioned (helps you know what to cut)

## The Article Report Format

```
ARTICLES MENTIONED:
================================================================================

📰 openai ceo declares code red combat threats chatgpt
   🔗 https://www.theinformation.com/articles/openai-ceo-declares...
   Found 2 mention(s)

   [1] 03:45
       "So I read this article about OpenAI's code red situation..."

   [2] 28:12
       "Going back to that OpenAI thing we mentioned earlier..."

--------------------------------------------------------------------------------

ARTICLES NOT MENTIONED:
================================================================================

❌ llm politics foaster
   🔗 https://llm-politics.foaster.ai

❌ ai super pac anthropic
   🔗 https://www.nytimes.com/2025/11/25/us/politics/ai-super-pac...

================================================================================
QUICK REFERENCE - JUMP TO TIMESTAMPS
================================================================================

openai ceo declares code red combat threats chatgpt:
  03:45, 28:12
```

## Tips for Best Results

### Article URL Format

The bot handles:
- Plain URLs: `https://example.com/article`
- Markdown links: `[text](https://example.com/article)`
- Bullet points: `* https://example.com/article`
- Underscores: `__https://example.com/article__`

Just paste your list as-is!

### For 45+ Minute Videos

- `base` model works great and is reasonably fast
- First run downloads the model (~150MB) - subsequent runs are faster
- If you have an NVIDIA GPU, it'll automatically use it for speed

### Workflow Tips

1. Download your podcast video
2. Paste your article URLs into a text file
3. Run: `python video_transcriber.py video.mov --articles urls.txt`
4. Check the report to see:
   - Which articles were discussed (with timestamps)
   - Which articles weren't mentioned (cut from your notes/prep)
5. Jump to those timestamps in your video editor
6. Done editing in a fraction of the time!

## Supported Video Formats

- QuickTime (.mov) ← Your main format
- MP4 (.mp4)
- M4V (.m4v)
- AVI (.avi)
- MKV (.mkv)
- WebM (.webm)

## Troubleshooting

**"FFmpeg not found"**
- Make sure FFmpeg is installed and in your PATH
- Test with: `ffmpeg -version`

**Articles not found**
- Some articles might not be mentioned - that's normal!
- The bot will clearly show which ones were and weren't mentioned
- Check the "NOT MENTIONED" section to see what got skipped

**Slow transcription**
- `base` model is the sweet spot
- `tiny` model is faster but less accurate
- If you have an NVIDIA GPU, it'll automatically speed things up

## Why This Helps Your Workflow

**Before:** 
- Scrub through 45 minutes of video looking for each article
- Guess which articles you actually discussed
- Waste time editing sections for articles you never mentioned

**After:**
- See exactly when each article is mentioned
- Know immediately which articles to cut from your notes
- Jump straight to the relevant timestamps
- Edit in half the time

Most video editors (Premiere, Final Cut, DaVinci Resolve) can import SRT files directly for reference while you work!
