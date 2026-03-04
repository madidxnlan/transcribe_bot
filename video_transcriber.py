#!/usr/bin/env python3
"""
Video Transcription Bot — with Speaker Diarization
Transcribes video/audio files using OpenAI's Whisper model.
Identifies speakers using pyannote.audio.
Handles long videos (45+ minutes) and finds timestamps where articles are discussed.

Setup:
  1. pip install pyannote.audio torch
  2. Get a free HuggingFace token at https://huggingface.co
  3. Accept model terms at https://huggingface.co/pyannote/speaker-diarization-3.1
  4. Add HF_TOKEN=hf_xxx to your .env file (or pass --hf-token on CLI)
"""

import os
import sys
from pathlib import Path
import whisper
from datetime import datetime
import argparse
import json
from typing import List, Dict, Optional
import re
from dotenv import load_dotenv

load_dotenv()


class VideoTranscriber:
    def __init__(self, model_size="base", hf_token: Optional[str] = None, speaker_names: Optional[Dict[str, str]] = None):
        """
        Initialize the transcriber with a Whisper model and optional speaker diarization.

        Args:
            model_size: One of ["tiny", "base", "small", "medium", "large"]
            hf_token: HuggingFace token for speaker diarization (or set HF_TOKEN in .env)
            speaker_names: Map speaker labels to real names
                           e.g. {"SPEAKER_00": "Rob", "SPEAKER_01": "Ryan", "SPEAKER_02": "Anna"}
        """
        print(f"Loading Whisper {model_size} model...")
        self.model = whisper.load_model(model_size)
        print("Model loaded successfully!")

        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.speaker_names = speaker_names or {}
        self.diarization_pipeline = None

        if self.hf_token:
            self._load_diarization_pipeline()
        else:
            print("⚠️  No HuggingFace token found — speaker diarization disabled.")
            print("   Add HF_TOKEN to your .env file to enable speaker labels.")

    def _load_diarization_pipeline(self):
        """Load the pyannote speaker diarization pipeline."""
        try:
            from pyannote.audio import Pipeline
            print("Loading speaker diarization model...")
            self.diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=self.hf_token
            )
            print("Speaker diarization ready!")
        except ImportError:
            print("⚠️  pyannote.audio not installed. Run: pip install pyannote.audio")
            self.diarization_pipeline = None
        except Exception as e:
            print(f"⚠️  Could not load diarization model: {e}")
            print("   Make sure you've accepted the model terms at:")
            print("   https://huggingface.co/pyannote/speaker-diarization-3.1")
            self.diarization_pipeline = None

    def _run_diarization(self, audio_path: str) -> List[Dict]:
        """
        Run speaker diarization on audio file.
        Returns list of {start, end, speaker} dicts.
        """
        if not self.diarization_pipeline:
            return []

        print("Identifying speakers... (this runs alongside transcription)")
        try:
            diarization = self.diarization_pipeline(audio_path)
            turns = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                # Map to real name if provided
                display_name = self.speaker_names.get(speaker, speaker)
                turns.append({
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": display_name,
                    "raw_label": speaker
                })
            print(f"Found {len(set(t['speaker'] for t in turns))} speakers")
            return turns
        except Exception as e:
            print(f"⚠️  Diarization failed: {e}")
            return []

    def _assign_speaker(self, segment_start: float, segment_end: float, turns: List[Dict]) -> str:
        """Find which speaker owns the majority of a transcript segment."""
        if not turns:
            return None

        seg_mid = (segment_start + segment_end) / 2

        # First try: find turn that contains the midpoint
        for turn in turns:
            if turn["start"] <= seg_mid <= turn["end"]:
                return turn["speaker"]

        # Fallback: find closest turn
        closest = min(turns, key=lambda t: min(
            abs(t["start"] - seg_mid),
            abs(t["end"] - seg_mid)
        ))
        return closest["speaker"]

    def transcribe_video(self, video_path, output_dir=None, language=None, articles_file=None):
        """
        Transcribe a video file with speaker labels and optionally find article mentions.

        Args:
            video_path: Path to the video file
            output_dir: Directory to save transcription (defaults to same dir as video)
            language: Language code (e.g., 'en') or None for auto-detect
            articles_file: Path to text file with article URLs/titles (one per line)

        Returns:
            Path to the transcription file
        """
        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        print(f"\n{'='*60}")
        print(f"Transcribing: {video_path.name}")
        print(f"{'='*60}\n")

        # Run diarization first (can run before transcription on same file)
        speaker_turns = self._run_diarization(str(video_path))

        # Transcribe with Whisper
        print("Processing audio... (this may take a while for long videos)")

        options = {
            "verbose": False,
            "task": "transcribe"
        }
        if language:
            options["language"] = language

        result = self.model.transcribe(str(video_path), **options)

        # Assign speakers to each Whisper segment
        segments_with_speakers = []
        for seg in result["segments"]:
            speaker = self._assign_speaker(seg["start"], seg["end"], speaker_turns)
            segments_with_speakers.append({
                **seg,
                "speaker": speaker
            })

        # Prepare output directory
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = video_path.parent

        # Generate output filename
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = video_path.stem

        # ── SAVE PLAIN TEXT (no speakers, existing behavior) ──
        txt_path = output_dir / f"{base_name}_transcript_{timestamp_str}.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"Transcription of: {video_path.name}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Language: {result.get('language', 'auto-detected')}\n")
            f.write(f"\n{'='*60}\n\n")
            f.write(result['text'])

        # ── SAVE SPEAKER-LABELED TRANSCRIPT ──
        speaker_txt_path = output_dir / f"{base_name}_speakers_{timestamp_str}.txt"
        self._save_speaker_transcript(segments_with_speakers, speaker_txt_path, video_path.name, result)

        # ── SAVE SRT (with speaker labels in subtitle text) ──
        srt_path = output_dir / f"{base_name}_transcript_{timestamp_str}.srt"
        self._save_srt(segments_with_speakers, srt_path)

        # ── SAVE JSON ──
        json_path = output_dir / f"{base_name}_transcript_{timestamp_str}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                **result,
                "segments": segments_with_speakers,
                "speaker_turns": speaker_turns
            }, f, indent=2, ensure_ascii=False)

        # ── FIND ARTICLES ──
        article_matches = None
        if articles_file:
            articles = self._load_articles(articles_file)
            if articles:
                article_matches = self._find_articles_in_transcript(articles, segments_with_speakers)
                article_report_path = output_dir / f"{base_name}_article_timestamps_{timestamp_str}.txt"
                self._save_article_report(article_matches, article_report_path, video_path.name)

        # ── PRINT SUMMARY ──
        print(f"\n{'='*60}")
        print("Transcription complete!")
        print(f"{'='*60}")
        print(f"\nFiles saved:")
        print(f"  📄 Plain text:      {txt_path}")
        print(f"  👥 With speakers:   {speaker_txt_path}")
        print(f"  🎬 Subtitles:       {srt_path}")
        print(f"  📊 JSON data:       {json_path}")

        if article_matches:
            print(f"  🔍 Article timestamps: {article_report_path}")
            print(f"\n{'='*60}")
            print("ARTICLE MENTIONS SUMMARY:")
            print(f"{'='*60}")
            self._print_article_summary(article_matches)

        # Print speaker summary
        if speaker_turns:
            self._print_speaker_summary(segments_with_speakers)

        return speaker_txt_path  # Return the speaker-labeled version as primary

    def _save_speaker_transcript(self, segments: List[Dict], output_path: Path, video_name: str, result: Dict):
        """
        Save transcript with speaker labels in format:
        [00:04:32] ROB: text here...
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"Transcription of: {video_name}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Language: {result.get('language', 'auto-detected')}\n")
            f.write(f"\n{'='*60}\n\n")

            current_speaker = None

            for seg in segments:
                speaker = seg.get("speaker")
                text = seg["text"].strip()
                ts = self._format_readable_timestamp(seg["start"])

                if not text:
                    continue

                # Only print speaker label when speaker changes
                if speaker and speaker != current_speaker:
                    if current_speaker is not None:
                        f.write("\n")  # blank line between speaker turns
                    speaker_display = speaker.upper() if speaker else "UNKNOWN"
                    f.write(f"[{ts}] {speaker_display}:\n")
                    current_speaker = speaker
                elif not speaker:
                    # No diarization — just print timestamp
                    f.write(f"[{ts}]\n")
                    current_speaker = None

                f.write(f"  {text}\n")

    def _print_speaker_summary(self, segments: List[Dict]):
        """Print a summary of speaking time per speaker."""
        speaker_time = {}
        for seg in segments:
            speaker = seg.get("speaker") or "Unknown"
            duration = seg["end"] - seg["start"]
            speaker_time[speaker] = speaker_time.get(speaker, 0) + duration

        total = sum(speaker_time.values())
        if total == 0:
            return

        print(f"\n{'='*60}")
        print("SPEAKER BREAKDOWN:")
        print(f"{'='*60}")
        for speaker, secs in sorted(speaker_time.items(), key=lambda x: -x[1]):
            pct = (secs / total) * 100
            mins = int(secs // 60)
            print(f"  {speaker.upper():<20} {mins:>3}min  ({pct:.0f}%)")

    # ─────────────────────────────────────────────
    # ALL ORIGINAL METHODS BELOW — UNCHANGED
    # ─────────────────────────────────────────────

    def _load_articles(self, articles_file: str) -> List[Dict[str, str]]:
        articles_path = Path(articles_file)
        if not articles_path.exists():
            print(f"⚠️  Articles file not found: {articles_file}")
            return []

        with open(articles_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]

        print(f"\n📚 Loading {len(lines)} articles...")

        articles = []
        for line in lines:
            url = self._extract_url(line)
            title = self._extract_title_from_url(url) if url else line
            title = self._clean_title(title)
            if title:
                articles.append({
                    'title': title,
                    'url': url,
                    'search_terms': self._create_search_terms(title)
                })

        print(f"   ✓ Loaded {len(articles)} articles")
        return articles

    def _extract_url(self, text: str) -> str:
        md_match = re.search(r'\(https?://[^\)]+\)', text)
        if md_match:
            return md_match.group(0)[1:-1]
        url_match = re.search(r'https?://[^\s\)]+', text)
        if url_match:
            return url_match.group(0).rstrip('_')
        return None

    def _extract_title_from_url(self, url: str) -> str:
        if not url:
            return ""
        path = url.split('/')[-1]
        path = re.sub(r'\?.*$', '', path)
        path = re.sub(r'\.html?$', '', path)
        title = path.replace('-', ' ').replace('_', ' ')
        title = re.sub(r'\d{4}/\d{2}/\d{2}/', '', title)
        return title.strip()

    def _clean_title(self, title: str) -> str:
        title = re.sub(r'__', '', title)
        title = re.sub(r'\[|\]|\(|\)', '', title)
        title = ' '.join(title.split())
        return title.strip()

    def _find_articles_in_transcript(self, articles: List[Dict], segments: List[Dict]) -> Dict:
        print("\n🔍 Searching for article mentions...")
        matches = {}
        found_count = 0

        for article in articles:
            article_matches = []
            search_terms = article['search_terms']

            for segment in segments:
                segment_text = segment['text'].lower()
                for term in search_terms:
                    if len(term) >= 3 and term.lower() in segment_text:
                        article_matches.append({
                            'start': segment['start'],
                            'end': segment['end'],
                            'text': segment['text'].strip(),
                            'matched_term': term,
                            'speaker': segment.get('speaker')
                        })
                        break

            matches[article['title']] = {
                'url': article['url'],
                'mentions': article_matches
            }
            if article_matches:
                found_count += 1

        print(f"   ✓ Found mentions for {found_count}/{len(articles)} articles")
        return matches

    def _create_search_terms(self, article: str) -> List[str]:
        terms = []
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                       'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during'}
        words = article.lower().split()
        filtered_words = [w for w in words if w not in common_words and len(w) > 2]
        if filtered_words:
            terms.append(' '.join(filtered_words))
        for word in filtered_words:
            if len(word) >= 4:
                terms.append(word)
        if len(filtered_words) >= 2:
            for i in range(len(filtered_words) - 1):
                terms.append(' '.join(filtered_words[i:i+2]))
        return list(set(terms))

    def _save_article_report(self, matches: Dict, output_path: Path, video_name: str):
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"Article Timestamp Report\n")
            f.write(f"Video: {video_name}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"\n{'='*80}\n\n")

            found = [(title, data) for title, data in matches.items() if data['mentions']]
            not_found = [(title, data) for title, data in matches.items() if not data['mentions']]
            found.sort(key=lambda x: x[1]['mentions'][0]['start'])

            if found:
                f.write("ARTICLES MENTIONED:\n")
                f.write("="*80 + "\n\n")
                for title, data in found:
                    mentions = data['mentions']
                    f.write(f"📰 {title}\n")
                    if data['url']:
                        f.write(f"   🔗 {data['url']}\n")
                    f.write(f"   Found {len(mentions)} mention(s)\n\n")
                    for i, mention in enumerate(mentions, 1):
                        timestamp = self._format_readable_timestamp(mention['start'])
                        speaker = mention.get('speaker')
                        speaker_label = f" [{speaker.upper()}]" if speaker else ""
                        f.write(f"   [{i}] {timestamp}{speaker_label}\n")
                        f.write(f"       \"{mention['text']}\"\n\n")
                    f.write("-" * 80 + "\n\n")

            if not_found:
                f.write("\n" + "="*80 + "\n")
                f.write("ARTICLES NOT MENTIONED:\n")
                f.write("="*80 + "\n\n")
                for title, data in not_found:
                    f.write(f"❌ {title}\n")
                    if data['url']:
                        f.write(f"   🔗 {data['url']}\n")
                    f.write("\n")

            if found:
                f.write("\n" + "="*80 + "\n")
                f.write("QUICK REFERENCE - JUMP TO TIMESTAMPS\n")
                f.write("="*80 + "\n\n")
                for title, data in found:
                    timestamps = [self._format_readable_timestamp(m['start']) for m in data['mentions']]
                    f.write(f"{title}:\n")
                    f.write(f"  {', '.join(timestamps)}\n\n")

    def _print_article_summary(self, matches: Dict):
        found = [(title, data) for title, data in matches.items() if data['mentions']]
        not_found = [(title, data) for title, data in matches.items() if not data['mentions']]

        if found:
            found.sort(key=lambda x: x[1]['mentions'][0]['start'])
            print("\n✅ MENTIONED:")
            for title, data in found:
                mentions = data['mentions']
                timestamps = [self._format_readable_timestamp(m['start']) for m in mentions[:3]]
                if len(mentions) > 3:
                    timestamps.append(f"... +{len(mentions)-3} more")
                print(f"   📰 {title}")
                print(f"      {', '.join(timestamps)}")

        if not_found:
            print(f"\n❌ NOT MENTIONED ({len(not_found)} articles):")
            for title, data in not_found[:5]:
                print(f"   • {title}")
            if len(not_found) > 5:
                print(f"   ... and {len(not_found)-5} more")

    def _format_readable_timestamp(self, seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    def _save_srt(self, segments, output_path):
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(segments, start=1):
                start = self._format_timestamp(segment['start'])
                end = self._format_timestamp(segment['end'])
                speaker = segment.get('speaker')
                text = segment['text'].strip()
                # Prepend speaker name to subtitle text if available
                if speaker:
                    text = f"[{speaker.upper()}] {text}"
                f.write(f"{i}\n{start} --> {end}\n{text}\n\n")

    def _format_timestamp(self, seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def transcribe_folder(self, folder_path, output_dir=None, extensions=None, articles_file=None):
        if extensions is None:
            extensions = ['.mov', '.mp4', '.m4v', '.avi', '.mkv', '.webm']

        folder_path = Path(folder_path)
        video_files = []
        for ext in extensions:
            video_files.extend(folder_path.glob(f"*{ext}"))
            video_files.extend(folder_path.glob(f"*{ext.upper()}"))

        if not video_files:
            print(f"No video files found in {folder_path}")
            return

        print(f"\nFound {len(video_files)} video file(s) to transcribe")
        for i, video_file in enumerate(video_files, 1):
            print(f"\n[{i}/{len(video_files)}]")
            try:
                self.transcribe_video(video_file, output_dir, articles_file=articles_file)
            except Exception as e:
                print(f"Error transcribing {video_file.name}: {e}")
                continue


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe videos with speaker labels and find article mentions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Transcribe with speaker detection
  python video_transcriber.py podcast_ep17.mov --speakers "Rob,Ryan,Anna"

  # Transcribe with articles + speakers
  python video_transcriber.py podcast.mov --articles articles.txt --speakers "Rob,Ryan,Guest"

  # Better accuracy
  python video_transcriber.py podcast.mov --model medium --speakers "Rob,Ryan"

  # Just transcribe without speakers (no HF token needed)
  python video_transcriber.py myvideo.mov

  # Batch folder
  python video_transcriber.py /path/to/videos --folder --speakers "Rob,Ryan,Anna"
        """
    )

    parser.add_argument('path', help='Path to video file or folder')
    parser.add_argument('--articles', '-a',
                       help='Path to text file with article URLs/titles (one per line)')
    parser.add_argument('--model', '-m',
                       choices=['tiny', 'base', 'small', 'medium', 'large'],
                       default='base',
                       help='Whisper model size (default: base)')
    parser.add_argument('--output', '-o',
                       help='Output directory for transcriptions')
    parser.add_argument('--language', '-l',
                       help='Language code (e.g., en, es, fr)')
    parser.add_argument('--folder', '-f',
                       action='store_true',
                       help='Process all videos in the specified folder')
    parser.add_argument('--speakers', '-s',
                       help='Speaker names in order, comma-separated: "Rob,Ryan,Anna"')
    parser.add_argument('--hf-token',
                       help='HuggingFace token (or set HF_TOKEN in .env)')

    args = parser.parse_args()

    # Build speaker name map from comma-separated list
    speaker_names = {}
    if args.speakers:
        for i, name in enumerate(args.speakers.split(",")):
            name = name.strip()
            if name:
                speaker_names[f"SPEAKER_{i:02d}"] = name

    # Initialize transcriber
    transcriber = VideoTranscriber(
        model_size=args.model,
        hf_token=args.hf_token,
        speaker_names=speaker_names
    )

    # Process
    if args.folder:
        transcriber.transcribe_folder(args.path, args.output, articles_file=args.articles)
    else:
        transcriber.transcribe_video(args.path, args.output, args.language, args.articles)


if __name__ == "__main__":
    main()
