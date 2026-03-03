import asyncio
import logging
import os
import re
import subprocess
import tempfile
from typing import List, Optional, Tuple

from langchain_core.documents import Document

from app.processing.audio_processor import _transcribe, _group_segments
from app.processing.image_processor import analyze_image, create_vision_llm
from app.processing.text_cleaner import clean_transcription

logger = logging.getLogger(__name__)

SUPPORTED_VIDEO_FORMATS = {'.mp4', '.avi', '.mkv', '.mov', '.webm', '.flv', '.wmv'}
MAX_KEY_FRAMES = 15
SEGMENT_DURATION = 30
# Concurrency limit for Groq Vision API calls (avoid rate-limit 429s)
_VISION_CONCURRENCY = 3


def _extract_audio(video_path: str, output_path: str) -> bool:
    """Extract audio from video using FFmpeg."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-i", video_path, "-vn", "-acodec", "libmp3lame", "-q:a", "4", "-y", output_path],
            capture_output=True, text=True, timeout=300,
        )
        return result.returncode == 0 and os.path.exists(output_path)
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.error(f"Audio extraction failed: {e}")
        return False


def _get_duration(video_path: str) -> Optional[float]:
    """Get video duration via FFprobe."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True, timeout=30,
        )
        return float(result.stdout.strip()) if result.returncode == 0 else None
    except Exception:
        return None


def _extract_frames(video_path: str, output_dir: str) -> List[Tuple[float, str]]:
    """Extract key frames using FFmpeg scene detection, fallback to interval.

    Scene detection captures actual ``pts_time`` from FFmpeg's showinfo
    filter so timestamps reflect real scene changes, not arbitrary
    multiples of SEGMENT_DURATION.
    """
    frames = []

    try:
        result = subprocess.run(
            ["ffmpeg", "-i", video_path, "-vf", "select='gt(scene,0.3)',showinfo",
             "-vsync", "vfr", "-q:v", "3", os.path.join(output_dir, "scene_%04d.jpg"), "-y"],
            capture_output=True, text=True, timeout=300,
        )

        # Parse actual pts_time values from FFmpeg showinfo stderr
        pts_times: list[float] = []
        for line in result.stderr.splitlines():
            if "pts_time:" in line:
                m = re.search(r"pts_time:\s*([\d.]+)", line)
                if m:
                    pts_times.append(float(m.group(1)))

        for i in range(1, MAX_KEY_FRAMES + 1):
            path = os.path.join(output_dir, f"scene_{i:04d}.jpg")
            if os.path.exists(path):
                # Use real pts_time if available, fallback to index-based estimate
                ts = pts_times[i - 1] if i - 1 < len(pts_times) else i * SEGMENT_DURATION
                frames.append((ts, path))
    except Exception:
        pass

    if not frames:
        duration = _get_duration(video_path) or 300
        interval = max(SEGMENT_DURATION, duration / MAX_KEY_FRAMES)
        try:
            subprocess.run(
                ["ffmpeg", "-i", video_path, "-vf", f"fps=1/{int(interval)}", "-q:v", "3",
                 os.path.join(output_dir, "frame_%04d.jpg"), "-y"],
                capture_output=True, timeout=300,
            )
            ts = 0.0
            for i in range(1, MAX_KEY_FRAMES + 1):
                path = os.path.join(output_dir, f"frame_{i:04d}.jpg")
                if os.path.exists(path):
                    frames.append((ts, path))
                    ts += interval
                else:
                    break
        except Exception as e:
            logger.error(f"Frame extraction failed: {e}")

    return frames[:MAX_KEY_FRAMES]


async def process_video(
    video_path: str,
    groq_api_key: str,
    file_name: Optional[str] = None,
    course_id: Optional[str] = None,
    source_id: Optional[str] = None,
    analyze_frames: bool = True,
) -> List[Document]:
    """
    Full pipeline: FFmpeg audio → Groq Whisper, FFmpeg frames → ChatGroq Vision,
    timestamp-aligned merge → LangChain Documents.

    Frame analysis runs **concurrently** (up to ``_VISION_CONCURRENCY``
    parallel Groq Vision calls) instead of sequentially, cutting latency
    from 30-75 s to ~5-10 s for a typical 15-frame video.
    """
    file_name = file_name or os.path.basename(video_path)
    source = source_id or file_name
    documents = []

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, "audio.mp3")
        transcript = {"text": "", "segments": []}
        if _extract_audio(video_path, audio_path):
            transcript = await _transcribe(audio_path, groq_api_key)

        grouped = _group_segments(transcript.get("segments", []), SEGMENT_DURATION)

        # ── Parallel frame analysis ───────────────────────────────
        frame_analyses = []
        if analyze_frames:
            extracted = _extract_frames(video_path, tmpdir)
            if extracted:
                # One shared LLM instance for all frames (no per-call overhead)
                vision_llm = create_vision_llm(groq_api_key)
                sem = asyncio.Semaphore(_VISION_CONCURRENCY)

                async def _analyze_one(ts: float, fpath: str) -> Optional[dict]:
                    async with sem:
                        try:
                            with open(fpath, 'rb') as f:
                                desc = await analyze_image(
                                    f.read(), groq_api_key,
                                    prompt="This frame is from an educational video. Describe slides, diagrams, code, or whiteboard content shown.",
                                    llm=vision_llm,
                                )
                            if desc and "[Image analysis failed" not in desc:
                                return {"timestamp": ts, "content": desc}
                        except Exception as e:
                            logger.warning(f"Frame analysis failed at {ts}s: {e}")
                    return None

                results = await asyncio.gather(
                    *[_analyze_one(ts, fp) for ts, fp in extracted]
                )
                frame_analyses = [r for r in results if r is not None]
                # Sort by timestamp so merge with audio segments works correctly
                frame_analyses.sort(key=lambda x: x["timestamp"])

        if grouped:
            for seg in grouped:
                text = clean_transcription(seg["text"])
                content = f"[AUDIO] {text}" if text else ""

                visuals = [fa for fa in frame_analyses if seg["start"] <= fa["timestamp"] <= seg["end"]]
                if visuals:
                    content += "\n\n[VISUAL]\n" + "\n".join(v["content"] for v in visuals)

                if content.strip():
                    documents.append(Document(
                        page_content=content,
                        metadata={
                            "source_type": "video", "source_id": source, "file_name": file_name,
                            "course_id": course_id, "page_number": None,
                            "start_time": round(seg["start"], 1), "end_time": round(seg["end"], 1),
                            "contains_visual": bool(visuals),
                        }
                    ))
        elif frame_analyses:
            for fa in frame_analyses:
                documents.append(Document(
                    page_content=f"[VISUAL]\n{fa['content']}",
                    metadata={
                        "source_type": "video", "source_id": source, "file_name": file_name,
                        "course_id": course_id, "page_number": None,
                        "start_time": round(fa["timestamp"], 1), "end_time": None,
                        "contains_visual": True,
                    }
                ))
        elif transcript["text"]:
            text = clean_transcription(transcript["text"])
            if text:
                documents.append(Document(
                    page_content=f"[AUDIO] {text}",
                    metadata={
                        "source_type": "video", "source_id": source, "file_name": file_name,
                        "course_id": course_id, "page_number": None,
                        "start_time": 0, "end_time": transcript.get("duration"),
                        "contains_visual": False,
                    }
                ))

    logger.info(f"Processed video '{file_name}': {len(documents)} segments")
    return documents


async def process_video_bytes(
    video_bytes: bytes, groq_api_key: str,
    file_name: str = "video.mp4", course_id: Optional[str] = None, source_id: Optional[str] = None,
) -> List[Document]:
    """Process video from bytes."""
    ext = os.path.splitext(file_name)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name
    try:
        return await process_video(tmp_path, groq_api_key, file_name, course_id, source_id)
    finally:
        os.unlink(tmp_path)