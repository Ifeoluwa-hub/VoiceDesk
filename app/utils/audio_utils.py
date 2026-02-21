from pathlib import Path
from typing import List

from app.models.transcript import Segment, SpeakerStat

ALLOWED_EXTENSIONS = {".wav", ".mp3"}
MAX_FILE_SIZE_MB = 200


def validate_audio_file(filename: str, file_size: int) -> None:
    """Validate audio file extension and size. Raises ValueError if invalid."""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported format: {ext}. Only .wav and .mp3 files are supported."
        )
    if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise ValueError(
            f"File too large. Maximum allowed size is {MAX_FILE_SIZE_MB}MB."
        )


def calculate_speaker_stats(segments: List[Segment]) -> List[SpeakerStat]:
    """Calculate speaking time statistics per speaker."""
    speaker_durations: dict = {}

    for seg in segments:
        duration = seg.end - seg.start
        if seg.speaker not in speaker_durations:
            speaker_durations[seg.speaker] = {"duration": 0.0, "count": 0}
        speaker_durations[seg.speaker]["duration"] += duration
        speaker_durations[seg.speaker]["count"] += 1

    total_duration = sum(d["duration"] for d in speaker_durations.values())
    if total_duration == 0:
        total_duration = 1.0

    stats = []
    for speaker, data in sorted(speaker_durations.items()):
        stats.append(
            SpeakerStat(
                speaker=speaker,
                total_duration_seconds=round(data["duration"], 2),
                percentage=round((data["duration"] / total_duration) * 100, 1),
                utterance_count=data["count"],
            )
        )
    return stats
