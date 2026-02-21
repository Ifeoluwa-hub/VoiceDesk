from pydantic import BaseModel
from typing import List
from datetime import datetime


class TranscriptSegment(BaseModel):
    speaker: str
    text: str
    start_time: float
    end_time: float


class UtteranceSentiment(BaseModel):
    speaker: str
    text: str
    start_time: float
    end_time: float
    sentiment: str
    sentiment_score: float


class SpeakerStats(BaseModel):
    speaker: str
    total_duration_seconds: float
    percentage: float
    utterance_count: int


class AnalysisResponse(BaseModel):
    report_id: str
    report_path: str
    audio_filename: str
    duration_seconds: float
    num_speakers: int
    processed_at: datetime
    summary: str
    overall_sentiment: str
    recommendations: List[str]
    transcript: List[TranscriptSegment]
    speaker_stats: List[SpeakerStats]
    sentiment_analysis: List[UtteranceSentiment]


class ErrorResponse(BaseModel):
    detail: str
    error_code: str
