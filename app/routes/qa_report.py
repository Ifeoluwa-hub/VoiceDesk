import os
import logging
import traceback
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse

from config import settings
from app.schemas import AnalysisResponse, TranscriptSegment, UtteranceSentiment, SpeakerStats
from app.utils.audio_utils import validate_audio_file, calculate_speaker_stats
from app.services.transcription import transcribe_and_diarize
from app.services.analysis import ConversationAnalyzer
from app.services.report_generator import generate_pdf, cleanup_temp_files
from app.models.transcript import ReportData

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_audio(audio_file: UploadFile = File(...)):
    """Upload an audio file and receive a full QA analysis with PDF report."""

    # 1. Validate file
    filename = audio_file.filename or "unknown.wav"
    # print(filename)
    content = await audio_file.read()
    file_size = len(content)
    # print(file_size)

    try:
        validate_audio_file(filename, file_size)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 2. Save uploaded file
    ext = Path(filename).suffix.lower()
    upload_name = f"{uuid4().hex}{ext}"
    upload_path = os.path.abspath(os.path.join(settings.uploads_dir, upload_name))
    print(f"Upload path: {upload_path}, exists: {os.path.exists(settings.uploads_dir)}")

    try:
        with open(upload_path, "wb") as f:
            f.write(content)

        # 3. Transcribe and diarize
        logger.info("Starting transcription for: %s", filename)
        try:
            transcription = transcribe_and_diarize(upload_path)
        except Exception as e:
            logger.error("Transcription failed: %s", str(e))
            logger.error("Full traceback:\n%s", traceback.format_exc())
            print("=== FULL ERROR TRACEBACK ===")
            traceback.print_exc()
            print("=== END TRACEBACK ===")
            raise HTTPException(
                status_code=500,
                detail=f"Transcription failed: {str(e)}",
            )

        if not transcription.segments:
            raise HTTPException(
                status_code=422,
                detail="No speech detected in the audio file.",
            )

        # 4. Calculate speaker stats
        speaker_stats = calculate_speaker_stats(transcription.segments)

        # 5. Analyze with GPT
        logger.info("Starting GPT analysis...")
        print('Starting GPT analysis...')
        try:
            analyzer = ConversationAnalyzer()
            analysis = analyzer.analyze(transcription.segments)
        except Exception as e:
            logger.error("Analysis failed: %s", str(e))
            raise HTTPException(
                status_code=500,
                detail=f"Analysis failed: {str(e)}",
            )

        # 6. Generate PDF report
        report_id = uuid4().hex[:12]
        processed_at = datetime.now(timezone.utc).isoformat()

        report_data = ReportData(
            report_id=report_id,
            audio_filename=filename,
            duration_seconds=transcription.duration_seconds,
            num_speakers=transcription.num_speakers,
            processed_at=processed_at,
            transcription=transcription,
            analysis=analysis,
            speaker_stats=speaker_stats,
        )

        try:
            report_path, _ = generate_pdf(report_data)
        except Exception as e:
            logger.error("Report generation failed: %s", str(e))
            raise HTTPException(
                status_code=500,
                detail=f"Report generation failed: {str(e)}",
            )
        finally:
            cleanup_temp_files()

        # 7. Build response
        transcript_segments = [
            TranscriptSegment(
                speaker=seg.speaker,
                text=seg.text,
                start_time=seg.start,
                end_time=seg.end,
            )
            for seg in transcription.segments
        ]

        sentiment_items = [
            UtteranceSentiment(
                speaker=s.speaker,
                text=s.text,
                start_time=s.start_time,
                end_time=s.end_time,
                sentiment=s.sentiment,
                sentiment_score=s.sentiment_score,
            )
            for s in analysis.sentiment.utterance_sentiments
        ]

        speaker_stats_response = [
            SpeakerStats(
                speaker=s.speaker,
                total_duration_seconds=s.total_duration_seconds,
                percentage=s.percentage,
                utterance_count=s.utterance_count,
            )
            for s in speaker_stats
        ]

        return AnalysisResponse(
            report_id=report_id,
            report_path=report_path,
            audio_filename=filename,
            duration_seconds=transcription.duration_seconds,
            num_speakers=transcription.num_speakers,
            processed_at=datetime.now(timezone.utc),
            summary=analysis.summary,
            overall_sentiment=analysis.sentiment.overall_sentiment,
            recommendations=analysis.recommendations,
            transcript=transcript_segments,
            speaker_stats=speaker_stats_response,
            sentiment_analysis=sentiment_items,
        )

    finally:
        # 8. Clean up uploaded file
        if os.path.exists(upload_path):
            os.remove(upload_path)


@router.get("/reports/{report_id}")
async def download_report(report_id: str):
    """Download a generated PDF report by its ID."""
    filename = f"qa_report_{report_id}.pdf"
    filepath = os.path.join(settings.reports_dir, filename)

    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Report not found.")

    return FileResponse(
        filepath,
        media_type="application/pdf",
        filename=filename,
    )
