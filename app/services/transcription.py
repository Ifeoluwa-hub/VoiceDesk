import gc
import time
import logging
import warnings
warnings.filterwarnings('ignore')

try:
    import static_ffmpeg
    static_ffmpeg.add_paths()
except ImportError:
    pass

from config import settings
from app.models.transcript import Segment, TranscriptionResult
import whisperx

logger = logging.getLogger(__name__)

def transcribe_and_diarize(audio_path: str) -> TranscriptionResult:
    """Run the full WhisperX pipeline: transcribe, align, and diarize."""
    pipeline_start = time.time()
    print('started transcription')
    device = settings.whisperx_device
    compute_type = settings.whisperx_compute_type
    batch_size = settings.whisperx_batch_size
    model_size = 'large_v2'
    print(device, compute_type, batch_size, model_size)

    logger.info("Loading WhisperX model: %s on %s", model_size, device)
    print(audio_path)

    # Load model and transcribe
    import os
    step_start = time.time()
    print(">>> STEP 1: Loading whisper model...")
    model = whisperx.load_model(
        "large-v2",
        device,
        compute_type=compute_type,
    )
    print(f">>> STEP 1 COMPLETE: Model loaded ({time.time() - step_start:.1f}s)")

    print(f"Audio file exists: {os.path.exists(audio_path)}, path: {os.path.abspath(audio_path)}")
    audio = _load_audio_fallback(audio_path)

    step_start = time.time()
    print(">>> STEP 1b: Transcribing...")
    result = model.transcribe(audio, batch_size=batch_size)
    print(f">>> STEP 1b COMPLETE: Transcription done ({time.time() - step_start:.1f}s)")

    detected_language = result.get("language", "en")
    print(f"Detected language: {detected_language}")
    logger.info("Detected language: %s", detected_language)

    # Free model memory
    del model
    gc.collect()
    _clear_gpu_cache(device)

    # Step 2: Align for word-level timestamps
    step_start = time.time()
    print(f">>> STEP 2: Loading alignment model for '{detected_language}'...")
    logger.info("Aligning transcript...")
    model_a, metadata = whisperx.load_align_model(
        language_code=detected_language,
        device=device,
    )
    print(f">>> STEP 2a COMPLETE: Alignment model loaded ({time.time() - step_start:.1f}s)")

    step_start = time.time()
    result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )
    print(f">>> STEP 2b COMPLETE: Alignment done ({time.time() - step_start:.1f}s)")

    del model_a
    gc.collect()
    _clear_gpu_cache(device)

    # Step 3: Diarize - assign speakers
    step_start = time.time()
    print(">>> STEP 3: Loading diarization model...")
    logger.info("Running speaker diarization...")
    diarize_model = whisperx.DiarizationPipeline(
        use_auth_token=settings.hf_auth_token,
        device=device,
    )
    print(f">>> STEP 3a COMPLETE: Diarization model loaded ({time.time() - step_start:.1f}s)")

    step_start = time.time()
    diarize_segments = diarize_model(audio)
    result = whisperx.assign_word_speakers(diarize_segments, result)
    print(f">>> STEP 3b COMPLETE: Diarization done ({time.time() - step_start:.1f}s)")

    del diarize_model
    gc.collect()
    _clear_gpu_cache(device)

    # Step 4: Convert to internal data model
    segments = []
    for seg in result["segments"]:
        speaker = seg.get("speaker", "UNKNOWN")
        text = seg.get("text", "").strip()
        if not text:
            continue
        segments.append(
            Segment(
                speaker=speaker,
                text=text,
                start=seg.get("start", 0.0),
                end=seg.get("end", 0.0),
            )
        )

    # Calculate duration from audio array (loaded at 16kHz)
    duration = len(audio) / 16000.0

    # Count unique speakers
    speakers = set(s.speaker for s in segments)

    logger.info(
        "Transcription complete: %d segments, %d speakers, %.1fs duration",
        len(segments),
        len(speakers),
        duration,
    )

    print(f">>> TOTAL PIPELINE TIME: {time.time() - pipeline_start:.1f}s")

    return TranscriptionResult(
        segments=segments,
        duration_seconds=round(duration, 2),
        num_speakers=len(speakers),
        language=detected_language,
    )


def _load_audio_fallback(file: str, sr: int = 16000):
    """Load audio using torchaudio instead of ffmpeg subprocess."""
    import numpy as np
    import torchaudio
    waveform, sample_rate = torchaudio.load(file)
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    # Resample if needed
    if sample_rate != sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=sr)
        waveform = resampler(waveform)
    return waveform.squeeze().numpy().astype(np.float32)


def _clear_gpu_cache(device: str) -> None:
    """Clear GPU memory if using CUDA."""
    if device == "cuda":
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass
