# Call Center QA Agent

An AI-powered quality assurance system for analyzing call center recordings. It automatically transcribes audio, identifies speakers, analyzes sentiment, and generates comprehensive PDF reports with actionable recommendations.

## Overview

The Call Center QA Agent processes call center recordings through a multi-stage pipeline:

1. **Transcription** - Converts speech to text using WhisperX (large-v2 model)
2. **Speaker Diarization** - Identifies and labels different speakers using Pyannote
3. **Sentiment Analysis** - Classifies each utterance as positive, neutral, or negative using GPT-4o
4. **Summary Generation** - Produces a concise summary of the conversation
5. **Recommendations** - Generates 3-5 actionable recommendations for agent performance improvement
6. **PDF Report** - Creates a professional, styled PDF report with charts and visualizations

---

## Architecture

```
                         +-------------------+
                         |   FastAPI Server   |
                         |    (main.py)       |
                         +---------+---------+
                                   |
                         +---------+---------+
                         |   /api/v1/analyze  |
                         |  (qa_report.py)    |
                         +---------+---------+
                                   |
              +--------------------+--------------------+
              |                    |                    |
    +---------v--------+ +--------v---------+ +--------v---------+
    |  Transcription   | |    Analysis      | | Report Generator |
    | (WhisperX +      | | (GPT-4o)         | | (ReportLab +     |
    |  Pyannote)       | |                  | |  Matplotlib)     |
    +------------------+ +------------------+ +------------------+
    | 1. Load Model    | | 1. Summary       | | 1. Header        |
    | 2. Transcribe    | | 2. Sentiment     | | 2. Summary       |
    | 3. Align         | | 3. Recommendations| | 3. Speaker Stats |
    | 4. Diarize       | |                  | | 4. Sentiment     |
    +------------------+ +------------------+ | 5. Transcript    |
                                              | 6. Recommendations|
                                              +------------------+
```

---

## Pipeline

```
Audio Upload (MP3/WAV) --> Validation (format, size <=200MB)
    --> WhisperX Transcription (large-v2, int8)
    --> Word-Level Alignment
    --> Speaker Diarization (Pyannote)
    --> Speaker Stats Calculation
    --> GPT-4o Summary Generation
    --> GPT-4o Sentiment Analysis (chunked, 25 utterances/chunk)
    --> GPT-4o Recommendations
    --> PDF Report Generation (charts + tables)
    --> JSON Response + Downloadable PDF
```

---

## API Endpoints

### Health Check

```
GET /api/v1/health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

### Analyze Audio

```
POST /api/v1/analyze
Content-Type: multipart/form-data
```

**Parameters:**

| Parameter    | Type       | Description                        |
|-------------|------------|------------------------------------|
| `audio_file` | UploadFile | WAV or MP3 file (max 200MB)        |

**Response:**

```json
{
  "report_id": "a1b2c3d4e5f6",
  "report_path": "reports/qa_report_a1b2c3d4e5f6.pdf",
  "audio_filename": "call_recording.mp3",
  "duration_seconds": 245.67,
  "num_speakers": 2,
  "processed_at": "2026-02-16T12:00:00Z",
  "summary": "The customer called regarding...",
  "overall_sentiment": "neutral",
  "recommendations": [
    "Improve greeting warmth...",
    "Confirm resolution before closing..."
  ],
  "transcript": [
    {
      "speaker": "SPEAKER_00",
      "text": "Hello, how can I help you today?",
      "start_time": 0.0,
      "end_time": 2.5
    }
  ],
  "speaker_stats": [
    {
      "speaker": "SPEAKER_00",
      "total_duration_seconds": 120.5,
      "percentage": 49.1,
      "utterance_count": 25
    }
  ],
  "sentiment_analysis": [
    {
      "speaker": "SPEAKER_00",
      "text": "Hello, how can I help you today?",
      "start_time": 0.0,
      "end_time": 2.5,
      "sentiment": "positive",
      "sentiment_score": 0.7
    }
  ]
}
```

### Download Report

```
GET /api/v1/reports/{report_id}
```

Returns the generated PDF report file.

---

## Project Structure

```
Agent/
├── main.py                          # FastAPI app entry point
├── config.py                        # Settings & environment config
├── setup.py                         # Package setup
├── requirements.txt                 # Python dependencies
├── constraints.txt                  # Dependency version constraints
├── .env                             # Environment variables (not in git)
├── .gitignore
├── app/
│   ├── __init__.py
│   ├── schemas.py                   # API request/response schemas
│   ├── models/
│   │   ├── __init__.py
│   │   └── transcript.py            # Internal data models
│   ├── routes/
│   │   ├── __init__.py
│   │   └── qa_report.py             # API endpoint handlers
│   ├── services/
│   │   ├── __init__.py
│   │   ├── transcription.py         # WhisperX transcription pipeline
│   │   ├── analysis.py              # GPT-4o analysis (summary, sentiment, recommendations)
│   │   └── report_generator.py      # PDF report generation
│   └── utils/
│       ├── __init__.py
│       └── audio_utils.py           # Audio file validation & speaker stats
├── reports/                         # Generated PDF reports (gitignored)
└── uploads/                         # Temporary audio uploads (gitignored)
```

---

## Installation

### Prerequisites

- Python 3.10+
- Conda (recommended)
- FFmpeg (for audio processing)
- OpenAI API key
- HuggingFace token (with access to pyannote/segmentation-3.0 and pyannote/speaker-diarization-3.1)

### Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Agent
   ```

2. **Create conda environment:**
   ```bash
   conda create -n qa-agent python=3.10 -y
   conda activate qa-agent
   conda install -c conda-forge ffmpeg -y
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Accept HuggingFace model licenses:**
   - Visit https://hf.co/pyannote/segmentation-3.0 and accept the license
   - Visit https://hf.co/pyannote/speaker-diarization-3.1 and accept the license

5. **Create `.env` file:**
   ```
   OPENAI_API_KEY=your_openai_api_key
   HF_AUTH_TOKEN=your_huggingface_token
   WHISPERX_MODEL_SIZE=large-v2
   WHISPERX_DEVICE=cpu
   WHISPERX_COMPUTE_TYPE=int8
   WHISPERX_BATCH_SIZE=4
   OPENAI_MODEL=gpt-4o
   ```

6. **Run the server:**
   ```bash
   python main.py
   ```
   Or with uvicorn:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

---

## Configuration

All settings are managed via environment variables (`.env` file):

| Variable               | Default     | Description                              |
|-----------------------|-------------|------------------------------------------|
| `OPENAI_API_KEY`       | -           | OpenAI API key for GPT-4o                |
| `HF_AUTH_TOKEN`        | -           | HuggingFace token for Pyannote models    |
| `WHISPERX_MODEL_SIZE`  | `large-v2`  | Whisper model size                       |
| `WHISPERX_DEVICE`      | `cpu`       | Device for inference (`cpu` or `cuda`)   |
| `WHISPERX_COMPUTE_TYPE`| `int8`      | Quantization type (`int8`, `float16`)    |
| `WHISPERX_BATCH_SIZE`  | `4`         | Transcription batch size                 |
| `OPENAI_MODEL`         | `gpt-4o`    | OpenAI model for analysis                |

---

## Usage

### Using cURL

```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -F "audio_file=@recording.mp3"
```

### Using Python

```python
import requests

url = "http://localhost:8000/api/v1/analyze"
files = {"audio_file": open("recording.mp3", "rb")}
response = requests.post(url, files=files)

data = response.json()
print(f"Summary: {data['summary']}")
print(f"Sentiment: {data['overall_sentiment']}")
print(f"Report: {data['report_path']}")
```

### Download PDF Report

```bash
curl -O http://localhost:8000/api/v1/reports/{report_id}
```

---

## PDF Report

The generated PDF report includes the following sections:

1. **Header** - Report metadata (filename, duration, number of speakers, date)
2. **Summary** - AI-generated conversation overview with overall sentiment badge
3. **Speaker Statistics** - Table and pie chart showing speaking time distribution per speaker
4. **Sentiment Analysis** - Sentiment trend line chart and table of top negative utterances
5. **Full Transcript** - Complete conversation transcript color-coded by speaker with timestamps
6. **Recommendations** - 3-5 actionable items for agent performance improvement

---

## Tech Stack

| Component             | Technology                                   |
|----------------------|----------------------------------------------|
| **Web Framework**     | FastAPI                                      |
| **Transcription**     | WhisperX (large-v2) + faster-whisper         |
| **Speaker Diarization** | Pyannote.audio                             |
| **AI Analysis**       | OpenAI GPT-4o                                |
| **PDF Generation**    | ReportLab + Matplotlib                       |
| **Audio Processing**  | torchaudio, soundfile                        |
| **Data Validation**   | Pydantic                                     |
| **ML Framework**      | PyTorch 2.1.2                                |

---

## Supported Audio Formats

- WAV (.wav)
- MP3 (.mp3)
- Maximum file size: 200MB
