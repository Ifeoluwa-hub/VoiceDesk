import json
import time
import logging
from typing import List

from openai import OpenAI

from config import settings
from app.models.transcript import (
    AnalysisResult,
    Segment,
    SentimentItem,
    SentimentResult,
)

logger = logging.getLogger(__name__)

MAX_TRANSCRIPT_CHARS = 24000
MAX_UTTERANCES_PER_CHUNK = 25


class ConversationAnalyzer:
    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model

    def analyze(self, segments: List[Segment]) -> AnalysisResult:
        """Run full analysis pipeline: summary, sentiment, recommendations."""
        logger.info("Starting conversation analysis with %d segments", len(segments))

        analysis_start = time.time()

        step_start = time.time()
        print(">>> ANALYSIS STEP 1: Generating summary...")
        summary = self._generate_summary(segments)
        print(f">>> ANALYSIS STEP 1 COMPLETE: Summary generated ({time.time() - step_start:.1f}s)")

        step_start = time.time()
        print(">>> ANALYSIS STEP 2: Analyzing sentiment...")
        sentiment = self._analyze_sentiment(segments)
        print(f">>> ANALYSIS STEP 2 COMPLETE: Sentiment analyzed ({time.time() - step_start:.1f}s)")

        step_start = time.time()
        print(">>> ANALYSIS STEP 3: Generating recommendations...")
        recommendations = self._generate_recommendations(segments, sentiment)
        print(f">>> ANALYSIS STEP 3 COMPLETE: Recommendations generated ({time.time() - step_start:.1f}s)")

        print(f">>> TOTAL ANALYSIS TIME: {time.time() - analysis_start:.1f}s")

        return AnalysisResult(
            summary=summary,
            sentiment=sentiment,
            recommendations=recommendations,
        )

    def _format_transcript(self, segments: List[Segment]) -> str:
        """Format segments into readable transcript for prompts."""
        lines = []
        for seg in segments:
            timestamp = f"[{seg.start:.1f}s - {seg.end:.1f}s]"
            lines.append(f"{timestamp} {seg.speaker}: {seg.text}")
        return "\n".join(lines)

    def _generate_summary(self, segments: List[Segment]) -> str:
        """Generate a concise summary of the conversation."""
        transcript = self._format_transcript(segments)
        if len(transcript) > MAX_TRANSCRIPT_CHARS:
            transcript = (
                transcript[:MAX_TRANSCRIPT_CHARS]
                + "\n\n[...transcript truncated for length]"
            )

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.3,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a call center quality assurance analyst. "
                        "Provide a concise 2-4 sentence summary of the following "
                        "call center conversation. Focus on the purpose of the call, "
                        "key topics discussed, and the outcome."
                    ),
                },
                {"role": "user", "content": transcript},
            ],
        )
        return response.choices[0].message.content.strip()

    def _analyze_sentiment(self, segments: List[Segment]) -> SentimentResult:
        """Analyze sentiment for each utterance and overall."""
        all_sentiments: List[SentimentItem] = []

        # Process in chunks to handle long conversations
        for chunk_start in range(0, len(segments), MAX_UTTERANCES_PER_CHUNK):
            chunk = segments[chunk_start : chunk_start + MAX_UTTERANCES_PER_CHUNK]
            chunk_sentiments = self._analyze_sentiment_chunk(chunk, chunk_start)
            all_sentiments.extend(chunk_sentiments)

        # Calculate overall sentiment
        if all_sentiments:
            avg_score = sum(s.sentiment_score for s in all_sentiments) / len(
                all_sentiments
            )
            if avg_score > 0.2:
                overall = "positive"
            elif avg_score < -0.2:
                overall = "negative"
            else:
                overall = "neutral"
        else:
            avg_score = 0.0
            overall = "neutral"

        return SentimentResult(
            utterance_sentiments=all_sentiments,
            overall_sentiment=overall,
            overall_score=round(avg_score, 3),
        )

    def _analyze_sentiment_chunk(
        self, segments: List[Segment], offset: int
    ) -> List[SentimentItem]:
        """Analyze sentiment for a chunk of utterances."""
        transcript = self._format_transcript(segments)

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            max_tokens=4096,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a sentiment analysis expert specializing in call center conversations. "
                        "Analyze each utterance and classify its emotional sentiment.\n\n"
                        "IMPORTANT GUIDELINES:\n"
                        "- Procedural or transactional statements (e.g. 'Let me open that', 'OK', "
                        "'One moment please') are NEUTRAL, not negative.\n"
                        "- Simple questions seeking information (e.g. 'Are you in Canada?', "
                        "'What is your account number?') are NEUTRAL.\n"
                        "- Short acknowledgments ('OK', 'Sure', 'Alright', 'Yes') are NEUTRAL.\n"
                        "- Only classify as NEGATIVE when there are clear indicators of frustration, "
                        "anger, dissatisfaction, complaints, or rudeness.\n"
                        "- Only classify as POSITIVE when there is clear warmth, gratitude, "
                        "enthusiasm, or satisfaction.\n"
                        "- When in doubt, classify as NEUTRAL.\n\n"
                        "For each utterance, provide a sentiment label (positive, neutral, negative) "
                        "and a score from -1.0 (very negative) to 1.0 (very positive).\n\n"
                        "Return JSON in this exact format:\n"
                        '{"utterances": [{"index": 0, "sentiment": "neutral", '
                        '"score": 0.0}, ...]}'
                    ),
                },
                {"role": "user", "content": transcript},
            ],
        )

        content = response.choices[0].message.content.strip()
        finish_reason = response.choices[0].finish_reason
        print(f"  Sentiment chunk: finish_reason={finish_reason}, response_length={len(content)}")
        if finish_reason == "length":
            print("  WARNING: Response was truncated!")
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"  JSON parse error: {e}")
            print(f"  Response tail: ...{content[-200:]}")
            raise
        utterances = data.get("utterances", [])
        # print(utterances)

        results = []
        for item in utterances:
            idx = item.get("index", 0)
            if idx < len(segments):
                seg = segments[idx]
                results.append(
                    SentimentItem(
                        index=offset + idx,
                        speaker=seg.speaker,
                        text=seg.text,
                        start_time=seg.start,
                        end_time=seg.end,
                        sentiment=item.get("sentiment", "neutral"),
                        sentiment_score=float(item.get("score", 0.0)),
                    )
                )

        return results

    def _generate_recommendations(
        self, segments: List[Segment], sentiment: SentimentResult
    ) -> List[str]:
        """Generate QA recommendations based on the conversation."""
        transcript = self._format_transcript(segments)
        if len(transcript) > MAX_TRANSCRIPT_CHARS:
            transcript = (
                transcript[:MAX_TRANSCRIPT_CHARS]
                + "\n\n[...transcript truncated for length]"
            )

        sentiment_summary = (
            f"Overall sentiment: {sentiment.overall_sentiment} "
            f"(score: {sentiment.overall_score})\n"
            f"Total utterances analyzed: {len(sentiment.utterance_sentiments)}\n"
        )
        negative_count = sum(
            1
            for s in sentiment.utterance_sentiments
            if s.sentiment == "negative"
        )
        positive_count = sum(
            1
            for s in sentiment.utterance_sentiments
            if s.sentiment == "positive"
        )
        sentiment_summary += (
            f"Negative utterances: {negative_count}\n"
            f"Positive utterances: {positive_count}"
        )

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.4,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a call center quality assurance expert. Based on the "
                        "following conversation transcript and sentiment analysis, "
                        "provide 3-5 actionable recommendations for improving agent "
                        "performance. Consider: tone, empathy, problem resolution, "
                        "compliance, communication clarity, and customer satisfaction.\n\n"
                        'Return JSON in this format: {"recommendations": ["rec1", "rec2", ...]}'
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"TRANSCRIPT:\n{transcript}\n\n"
                        f"SENTIMENT ANALYSIS:\n{sentiment_summary}"
                    ),
                },
            ],
        )

        content = response.choices[0].message.content.strip()
        finish_reason = response.choices[0].finish_reason
        print(f"  Recommendations: finish_reason={finish_reason}, response_length={len(content)}")
        if finish_reason == "length":
            print("  WARNING: Response was truncated!")
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"  JSON parse error: {e}")
            print(f"  Response tail: ...{content[-200:]}")
            raise
        return data.get("recommendations", [])
