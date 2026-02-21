import os
import logging
from uuid import uuid4
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor, Color
from reportlab.lib.units import inch, cm
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
)

from config import settings
from app.models.transcript import ReportData, SentimentItem, SpeakerStat

logger = logging.getLogger(__name__)

# Color scheme
PRIMARY = HexColor("#003366")
SECONDARY = HexColor("#0066CC")
ACCENT = HexColor("#FF6600")
LIGHT_BG = HexColor("#F0F4F8")
WHITE = HexColor("#FFFFFF")
DARK_TEXT = HexColor("#1A1A2E")

SPEAKER_COLORS = [
    HexColor("#003366"),
    HexColor("#CC3300"),
    HexColor("#006633"),
    HexColor("#663399"),
    HexColor("#CC6600"),
]

SPEAKER_CHART_COLORS = ["#003366", "#CC3300", "#006633", "#663399", "#CC6600"]


def generate_pdf(report_data: ReportData) -> Tuple[str, str]:
    """Generate a styled PDF report. Returns (file_path, report_id)."""
    report_id = report_data.report_id
    filename = f"qa_report_{report_id}.pdf"
    filepath = os.path.join(settings.reports_dir, filename)

    doc = SimpleDocTemplate(
        filepath,
        pagesize=A4,
        rightMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    styles = _get_custom_styles()
    story = []

    # Header section
    story.extend(_build_header(report_data, styles))
    story.append(Spacer(1, 0.5 * cm))

    # Summary section
    story.extend(_build_summary(report_data, styles))
    story.append(Spacer(1, 0.5 * cm))

    # Speaker statistics with pie chart
    story.extend(_build_speaker_stats(report_data, styles))
    story.append(PageBreak())

    # Sentiment analysis with trend chart
    story.extend(_build_sentiment_section(report_data, styles))
    story.append(PageBreak())

    # Transcript section
    story.extend(_build_transcript(report_data, styles))
    story.append(PageBreak())

    # Recommendations section
    story.extend(_build_recommendations(report_data, styles))

    # Build PDF
    doc.build(story, onFirstPage=_add_page_number, onLaterPages=_add_page_number)

    logger.info("PDF report generated: %s", filepath)
    return filepath, report_id


def _get_custom_styles():
    """Create custom paragraph styles for the report."""
    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            "ReportTitle",
            parent=styles["Title"],
            fontSize=22,
            textColor=PRIMARY,
            spaceAfter=20,
            alignment=TA_CENTER,
        )
    )
    styles.add(
        ParagraphStyle(
            "SectionHeading",
            parent=styles["Heading1"],
            fontSize=14,
            textColor=PRIMARY,
            spaceBefore=15,
            spaceAfter=8,
            borderWidth=0,
            borderPadding=0,
        )
    )
    styles.add(
        ParagraphStyle(
            "BodyText2",
            parent=styles["BodyText"],
            fontSize=10,
            textColor=DARK_TEXT,
            leading=14,
        )
    )
    styles.add(
        ParagraphStyle(
            "TranscriptSpeaker",
            parent=styles["BodyText"],
            fontSize=9,
            leading=13,
            spaceBefore=4,
        )
    )
    styles.add(
        ParagraphStyle(
            "MetaInfo",
            parent=styles["BodyText"],
            fontSize=9,
            textColor=HexColor("#555555"),
            alignment=TA_CENTER,
        )
    )
    return styles


def _build_header(report_data: ReportData, styles) -> list:
    """Build the report header with metadata."""
    elements = []

    elements.append(Paragraph("Call Center QA Report", styles["ReportTitle"]))
    elements.append(
        Paragraph(
            f"File: {report_data.audio_filename} | "
            f"Duration: {report_data.duration_seconds:.1f}s | "
            f"Speakers: {report_data.num_speakers} | "
            f"Date: {report_data.processed_at}",
            styles["MetaInfo"],
        )
    )
    elements.append(Spacer(1, 0.3 * cm))

    # Horizontal divider via table
    divider_data = [["" ]]
    divider_table = Table(divider_data, colWidths=[17 * cm])
    divider_table.setStyle(
        TableStyle(
            [
                ("LINEBELOW", (0, 0), (-1, -1), 1.5, SECONDARY),
            ]
        )
    )
    elements.append(divider_table)
    return elements


def _build_summary(report_data: ReportData, styles) -> list:
    """Build the summary section."""
    elements = []
    elements.append(Paragraph("Conversation Summary", styles["SectionHeading"]))
    elements.append(Paragraph(report_data.analysis.summary, styles["BodyText2"]))

    # Overall sentiment badge
    sentiment = report_data.analysis.sentiment
    sentiment_color = _sentiment_color(sentiment.overall_sentiment)
    elements.append(Spacer(1, 0.3 * cm))
    elements.append(
        Paragraph(
            f'<b>Overall Sentiment:</b> <font color="{sentiment_color}">'
            f"{sentiment.overall_sentiment.upper()}</font> "
            f"(score: {sentiment.overall_score:.2f})",
            styles["BodyText2"],
        )
    )
    return elements


def _build_speaker_stats(report_data: ReportData, styles) -> list:
    """Build speaker statistics section with pie chart."""
    elements = []
    elements.append(Paragraph("Speaker Statistics", styles["SectionHeading"]))

    # Stats table
    table_data = [["Speaker", "Duration (s)", "Share (%)", "Utterances"]]
    for stat in report_data.speaker_stats:
        table_data.append(
            [
                stat.speaker,
                f"{stat.total_duration_seconds:.1f}",
                f"{stat.percentage:.1f}%",
                str(stat.utterance_count),
            ]
        )

    table = Table(table_data, colWidths=[5 * cm, 3.5 * cm, 3 * cm, 3 * cm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
                ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
                ("FONTSIZE", (0, 0), (-1, 0), 10),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 1), (-1, -1), 9),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#CCCCCC")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_BG]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    elements.append(table)
    elements.append(Spacer(1, 0.5 * cm))

    # Pie chart
    chart_path = _create_speaker_pie_chart(report_data.speaker_stats)
    if chart_path:
        elements.append(Image(chart_path, width=10 * cm, height=8 * cm))
        _schedule_cleanup(chart_path)

    return elements


def _build_sentiment_section(report_data: ReportData, styles) -> list:
    """Build sentiment analysis section with trend chart."""
    elements = []
    elements.append(Paragraph("Sentiment Analysis", styles["SectionHeading"]))

    sentiments = report_data.analysis.sentiment.utterance_sentiments

    # Sentiment breakdown summary
    pos_count = sum(1 for s in sentiments if s.sentiment == "positive")
    neg_count = sum(1 for s in sentiments if s.sentiment == "negative")
    neu_count = sum(1 for s in sentiments if s.sentiment == "neutral")

    elements.append(
        Paragraph(
            f"<b>Positive:</b> {pos_count} | "
            f"<b>Neutral:</b> {neu_count} | "
            f"<b>Negative:</b> {neg_count}",
            styles["BodyText2"],
        )
    )
    elements.append(Spacer(1, 0.3 * cm))

    # Sentiment trend chart
    chart_path = _create_sentiment_chart(sentiments)
    if chart_path:
        elements.append(Image(chart_path, width=16 * cm, height=6 * cm))
        _schedule_cleanup(chart_path)

    elements.append(Spacer(1, 0.5 * cm))

    # Per-utterance sentiment table (top 10 most negative for quick review)
    negative_items = sorted(
        [s for s in sentiments if s.sentiment == "negative"],
        key=lambda x: x.sentiment_score,
    )[:10]

    if negative_items:
        elements.append(
            Paragraph("Notable Negative Utterances", styles["SectionHeading"])
        )
        table_data = [["Speaker", "Text", "Score"]]
        for item in negative_items:
            text = item.text[:80] + "..." if len(item.text) > 80 else item.text
            table_data.append(
                [item.speaker, text, f"{item.sentiment_score:.2f}"]
            )

        table = Table(table_data, colWidths=[3 * cm, 11 * cm, 2.5 * cm])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), HexColor("#CC3300")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#CCCCCC")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_BG]),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        elements.append(table)

    return elements


def _build_transcript(report_data: ReportData, styles) -> list:
    """Build the full transcript section with color-coded speakers."""
    elements = []
    elements.append(Paragraph("Full Transcript", styles["SectionHeading"]))

    speaker_color_map = {}
    for seg in report_data.transcription.segments:
        if seg.speaker not in speaker_color_map:
            idx = len(speaker_color_map) % len(SPEAKER_CHART_COLORS)
            speaker_color_map[seg.speaker] = SPEAKER_CHART_COLORS[idx]

    for seg in report_data.transcription.segments:
        color_hex = speaker_color_map.get(seg.speaker, "#003366")
        timestamp = f"[{seg.start:.1f}s]"
        elements.append(
            Paragraph(
                f'<font color="#888888" size="7">{timestamp}</font> '
                f'<font color="{color_hex}"><b>{seg.speaker}:</b></font> '
                f"{seg.text}",
                styles["TranscriptSpeaker"],
            )
        )

    return elements


def _build_recommendations(report_data: ReportData, styles) -> list:
    """Build the recommendations section."""
    elements = []
    elements.append(Paragraph("Recommendations", styles["SectionHeading"]))

    for i, rec in enumerate(report_data.analysis.recommendations, 1):
        elements.append(
            Paragraph(
                f"<b>{i}.</b> {rec}",
                styles["BodyText2"],
            )
        )
        elements.append(Spacer(1, 0.2 * cm))

    return elements


def _create_sentiment_chart(sentiments: List[SentimentItem]) -> str:
    """Generate sentiment trend line chart and return path to temp PNG."""
    if not sentiments:
        return ""

    fig, ax = plt.subplots(figsize=(8, 3))

    times = [(s.start_time + s.end_time) / 2 for s in sentiments]
    scores = [s.sentiment_score for s in sentiments]

    ax.plot(times, scores, color="#0066CC", linewidth=1.5, alpha=0.8)
    ax.fill_between(times, scores, alpha=0.1, color="#0066CC")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)

    # Color bands
    ax.axhspan(0.2, 1.1, alpha=0.05, color="green")
    ax.axhspan(-1.1, -0.2, alpha=0.05, color="red")

    ax.set_xlabel("Time (seconds)", fontsize=9)
    ax.set_ylabel("Sentiment Score", fontsize=9)
    ax.set_ylim(-1.1, 1.1)
    ax.set_title("Sentiment Trend Over Conversation", fontsize=11, fontweight="bold")
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    chart_path = os.path.join(
        settings.reports_dir, f"_tmp_sentiment_{uuid4().hex[:8]}.png"
    )
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return chart_path


def _create_speaker_pie_chart(speaker_stats: List[SpeakerStat]) -> str:
    """Generate speaker time distribution pie chart and return path to temp PNG."""
    if not speaker_stats:
        return ""

    fig, ax = plt.subplots(figsize=(5, 4))

    labels = [s.speaker for s in speaker_stats]
    sizes = [s.percentage for s in speaker_stats]
    colors = SPEAKER_CHART_COLORS[: len(labels)]

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 9},
    )
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontweight("bold")

    ax.set_title(
        "Speaker Time Distribution", fontsize=11, fontweight="bold", pad=15
    )

    plt.tight_layout()
    chart_path = os.path.join(
        settings.reports_dir, f"_tmp_pie_{uuid4().hex[:8]}.png"
    )
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return chart_path


def _sentiment_color(sentiment: str) -> str:
    """Return hex color for a sentiment label."""
    mapping = {
        "positive": "#228B22",
        "neutral": "#666666",
        "negative": "#CC3300",
    }
    return mapping.get(sentiment, "#666666")


def _add_page_number(canvas, doc):
    """Add page number to the footer."""
    page_num = canvas.getPageNumber()
    text = f"Page {page_num}"
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(HexColor("#888888"))
    canvas.drawCentredString(A4[0] / 2, 1 * cm, text)
    canvas.restoreState()


_cleanup_files: List[str] = []


def _schedule_cleanup(path: str):
    """Schedule a temp file for cleanup after PDF generation."""
    _cleanup_files.append(path)


def cleanup_temp_files():
    """Remove all temporary chart files."""
    for path in _cleanup_files:
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass
    _cleanup_files.clear()
