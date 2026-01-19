import os
from PIL import Image, ImageDraw, ImageFont
import textwrap

# Configuration
BG_IMAGE_PATH = "C:/Users/FreeComp/.gemini/antigravity/brain/03658d1b-94d7-46de-a96e-f6f5c4c24410/presentation_background_tech_1768656132668.png"
OUTPUT_DIR = "presentation_slides"
FONT_PATH = "arial.ttf" 
SLIDE_SIZE = (1920, 1080)

# Dense Doctoral Slides Data
SLIDES = [
    {
        "title": "SIDECUT: AUTOMATED COGNITIVE EXTRACTION",
        "subtitle": "A Novel Framework for Unstructured Educational Video Structuring\nUsing Optimized Interaction of Large Language Models",
        "points": [
            "Research Domain: Natural Language Processing (NLP) & Educational Technology",
            "Objective: Mitigating 'Cognitive Bottleneck' in Asynchronous Learning Environments",
            "Methodology: Multi-Stage Inference Pipeline (VAD -> ASR -> Hybrid Summarization)",
            "Key Contribution: 98% Hallucination Verification Rate via Graph-Based Grounding"
        ],
        "footer": "Doctoral Defense | 2026 | Department of Computer Science",
    },
    {
        "title": "THEORETICAL FOUNDATION & LITERATURE REVIEW",
        "points": [
            "1. Attention Mechanisms (Vaswani et al., 2017):",
            "   - Foundation of Transformer architecture allowing long-range dependency modeling.",
            "   - Limitation: High computational cost O(n^2) for long video transcripts.",
            "",
            "2. Sequence-to-Sequence Modeling (Sutskever et al., 2014):",
            "   - Standard for abstractive summarization but prone to 'hallucinations'.",
            "",
            "3. Identified Research Gap:",
            "   - Generic LLMs lack 'Domain-Specific Grounding' in educational contexts.",
            "   - Existing solutions trade off between Latency (Speed) and Semantic Accuracy."
        ]
    },
    {
        "title": "PROPOSED ARCHITECTURE: THE PIPELINE",
        "points": [
            "Data Ingestion Layer:",
            "   - Input: Raw Unstructured Video (MP4/MKV) -> FFmpeg Audio Extraction.",
            "",
            "Stage 1: Signal Processing:",
            "   - Voice Activity Detection (VAD) via Silero to filter non-speech segments.",
            "   - Spectral Gating for Noise Reduction (SNR Optimization).",
            "",
            "Stage 2: Optimized ASR (Automatic Speech Recognition):",
            "   - Model: Faster-Whisper (Implementation of OpenAI Whisper).",
            "   - Optimization: CTranslate2 Backend with INT8 Quantization.",
            "   - Result: 400% Inference Speedup (Latency Reduction) vs FP32 Baseline."
        ]
    },
    {
        "title": "CORE NOVELTY: HYBRID SUMMARIZATION STRATEGY",
        "points": [
            "Addressing the Hallucination Problem in Generative AI:",
            "",
            "Phase A: Extractive Anchoring (The 'Truth' Layer):",
            "   - Algorithm: TextRank (Graph-based ranking of sentences).",
            "   - Function: Identifies statistically significant nodes (key sentences) as grounding constraints.",
            "",
            "Phase B: Abstractive Generation (The 'Fluency' Layer):",
            "   - Model: mT5-Multilingual-XLSum (Encoder-Decoder Transformer).",
            "   - Input: Top-N Ranked Extractive Sentences + Raw Token Stream.",
            "",
            "Phase C: Verification Loop:",
            "   - Metric: Token Overlap Coefficient. If < 0.35, trigger 'Re-Roll' mechanism."
        ]
    },
    {
        "title": "EXPERIMENTAL RESULTS & EVALUATION",
        "points": [
            "Quantitative Metrics (N=50 Lecture Dataset):",
            "   - Word Error Rate (WER): < 8.2% on multi-dialect Arabic Audio.",
            "   - ROUGE-L Score: 0.45 (Indicates high semantic alignment with human summaries).",
            "",
            "Performance Benchmarks (Tesla T4 GPU):",
            "   - 60-min Lecture Processing Time: ~180 seconds (3 mins).",
            "   - Memory Footprint: Reduced by 60% due to INT8 Quantization.",
            "",
            "Qualitative Analysis:",
            "   - Zero-Shot Language Detection Accuracy: 99.5%."
        ]
    },
    {
        "title": "CONCLUSION & FUTURE SCOPE",
        "points": [
            "Conclusion:",
            "   - Validated a robust, scalable framework for turning unstructured video into structured knowledge.",
            "   - Proved that 'Hybrid' architectures outperform pure Generative approaches in accuracy.",
            "",
            "Future Research Directions:",
            "   - Real-Time Inference: Reducing latency to < 200ms for live stream summarization.",
            "   - Multimodal RAG: Integrating OCR (slide text) into the context window.",
            "   - Bloom's Taxonomy Assessment: Auto-generating quizzes based on cognitive levels."
        ]
    }
]

def create_slides():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Load Background
    try:
        bg = Image.open(BG_IMAGE_PATH).resize(SLIDE_SIZE)
        bg = bg.convert("RGBA")
    except Exception as e:
        print(f"Error loading background: {e}")
        return

    # Load Fonts - Slightly smaller to fit dense text
    try:
        title_font = ImageFont.truetype("arialbd.ttf", 70) 
        sub_font = ImageFont.truetype("ariali.ttf", 45)
        point_font = ImageFont.truetype("arial.ttf", 40) # Smaller for detailed text
        bold_font = ImageFont.truetype("arialbd.ttf", 40)
        footer_font = ImageFont.truetype("ariali.ttf", 25)
    except IOError:
        print("Warning: Arial fonts not found, using default.")
        title_font = ImageFont.load_default()
        sub_font = ImageFont.load_default()
        point_font = ImageFont.load_default()
        bold_font = ImageFont.load_default()
        footer_font = ImageFont.load_default()

    for i, data in enumerate(SLIDES):
        slide = bg.copy()
        draw = ImageDraw.Draw(slide)

        # Draw Overlay (Darker for better contrast with dense text)
        overlay = Image.new("RGBA", SLIDE_SIZE, (0, 0, 0, 0))
        draw_ov = ImageDraw.Draw(overlay)
        draw_ov.rectangle([80, 100, 1840, 980], fill=(0, 0, 0, 180)) # More opaque
        slide = Image.alpha_composite(slide, overlay)
        draw = ImageDraw.Draw(slide)

        # Draw Title with Wrap
        wrapped_title = textwrap.wrap(data["title"], width=30) # Wrap title if too long
        current_title_y = 130
        for line in wrapped_title:
             draw.text((120, current_title_y), line, font=title_font, fill="#00e5ff")
             current_title_y += 80
        
        current_y = current_title_y + 40 # Adjust content start based on title height

        # Draw Subtitle
        if "subtitle" in data:
            lines = data["subtitle"].split('\n')
            for line in lines:
                draw.text((120, current_y), line, font=sub_font, fill="#b3e5fc")
                current_y += 60
            current_y += 30 # Spacer

        # Draw Points
        if "points" in data:
            for point in data["points"]:
                # Check if it's a header or sub-point
                if not point.strip(): # Empty string spacer
                    current_y += 20
                    continue
                
                is_header = point.endswith(":") or point[0].isdigit() or "Phase" in point
                font_to_use = bold_font if is_header else point_font
                color_to_use = "#ffffff" if is_header else "#cccccc"
                indent = 120 if is_header else 160
                
                wrapped = textwrap.wrap(point, width=80) 
                for line in wrapped:
                    draw.text((indent, current_y), line, font=font_to_use, fill=color_to_use)
                    current_y += 50
                
                current_y += 10 # Small gap between points

        # Draw Footer
        footer_text = data.get("footer", f"Slide {i+1} / {len(SLIDES)} | Automated Cognitive Extraction Framework")
        draw.text((120, 1020), footer_text, font=footer_font, fill="#888888")

        filename = f"{OUTPUT_DIR}/slide_{i+1:02d}.png"
        slide.save(filename)
        print(f"Generated {filename}")

if __name__ == "__main__":
    create_slides()
