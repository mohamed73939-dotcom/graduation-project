# Sidecut AI | Lecture Video Summarizer ✂️🧠

**Sidecut AI** is a professional, high-performance tool designed to transform long lecture videos into concise, readable summaries using state-of-the-art AI models. It handles everything from transcription and slide extraction to hierarchical summarization and PDF generation, with full support for Arabic and English.

![Main UI](https://via.placeholder.com/800x400?text=Sidecut+AI+Dashboard+Preview)

## 🚀 Key Features

- **High-Speed Transcription:** Powered by OpenAI Whisper & Faster-Whisper (CTranslate2) for 4x faster processing on CPU.
- **Smart Summarization:** Uses a custom-trained **mT5** model for abstractive summarization (Multi-stage hierarchical approach).
- **Computer Vision (Slide Detection):** Automatically detects and extracts lecture slides/frames using histogram comparison and EasyOCR.
- **Arabic Language Support:** Specialized NLP pipeline for Arabic normalization, reshaping, and RTL PDF generation.
- **Modern UI:** Luxury "Glassmorphism" interface built with Streamlit & NiceGUI.
- **Offline Capability:** Designed to run locally with optimized model quantization (INT8).

---

## 🏗️ Project Structure

```bash
📂 sidecut-ai/
├── 📂 backend/           # FastAPI Server & AI Engines
│   ├── 📂 scripts/       # Utility scripts (video rendering, slides)
│   ├── api.py            # Main API gateway
│   ├── summarizer.py     # Hierarchical Abstractive Summarization
│   └── whisper_engine.py # Speech-to-Text engine
├── 📂 frontend/          # Multi-version Frontends
│   ├── 📂 streamlit-app/ # Main Dashboard
│   └── 📂 nice-ui/       # Modern Minimal UI
├── 📂 models/            # Model Weights & Data (Ignored by Git)
├── 📂 uploads/           # Temp storage for video uploads
├── 📂 outputs/           # Summaries, PDF, and Slides storage
└── run_app.bat           # One-click launcher
```

---

## 🛠️ Installation & Setup

### 1. Prerequisites
- Python 3.10+
- **FFmpeg** (installed and added to your system's PATH)
- NVIDIA GPU (Optional, but recommended for faster processing)

### 2. Clone & Install
```bash
git clone https://github.com/yourusername/lecture-summarizer.git
cd lecture-summarizer
pip install -r backend/requirements.txt
```

### 3. Setup Models
- Place your model weights in `models/weights/custom_mt5-small/`.
- The system will automatically download Whisper weights on first run.

---

## 🚦 How to Use

1. **One-Click Launch:** Run `run_app.bat` to start both Backend and Frontend.
2. **Manual Launch:**
   - **Backend:** `cd backend && python -m uvicorn api:app --port 8000`
   - **Frontend:** `streamlit run frontend/streamlit-app/app.py`
3. Upload your `.mp4` video, select the language, and click **Analyze Video**.
4. Download the generated PDF summary and view extracted slides.

---

## 🧪 Algorithms Used

- **Speech-to-Text:** OpenAI Whisper (Base/Small) with CTranslate2 optimization.
- **Ranking:** TextRank (Sentence Centrality) for extractive basis.
- **NLP:** Bidirectional Recurrent processing via Transformers (mT5).
- **CV:** Background subtraction and Histogram Correlation for slide extraction.

---

## 📄 License
This project is licensed under the MIT License - see the `LICENSE` file for details.

## 🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request.
