import os
import requests
import streamlit as st
import streamlit.components.v1 as components
import base64
from locales import TRANSLATIONS

# ------------------ Page Config ------------------
st.set_page_config(
    page_title="Sidecut | AI Video Summarizer",
    page_icon="✂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------ Session State & Theme Management ------------------
DEFAULT_KEYS = {
    "result_data": None,
    "upload_bytes": None,
    "filename": None,
    "mimetype": None,
    "show_full_transcription": False,
    "summarizing": False,
    "ui_lang": "en"   # 'en' or 'ar'
}
for k, v in DEFAULT_KEYS.items():
    if k not in st.session_state:
        st.session_state[k] = v

def set_language():
    pass

# Helper to get current translation
t = TRANSLATIONS[st.session_state.ui_lang]

# ------------------ 3D Background & Custom CSS (Injected) ------------------
# Enforced Dark Theme
current_theme = {
    "--bg-gradient": "linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%)",
    "--glass-bg": "rgba(17, 25, 40, 0.75)",
    "--glass-border": "rgba(255, 255, 255, 0.125)",
    "--text-primary": "#f8fafc",
    "--text-secondary": "#94a3b8",
    "--accent-color": "#38bdf8",
    "--accent-hover": "#0ea5e9",
    "--card-shadow": "0 8px 32px 0 rgba(0, 0, 0, 0.37)",
    "--input-bg": "rgba(255, 255, 255, 0.05)"
}

# 3D Background Script (Luxury Starlight Roof Effect)
# 3D Background Script (CSS-Based Luxury Starlight)
# NOTE: We use pure CSS/HTML injected via st.markdown to ensure it covers the full page
# and isn't trapped in a Streamlit iframe component.
import random

def generate_star_css(num_stars=200):
    css = ""
    html = ""
    for i in range(num_stars):
        x = random.randint(0, 100)
        y = random.randint(0, 100)
        duration = random.uniform(1.5, 4.0)
        delay = random.uniform(0, 2.0)
        size = random.uniform(1, 3)
        opacity = random.uniform(0.4, 0.9)
        
        # Star HTML
        html += f'<div class="star" style="top:{y}%; left:{x}%; width:{size}px; height:{size}px; animation-duration:{duration}s; animation-delay:{delay}s; opacity:{opacity};"></div>'
    
    # Shooting stars HTML
    html += '<div class="shooting_star"></div><div class="shooting_star" style="animation-delay: 5s; top: 30%;"></div>'
    
    return html

stars_html = generate_star_css(300)

BACKGROUND_HTML = f"""
<style>
  [data-testid="stAppViewContainer"] > .main {{
    background: {current_theme["--bg-gradient"]};
  }}
  
  [data-testid="stAppViewContainer"] {{
    background: {current_theme["--bg-gradient"]};
  }}
  
  /* Star Styles */
  .star {{
    position: fixed;
    background: white;
    border-radius: 50%;
    z-index: 0; /* Behind content but in front of background color */
    box-shadow: 0 0 2px white;
    animation: twinkle infinite ease-in-out alternate;
  }}
  
  @keyframes twinkle {{
    0% {{ transform: scale(1); opacity: 0.4; }}
    100% {{ transform: scale(1.2); opacity: 1; }}
  }}
  
  /* Shooting Star */
  .shooting_star {{
    position: fixed;
    left: 50%;
    top: 50%;
    height: 2px;
    background: linear-gradient(-45deg, rgba(255,255,255,1), rgba(0,0,255,0));
    border-radius: 999px;
    filter: drop-shadow(0 0 6px rgba(105, 155, 255, 1));
    animation: tail 3000ms ease-in-out infinite, shooting 3000ms ease-in-out infinite;
    z-index: 0;
  }}
  
  @keyframes tail {{
    0% {{ width: 0; }}
    30% {{ width: 100px; }}
    100% {{ width: 0; }}
  }}
  
  @keyframes shooting {{
    0% {{ transform: translateX(0); }}
    100% {{ transform: translateX(300px) translateY(-100px); opacity: 0;}}
  }}
  
  /* Ensure content is above stars */
  .block-container {{
    position: relative;
    z-index: 1;
  }}
  
</style>

<!-- Injected Stars -->
<div id="star-container">
  {stars_html}
</div>
"""

# Inject using st.markdown to bypass iframe sandbox
st.markdown(BACKGROUND_HTML, unsafe_allow_html=True)

# Global CSS for Glassmorphism & Animations
CSS_STYLES = f"""
<style>
/* Font Import */
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;600;700&family=Outfit:wght@300;400;600;700&display=swap');

:root {{
    --glass-bg: {current_theme["--glass-bg"]};
    --glass-border: {current_theme["--glass-border"]};
    --text-primary: {current_theme["--text-primary"]};
    --text-secondary: {current_theme["--text-secondary"]};
    --accent: {current_theme["--accent-color"]};
    --accent-hover: {current_theme["--accent-hover"]};
    --card-shadow: {current_theme["--card-shadow"]};
    --input-bg: {current_theme["--input-bg"]};
    --dir: {t["dir"]};
    --align: {t["align"]};
}}

html, body, [class*="css"] {{
    font-family: '{t["font"]}', sans-serif;
    color: var(--text-primary);
    direction: var(--dir);
}}

/* Hide standard Streamlit header/footer/menu */
header {{visibility: hidden;}}
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}

/* Reset Streamlit container padding */
.block-container {{
    padding-top: 2rem;
    padding-bottom: 5rem;
    max-width: 1200px;
}}

/* Glass Cards */
.glass-card {{
    background: var(--glass-bg);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--glass-border);
    border-radius: 16px;
    padding: 2rem;
    box-shadow: var(--card-shadow);
    margin-bottom: 2rem;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}}

.glass-card:hover {{
    transform: translateY(-4px);
    box-shadow: 0 12px 40px 0 rgba(0,0,0,0.25);
}}

/* Headings */
h1, h2, h3, h4 {{
    font-weight: 700;
    color: var(--text-primary) !important;
}}

.hero-title {{
    font-size: 3.5rem;
    background: linear-gradient(to right, var(--accent), #a855f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 0.5rem;
}}

.hero-subtitle {{
    text-align: center;
    color: var(--text-secondary);
    font-size: 1.2rem;
    margin-bottom: 2.5rem;
}}

/* Buttons */
.stButton > button {{
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent-hover) 100%);
    color: white !important;
    border: none;
    border-radius: 12px;
    padding: 0.6rem 1.5rem;
    font-weight: 600;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 4px 14px 0 rgba(0,0,0,0.2);
    width: 100%;
}}

.stButton > button:hover {{
    filter: brightness(1.1);
    transform: scale(1.05);
    box-shadow: 0 8px 25px 0 rgba(0,0,0,0.3);
}}

/* Secondary Button (Clear) */
.clear-btn > button {{
    background: transparent !important;
    border: 1px solid var(--text-secondary) !important;
    color: var(--text-secondary) !important;
    box-shadow: none;
}}
.clear-btn > button:hover {{
    background: rgba(255,0,0,0.1) !important;
    color: #ef4444 !important;
    border-color: #ef4444 !important;
}}

/* Inputs */
.stTextInput input, .stSelectbox [data-baseweb="select"] {{
    background: var(--input-bg) !important;
    border: 1px solid var(--glass-border) !important;
    color: var(--text-primary) !important;
    border-radius: 10px;
}}

/* Dropdown Menus & Popovers */
div[data-baseweb="popover"], div[data-baseweb="menu"], div[role="listbox"] {{
    background: var(--glass-bg) !important;
    backdrop-filter: blur(16px);
    border: 1px solid var(--glass-border) !important;
}}

/* Menu Options */
ul[data-baseweb="menu"] li, li[role="option"] {{
    color: var(--text-primary) !important;
}}

/* Selected/Hovered Option */
ul[data-baseweb="menu"] li[aria-selected="true"], li[role="option"][aria-selected="true"] {{
    background: var(--accent) !important;
    color: white !important;
}}

/* File Uploader */
.stFileUploader section {{
    background: rgba(255,255,255,0.03);
    border: 2px dashed var(--glass-border);
    border-radius: 12px;
    transition: border-color 0.3s ease, background 0.3s ease;
}}
.stFileUploader section:hover {{
    border-color: var(--accent);
    background: rgba(255,255,255,0.06);
}}

/* Summary Box */
.summary-content {{
    font-size: 1.05rem;
    line-height: 1.8;
    white-space: pre-wrap;
    color: var(--text-primary);
    font-family: '{t["font"]}', sans-serif;
}}

/* Metrics Grid */
.metric-container {{
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 1rem;
    background: rgba(255,255,255,0.03);
    border-radius: 12px;
    border: 1px solid var(--glass-border);
    transition: transform 0.2s;
}}
.metric-container:hover {{
    transform: scale(1.03);
    background: rgba(255,255,255,0.05);
}}
.metric-value {{
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--accent);
}}
.metric-label {{
    font-size: 0.85rem;
    color: var(--text-secondary);
    margin-top: 0.25rem;
    text-align: center;
}}

.highlight {{
    color: #ffd700; /* Gold color */
    font-weight: bold;
    background: rgba(255, 215, 0, 0.1);
    padding: 0 4px;
    border-radius: 4px;
}}

/* Animations */
@keyframes fadeIn {{
    from {{ opacity: 0; transform: translateY(10px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}

@keyframes slideUp {{
    from {{ opacity: 0; transform: translateY(30px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}

@keyframes pulse {{
    0% {{ box-shadow: 0 0 0 0 rgba(56, 189, 248, 0.4); }}
    70% {{ box-shadow: 0 0 0 10px rgba(56, 189, 248, 0); }}
    100% {{ box-shadow: 0 0 0 0 rgba(56, 189, 248, 0); }}
}}

.fade-in {{
    animation: fadeIn 0.8s cubic-bezier(0.2, 0.8, 0.2, 1) forwards;
}}

.slide-up {{
    animation: slideUp 0.8s cubic-bezier(0.2, 0.8, 0.2, 1) forwards;
}}

/* Delay classes for staggered animations */
.delay-1 {{ animation-delay: 0.1s; }}
.delay-2 {{ animation-delay: 0.2s; }}
.delay-3 {{ animation-delay: 0.3s; }}

/* Pulse effect class for important buttons */
.pulse-btn {{
     animation: pulse 2s infinite;
}}

</style>
"""

# Inject Background & CSS (Background already injected above via st.markdown)
st.markdown(CSS_STYLES, unsafe_allow_html=True)

# ------------------ Sidebar (Settings) ------------------
with st.sidebar:
    st.markdown(f"<h3 style='margin-bottom:1rem;'>{t['settings']}</h3>", unsafe_allow_html=True)
    
    # Interface Language (First item for visibility)
    lang_options = {"en": "🇺🇸 English", "ar": "🇸🇦 العربية"}
    
    # We use a callback or just check session state on rerun. 
    # To keep it simple, we use the key to bind to session_state.
    selected_ui_lang = st.selectbox(
        t["ui_lang"],
        options=list(lang_options.keys()),
        format_func=lambda x: lang_options[x],
        index=0 if st.session_state.ui_lang == "en" else 1,
        key="ui_lang_selector"
    )
    
    if selected_ui_lang != st.session_state.ui_lang:
        st.session_state.ui_lang = selected_ui_lang
        st.rerun()

    st.markdown("---")
    
    default_api = os.getenv("SIDECUT_API_URL", "http://localhost:8000")
    api_url = st.text_input(t["api_url"], value=default_api)
    
    video_lang = st.selectbox(
        t["source_lang"],
        ["auto", "ar", "en"],
        format_func=lambda x: {"ar": "🇸🇦 Arabic", "en": "🇬🇧 English", "auto": "🌍 Auto-Detect"}[x],
        help=t["source_lang_help"]
    )
    
    st.markdown("---")
    st.caption(t["version"])


# ------------------ Main Hero Section ------------------
st.markdown('<div class="fade-in">', unsafe_allow_html=True)
st.markdown(f'<h1 class="hero-title floating">{t["hero_title"]}<span style="color:var(--text-secondary);font-size:0.5em;">.ai</span></h1>', unsafe_allow_html=True)
st.markdown(f'<p class="hero-subtitle">{t["hero_subtitle"]}</p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ------------------ Upload Section ------------------
st.markdown('<div class="glass-card slide-up delay-1">', unsafe_allow_html=True)

col_upload_left, col_upload_right = st.columns([2, 1])

with col_upload_left:
    uploaded_file = st.file_uploader(
        t["upload_label"], 
        type=["mp4", "avi", "mov", "mkv", "webm"],
        help=t["upload_help"]
    )

with col_upload_right:
    st.markdown(f"""
    <div style="padding-top: 0.5rem; color: var(--text-secondary); font-size: 0.9rem; direction: {t['dir']};">
        <strong>{t['supported_formats']}</strong><br>
        MP4, AVI, MOV, MKV<br><br>
        <strong>{t['processing_steps']}</strong><br>
        {t['step_1']}<br>
        {t['step_2']}<br>
        {t['step_3']}<br>
        {t['step_4']}
    </div>
    """, unsafe_allow_html=True)

# State Management for Upload
if uploaded_file:
    if (st.session_state.upload_bytes is None) or (uploaded_file.name != st.session_state.filename):
        st.session_state.upload_bytes = uploaded_file.read()
        st.session_state.filename = uploaded_file.name
        st.session_state.mimetype = uploaded_file.type
        st.session_state.result_data = None
        st.session_state.summarizing = False
        st.session_state.show_full_transcription = False

elif st.session_state.upload_bytes is not None and not uploaded_file:
    # Clear state if removed
    st.session_state.upload_bytes = None
    st.session_state.filename = None
    st.session_state.result_data = None

st.markdown('</div>', unsafe_allow_html=True)

# ------------------ Preview & Action ------------------
if st.session_state.upload_bytes:
    st.markdown('<div class="glass-card slide-up delay-2">', unsafe_allow_html=True)
    st.subheader(f"{t['preview_title']}: {st.session_state.filename}")
    
    col_vid, col_actions = st.columns([1.5, 1])
    
    with col_vid:
        st.video(st.session_state.upload_bytes)
    
    with col_actions:
        st.markdown("<br>", unsafe_allow_html=True)
        # Use a container to apply the pulse class if desired, or just standard button
        summarize_clicked = st.button(t["btn_analyze"], use_container_width=True, type="primary")
        
        st.markdown("<br>", unsafe_allow_html=True)
        # Clear/Reset
        if st.button(t["btn_reset"], key="reset_btn"):
             st.session_state.result_data = None
             st.session_state.summarizing = False
             st.rerun()

    if summarize_clicked:
        if not api_url:
            st.error("Please configure the API URL in the sidebar.")
        else:
            st.session_state.summarizing = True
            st.session_state.result_data = None
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------ Processing Logic ------------------
def call_backend():
    files = {
        "video": (
            st.session_state.filename,
            st.session_state.upload_bytes,
            st.session_state.mimetype or "application/octet-stream"
        ),
        "language": (None, video_lang)
    }
    try:
        # print(f"DEBUG: sending {len(st.session_state.upload_bytes)} bytes", flush=True)
        resp = requests.post(f"{api_url.rstrip('/')}/api/summarize", files=files, timeout=3600)
        return resp
    except Exception as e:
        return e

if st.session_state.summarizing and st.session_state.result_data is None:
    # Custom Loader Overlay
    loader_placeholder = st.empty()
    loader_placeholder.markdown(f"""
    <div class="glass-card fade-in" style="text-align:center; padding: 3rem;">
        <div class="luxury-loader"></div>
        <h3 style="margin-top: 1rem;">{t["spinner_msg"]}</h3>
        <p style="color: var(--text-secondary); font-size: 0.9rem;">Processing video content with AI...</p>
    </div>
    """, unsafe_allow_html=True)
    
    response = call_backend()
    
    # Clear loader
    loader_placeholder.empty()

    if isinstance(response, Exception):
        st.error(f"Connection Error: {response}")
        st.session_state.summarizing = False
    elif response.status_code == 200:
        payload = response.json()
        if payload.get("status") == "success":
            st.session_state.result_data = payload["data"]
            st.session_state.summarizing = False
            st.rerun()
        else:
            st.error(f"{t['error_api']}: {payload.get('message')}")
            st.session_state.summarizing = False
    else:
        st.error(f"{t['error_server']} {response.status_code}: {response.text}")
        st.session_state.summarizing = False

# ------------------ Results View ------------------
if st.session_state.result_data:
    data = st.session_state.result_data
    detected_lang_code = data.get("detected_language", "en")
    
    # For content direction (summary/transcript), we use the DETECTED language, not the UI language
    content_dir = "rtl" if detected_lang_code == "ar" else "ltr"
    content_align = "right" if detected_lang_code == "ar" else "left"
    
    # 1. Metrics Row
    st.markdown('<div class="slide-up delay-1">', unsafe_allow_html=True)
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    with m_col1:
        st.markdown(f"""<div class="metric-container"><div class="metric-value">{data.get('metrics', {}).get('latency_seconds', 0):.1f}s</div><div class="metric-label">{t['metric_latency']}</div></div>""", unsafe_allow_html=True)
    with m_col2:
        st.markdown(f"""<div class="metric-container"><div class="metric-value">{detected_lang_code.upper()}</div><div class="metric-label">{t['metric_lang']}</div></div>""", unsafe_allow_html=True)
    with m_col3:
        conf = data.get('confidence', {}).get('avg_logprob', 0) or 0
        st.markdown(f"""<div class="metric-container"><div class="metric-value">{conf:.2f}</div><div class="metric-label">{t['metric_conf']}</div></div>""", unsafe_allow_html=True)
    with m_col4:
        # Grounding/Summary score
        sum_score = data.get('confidence', {}).get('summary_score', 0) or 0
        st.markdown(f"""<div class="metric-container"><div class="metric-value">{sum_score:.2f}</div><div class="metric-label">{t.get('metric_sum_conf', 'Model Conf')}</div></div>""", unsafe_allow_html=True)
    with m_col5:
        words = data.get('summary_metadata', {}).get('word_count', 0)
        st.markdown(f"""<div class="metric-container"><div class="metric-value">{words}</div><div class="metric-label">{t['metric_words']}</div></div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 2. Summary Card
    st.markdown('<div class="glass-card slide-up delay-2">', unsafe_allow_html=True)
    st.subheader(t["header_summary"])
    
    summary_text = data.get("summary", "")
    
    st.markdown(f"""
    <div class="summary-content" style="direction: {content_dir}; text-align: {content_align};">
        {summary_text}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Generate PDF Button
    from pdf_generator import generate_pdf_bytes, clean_html_for_pdf
    import re
    
    # Clean the summary text for PDF (remove HTML tags if any remain, though our summary var is usually clean text here)
    # Actually data['summary'] in api.py is extractive text (plain) or formatted bullets (HTML). 
    # For PDF we want the clean text version.
    # The 'summary' key in API response (line 323 in api.py) is extractive_sentences joined by newline (plain text) 
    # OR bullet HTML if fallback.
    # We must strip HTML tags (including the highlight spans we just added)
    
    raw_summary = data.get("summary", "")
    
    # Use centralized cleaning function
    clean_text = clean_html_for_pdf(raw_summary)
    
    # print(f"DEBUG: raw_summary len={len(raw_summary)}")
    # print(f"DEBUG: raw_summary content: {raw_summary[:500]}...")
    # print(f"DEBUG: clean_text len={len(clean_text)}")
    # print(f"DEBUG: clean_text content: {clean_text[:500]}...")
    
    try:
        pdf_data = generate_pdf_bytes(st.session_state.filename, clean_text, language=detected_lang_code)
        st.download_button(
            label=t["btn_dl_summary"].replace("Text", "PDF"), # Update label dynamically or just use "Download PDF"
            data=pdf_data,
            file_name=f"summary_{st.session_state.filename}.pdf",
            mime="application/pdf"
        )
    except Exception as e:
        st.error(f"PDF Generation Error: {e}")
        # Fallback to text
        st.download_button(t["btn_dl_summary"], raw_summary, file_name="summary.txt")
        
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 2.5 Slide Gallery
    slides = data.get("slides", [])
    if slides:
        st.markdown('<div class="glass-card slide-up delay-3">', unsafe_allow_html=True)
        st.subheader("📸 Extracted Slides")
        
        # Display in rows of 3
        cols = st.columns(3)
        for i, slide in enumerate(slides):
            with cols[i % 3]:
                # We need to serve the image. For local dev, we can use the file path directly if frontend wraps it.
                # However, Streamlit st.image handles local paths nicely.
                # The path from backend is absolute or relative to backend. Frontend is in same root.
                # Verify path existence or strictness.
                
                # Check if it's a valid file path
                img_path = slide.get("path")
                if img_path and os.path.exists(img_path):
                    st.image(str(img_path), caption=f"Slide at {int(slide.get('time', 0))}s", use_column_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 3. Transcription Expandable
    with st.expander(t["expander_transcription"]):
        full_text = data.get("transcription_full", "")
        st.markdown(f"""
        <div style="direction: {content_dir}; text-align: {content_align}; white-space: pre-wrap; color: var(--text-secondary);">
            {full_text}
        </div>
        """, unsafe_allow_html=True)
        st.download_button(t["btn_dl_transcription"], full_text, file_name="transcript.txt")
