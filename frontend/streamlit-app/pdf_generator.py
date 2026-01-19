import os
import re
from fpdf import FPDF
import arabic_reshaper
from bidi.algorithm import get_display

class PDFGenerator(FPDF):
    def __init__(self):
        # DISABLE native text shaping to avoid library conflicts/errors.
        # We will handle shaping manually for 100% control.
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
        
        # Load Fonts
        # Arial is standard and supports Arabic visuals
        font_path = "C:/Windows/Fonts/arial.ttf"
        font_path_bd = "C:/Windows/Fonts/arialbd.ttf"
        
        try:
            if os.path.exists(font_path):
                self.add_font("Arial", "", font_path)
            if os.path.exists(font_path_bd):
                self.add_font("Arial", "B", font_path_bd)
        except Exception as e:
            print(f"Warning: Could not load Arial fonts: {e}")

    def header(self):
        self.set_font("Arial", "B", 15)
        self.set_text_color(100, 100, 100)
        self.cell(self.epw, 10, "Sidecut AI Summary", ln=True, align="C")
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "", 8)
        self.set_text_color(128)
        self.cell(self.epw, 10, f"Page {self.page_no()}", align="C")

    def add_summary(self, title, text, language="en"):
        self.add_page()
        self.set_text_color(0, 0, 0)
        
        # Detect Arabic
        if language != "ar" and re.search(r'[\u0600-\u06FF]', title + text):
            language = "ar"
            
        align = "R" if language == "ar" else "L"
        
        # Title
        self.set_font("Arial", "B", 16)
        # Use simple multi_cell for title (usually short enough), or manual if safe
        if language == "ar":
            self.safe_multi_cell_arabic(title, align, is_title=True)
        else:
            self.multi_cell(self.epw, 10, title, align="C")
            
        self.ln(5)
        self.set_draw_color(200, 200, 200)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(10)
        
        # Body
        self.set_font("Arial", "", 12)
        
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                self.ln(3)
                continue
            
            # Bullet Logic
            is_bullet = line.startswith("•") or line.startswith("-")
            indent = 0
            if is_bullet:
                 indent = 5
            
            # Apply indentation
            current_x = self.get_x()
            # For RTL, we technically should indent from right, but FPDF coordinate system is LTR.
            # So indenting 'x' moves start to the right, which is fine for visual indent.
            if indent > 0:
                self.set_x(current_x + indent)
                
            avail_width = self.epw - indent
            
            if language == "ar":
                self.safe_multi_cell_arabic(line, align, width=avail_width)
            else:
                self.multi_cell(avail_width, 8, line, align=align)
                
            # Reset X after bullet (autoflow handles it usually, but safety check)
            self.set_x(self.l_margin)


    def safe_multi_cell_arabic(self, text, align, is_title=False, width=None):
        """
        Manually wraps Arabic text to ensure it fits width AND renders correctly.
        """
        if width is None:
            width = self.epw
            
        height = 10 if is_title else 8
        
        words = text.split()
        current_line = []
        
        for word in words:
            # Try adding word to current line
            test_line_words = current_line + [word]
            test_line = " ".join(test_line_words)
            
            # Reshape to measure ACTUAL width of shaped text
            reshaped_test = self.reshape_text(test_line)
            
            if self.get_string_width(reshaped_test) <= width:
                current_line.append(word)
            else:
                # Line is full, print current_line
                if current_line:
                    final_line_text = " ".join(current_line)
                    final_reshaped = self.reshape_text(final_line_text)
                    self.cell(width, height, final_reshaped, ln=True, align=align)
                    current_line = [word] # Start new line with current word
                else:
                    # Single word is too wide (unlikely but possible)
                    reshaped_word = self.reshape_text(word)
                    self.cell(width, height, reshaped_word, ln=True, align=align)
                    current_line = []
        
        # Print remaining
        if current_line:
            final_line_text = " ".join(current_line)
            final_reshaped = self.reshape_text(final_line_text)
            self.cell(width, height, final_reshaped, ln=True, align=align)

    def reshape_text(self, text):
        try:
            # 1. Connect letters
            reshaped_text = arabic_reshaper.reshape(text)
            # 2. Reorder for RTL (Visual)
            bidi_text = get_display(reshaped_text)
            return bidi_text
        except Exception:
            return text

def generate_pdf_bytes(filename, summary_text, language="en"):
    pdf = PDFGenerator()
    pdf.add_summary(f"Summary: {filename}", summary_text, language)
    return bytes(pdf.output(dest="S"))

def clean_html_for_pdf(raw_text: str) -> str:
    if not raw_text: return ""
    # Standard cleanup
    formatted_text = raw_text.replace("<li>", "• ").replace("</li>", "\n") \
                             .replace("<ul>", "").replace("</ul>", "\n")
    clean_text = re.sub(r'<[^>]+>', '', formatted_text)
    clean_text = clean_text.replace("&quot;", '"').replace("&amp;", "&") \
                           .replace("&lt;", "<").replace("&gt;", ">").replace("&nbsp;", " ")
    lines = [line.strip() for line in clean_text.split('\n') if line.strip()]
    return "\n".join(lines)
