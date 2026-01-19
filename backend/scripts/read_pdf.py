import sys

try:
    import pypdf
    reader_cls = pypdf.PdfReader
except ImportError:
    try:
        import PyPDF2
        reader_cls = PyPDF2.PdfReader
    except ImportError:
        print("No PDF library found (pypdf or PyPDF2)")
        sys.exit(1)

try:
    reader = reader_cls("d:/_Lecture Video Summarizer__/Lecture Video Summarizer1111(1).pdf")
    text = ""
    # Extract from all pages or a reasonable limit
    for i, page in enumerate(reader.pages):
        text += f"\n--- Page {i+1} ---\n"
        text += page.extract_text()
        if i > 20: # Limit to 20 pages to avoid huge output
            break
    print(text)
except Exception as e:
    print(f"Error reading PDF: {e}")
