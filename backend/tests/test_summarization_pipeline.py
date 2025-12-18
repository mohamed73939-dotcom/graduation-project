from summarizer import LectureSummarizer

def test_summarizer_nonempty_output():
    s = LectureSummarizer()
    text = "هذه جملة عربية طويلة نوعاً ما تحتوي على معلومات عديدة. هذا مثال آخر للجملة. نريد التلخيص هنا."
    summary = s.summarize(text, max_length=100, min_length=30, language='ar')
    assert summary
    assert len(summary.strip()) > 0