
import unittest
import sys
import os

# Add project root to path so we can import frontend modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from frontend.pdf_generator import clean_html_for_pdf

class TestPDFGenerator(unittest.TestCase):
    def test_clean_html_bullets(self):
        html = "<ul><li>First</li><li>Second</li></ul>"
        expected = "• First\n• Second"
        self.assertEqual(clean_html_for_pdf(html), expected)

    def test_clean_html_tags(self):
        html = "<p>Hello <b>World</b></p><br>"
        expected = "Hello World"
        self.assertEqual(clean_html_for_pdf(html), expected)

    def test_clean_html_entities(self):
        html = "Fish &amp; Chips"
        expected = "Fish & Chips"
        self.assertEqual(clean_html_for_pdf(html), expected)

    def test_clean_complex(self):
        html = "<ul><li>Item 1 <span class='highlight'>Important</span></li><li>Item 2</li></ul>"
        expected = "• Item 1 Important\n• Item 2"
        self.assertEqual(clean_html_for_pdf(html), expected)

    def test_empty(self):
        self.assertEqual(clean_html_for_pdf(None), "")
        self.assertEqual(clean_html_for_pdf(""), "")

if __name__ == '__main__':
    unittest.main()
