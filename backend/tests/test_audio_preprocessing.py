import os
from pathlib import Path
from audio_preprocessing import AudioPreprocessor

def test_preprocess_and_chunk_smoke():
    # This is a smoke test; requires a small wav file in tests/data/sample.wav
    sample = Path("tests/data/sample.wav")
    if not sample.exists():
        return  # Skip if test asset missing
    ap = AudioPreprocessor()
    result = ap.preprocess_and_chunk(str(sample), chunk_duration=5, chunk_overlap=1.0)
    assert "clean_audio" in result
    assert "chunks" in result
    assert len(result["chunks"]) >= 1