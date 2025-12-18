import pandas as pd
import sys
import os
from pathlib import Path

# Add backend directory to sys.path to allow imports
backend_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(backend_dir))

from summarizer import LectureSummarizer
from tqdm import tqdm

def generate_teacher_summaries():
    print("Loading dataset...")
    dataset_path = Path("newModelDataset/dataset.parquet")
    if not dataset_path.exists():
        # Handle case where script is run from backend dir
        dataset_path = Path("../../newModelDataset/dataset.parquet")
        
    if not dataset_path.exists():
        print(f"Error: Could not find dataset at {dataset_path}")
        return

    df = pd.read_parquet(dataset_path)
    print(f"Dataset loaded. Shape: {df.shape}")

    # Initialize Teacher Model (Using the default existing model)
    print("Initializing Teacher Model (Standard Summarizer)...")
    summarizer = LectureSummarizer() 

    summaries = []
    
    # Process only first 50 rows for demonstration/speed if dataset is huge, 
    # or all if it's reasonable. Let's do all but show progress.
    # If it's too large, we might want to batch it.
    
    # ADDED: Limit for testing in interactive session
    LIMIT = 5
    if LIMIT:
        df = df.head(LIMIT)
        print(f"Generating summaries for subset (Limit: {LIMIT} for testing)...")
    
    # Process rows
    for index, row in tqdm(df.iterrows(), total=len(df)):
        # removed break since we sliced df
        pass # loop continues normally
        
        transcript = row['subtitles']
        if not transcript or not isinstance(transcript, str):
            summaries.append("")
            continue
            
        try:
            # We use the existing high-quality pipeline
            # Using 'auto' language or forcing english if topic implies it?
            # Let's detect or default to english for this dataset as it looks mixed/english
            # The summarizer handles detection internally if api calls it, but here we call summarize directly.
            # We'll pass 'en' as default or detect.
            # Actually, let's pass 'en' since the sample looked English, but to be safe we can let it handle it 
            # if we had detection. For now, strict 'en' is safer for consistent training targets 
            # unless the dataset is multilingual. The sample was English.
            summary, _ = summarizer.hierarchical_summarize(transcript, language='en')
            summaries.append(summary)
        except Exception as e:
            print(f"Error processing row {index}: {e}")
            summaries.append("")

    df['summary'] = summaries
    
    # Filter out failed ones
    df = df[df['summary'] != ""]
    print(f"Final valid dataset shape: {df.shape}")

    # Save
    output_path = dataset_path.with_name("dataset_with_summaries.parquet")
    df.to_parquet(output_path)
    print(f"Saved labelled dataset to {output_path}")

if __name__ == "__main__":
    generate_teacher_summaries()
