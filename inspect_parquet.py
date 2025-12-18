import pandas as pd

try:
    df = pd.read_parquet('newModelDataset/dataset.parquet')
    print("Columns:", list(df.columns))
    # print sample of first row for all columns to see what keys to access
    print("First row keys:", df.iloc[0].to_dict().keys())
    print("First row sample values:", {k: str(v)[:50] for k, v in df.iloc[0].to_dict().items()})
except Exception as e:
    print(f"Error reading parquet: {e}")
