import pandas as pd
import sys
from pathlib import Path
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    DataCollatorForSeq2Seq, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer
)
from datasets import Dataset
import torch

# Configuration
MODEL_CHECKPOINT = "google/mt5-small" # Good balance for multilingual/resource
MAX_INPUT_LENGTH = 1024
MAX_TARGET_LENGTH = 256
BATCH_SIZE = 4
EPOCHS = 3
LEARNING_RATE = 2e-5

def preprocess_function(examples, tokenizer):
    inputs = ["summarize: " + doc for doc in examples["subtitles"]]
    model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"], max_length=MAX_TARGET_LENGTH, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def train():
    print("Loading prepared dataset...")
    # Check for the generated dataset
    data_path = Path("newModelDataset/dataset_with_summaries.parquet")
    if not data_path.exists():
        data_path = Path("../../newModelDataset/dataset_with_summaries.parquet")
    
    if not data_path.exists():
        print("Error: dataset_with_summaries.parquet not found. Run generate_data.py first.")
        sys.exit(1)
        
    df = pd.read_parquet(data_path)
    # Ensure all data is string
    df = df.dropna(subset=['subtitles', 'summary'])
    df['subtitles'] = df['subtitles'].astype(str)
    df['summary'] = df['summary'].astype(str)
    
    hf_dataset = Dataset.from_pandas(df)
    
    # Train/Test Split
    # Ensure at least 1 test sample if dataset is small
    test_size = 0.1
    if len(hf_dataset) < 20:
        test_size = 0.5 # Force split for tiny datasets
        
    split_dataset = hf_dataset.train_test_split(test_size=test_size)
    
    print("Initializing Model and Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)
    
    print("Tokenizing dataset...")
    tokenized_datasets = split_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=split_dataset["train"].column_names
    )
    
    # Setup Training
    model_name = MODEL_CHECKPOINT.split("/")[-1]
    output_dir = Path(f"backend/models/custom_{model_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch", # Updated from evaluation_strategy
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=EPOCHS,
        predict_with_generate=True,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        fp16=torch.cuda.is_available(), # Use mixed precision if GPU available
        push_to_hub=False,
    )
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    print("Starting Training...")
    trainer.train()
    
    print("Saving Metadata and Model...")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Model saved to {output_dir}")

if __name__ == "__main__":
    train()
