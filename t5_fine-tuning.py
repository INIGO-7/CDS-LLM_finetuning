import os
import torch
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from torch.utils.data import Dataset
import json

def chunk_sequence(sequence, tokenizer, max_length, stride):
    tokenized = tokenizer(
        sequence,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=False,
    )
    return [
        {"input_ids": chunk, "attention_mask": tokenized["attention_mask"][i]}
        for i, chunk in enumerate(tokenized["input_ids"])
    ]

def preprocess_with_chunking(dataset, tokenizer, max_length=512, stride=256):
    chunked_data = []
    for item in dataset:
        chunks = chunk_sequence(item["input"], tokenizer, max_length, stride)
        for chunk in chunks:
            chunked_data.append({
                **chunk,
                "labels": tokenizer(
                    item["output"],
                    max_length=128,
                    truncation=True,
                    padding="max_length",
                )["input_ids"],
            })
    return chunked_data

class ChunkedDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        labels = [x if x != self.tokenizer.pad_token_id else -100 for x in item["labels"]]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

def fine_tune_t5(train_path, val_path, output_dir, max_length=512, stride=128):
    tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=False)
    
    # Load and preprocess data
    with open(train_path, "r") as f:
        train_data = preprocess_with_chunking(json.load(f), tokenizer, max_length, stride)
    with open(val_path, "r") as f:
        val_data = preprocess_with_chunking(json.load(f), tokenizer, max_length, stride)

    # Create datasets
    train_dataset = ChunkedDataset(train_data, tokenizer)
    val_dataset = ChunkedDataset(val_data, tokenizer)

    # Model and data collator
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy="steps",
        eval_steps=500,
        save_steps=1000,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        logging_dir=f"{output_dir}/logs",
        predict_with_generate=True,
        save_total_limit=2,
    )

    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Train and save
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

# Main function to trigger fine-tuning
def main():
    # Run the fine-tuning process
    datasets_dir = "output"
    output_dir = "output"
    train_file = "train_set.json"
    val_file = "val_set.json"
    train_path = os.path.join(datasets_dir, train_file)
    val_path = os.path.join(datasets_dir, val_file)
    fine_tune_t5(train_path, val_path, output_dir)

if __name__ == "__main__":
    main()
