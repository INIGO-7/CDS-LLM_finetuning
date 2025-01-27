from transformers import (
    PreTrainedTokenizerFast,
    T5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    TrainerCallback
)
import os
import json
import torch
import logging
from torch.utils.data import Dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def chunk_sequence(input_text, tokenizer, max_length=512, stride=256):
    """Chunk sequences while preserving task context"""
    # Tokenize without padding
    tokens = tokenizer.encode(input_text, add_special_tokens=False)
    
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + max_length
        chunk = tokens[start:end]
        
        # Add task token if missing
        if chunk[0] != tokenizer.convert_tokens_to_ids("<task>"):
            chunk = tokenizer.convert_tokens_to_ids("<task>") + chunk
        
        # Pad if necessary
        if len(chunk) < max_length:
            chunk += [tokenizer.pad_token_id] * (max_length - len(chunk))
        else:
            chunk = chunk[:max_length]
        
        chunks.append({
            "input_ids": chunk,
            "attention_mask": [1]*len(chunk) + [0]*(max_length - len(chunk))
        })
        start += max_length - stride
    
    return chunks

def preprocess_with_chunking(dataset, tokenizer, max_length=512, stride=256):
    """Preprocess with DNA-aware chunking"""
    chunked_data = []
    for idx, item in enumerate(dataset):
        if idx % 100 == 0:
            logger.info(f"Processing item {idx}/{len(dataset)}")
        
        # Format input with task prefix
        input_text = f"<task>DetectCDS|{item['input']}"
        output_text = item['output_positions']
        
        # Generate chunks
        chunks = chunk_sequence(input_text, tokenizer, max_length, stride)
        
        # Tokenize output positions once per original sequence
        labels = tokenizer.encode(output_text, max_length=128, truncation=True)
        
        # Create chunked items
        for chunk in chunks:
            chunked_data.append({
                "input_ids": chunk["input_ids"],
                "attention_mask": chunk["attention_mask"],
                "labels": labels
            })
    
    logger.info(f"Total chunks generated: {len(chunked_data)}")
    return chunked_data

class PositionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        logger.info(f"Initialized dataset with {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        labels = item["labels"] + [self.tokenizer.pad_token_id] * (128 - len(item["labels"]))
        labels = [l if l != self.tokenizer.pad_token_id else -100 for l in labels]
        
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }

class TrainingMonitor(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero:
            logger.info(f"Step {state.global_step} - Loss: {logs.get('loss', 'NA'):.4f}")
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            logger.info("Validation Results:")
            for k, v in metrics.items():
                logger.info(f"  {k}: {v:.4f}")

def fine_tune_t5(train_path, val_path, output_dir, tokenizer_path):
    # Initialize custom tokenizer
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    tokenizer.pad_token = "<pad>"
    tokenizer.unk_token = "<unk>"
    tokenizer.add_special_tokens({"additional_special_tokens": ["<task>"]})
    tokenizer.model_max_length = 512
    
    logger.info("Initialized DNA tokenizer with vocabulary size: %d", len(tokenizer))
    
    # Load and preprocess data
    logger.info("Loading training data...")
    with open(train_path) as f:
        train_data = preprocess_with_chunking(json.load(f), tokenizer)
    
    logger.info("Loading validation data...")
    with open(val_path) as f:
        val_data = preprocess_with_chunking(json.load(f), tokenizer)

    # Create datasets
    train_dataset = PositionDataset(train_data, tokenizer)
    val_dataset = PositionDataset(val_data, tokenizer)

    # Initialize model with resized embeddings
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    model.resize_token_embeddings(len(tokenizer))
    logger.info("Model initialized with resized embeddings")

    # Training configuration
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=1000,
        learning_rate=3e-4,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        predict_with_generate=True,
        generation_max_length=128,
        logging_dir=f"{output_dir}/logs",
        save_total_limit=2,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        tokenizer=tokenizer,
        callbacks=[TrainingMonitor()]
    )

    # Start training
    logger.info("Beginning training...")
    trainer.train()
    
    # Save final model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Training complete! Model saved to %s", output_dir)

def main():
    datasets_dir = "output"
    output_dir = "models"
    train_path = os.path.join(datasets_dir, "train_set.json")
    val_path = os.path.join(datasets_dir, "val_set.json")
    tokenizer_path = os.path.join(output_dir, "dna_tokenizer.json")
    
    fine_tune_t5(train_path, val_path, output_dir, tokenizer_path)

if __name__ == "__main__":
    main()