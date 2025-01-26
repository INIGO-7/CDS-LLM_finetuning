from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset
import os

# Step 1: Load and Prepare the Dataset
def preprocess_data(train_path, val_path):
    """Prepares the dataset for T5 fine-tuning."""
    # Load the dataset from JSONL
    dataset = load_dataset("json", data_files={"train": train_path, "val": val_path})

    # Tokenizer and Model
    model_name = "t5-small"  # You can change this to "t5-base" or larger variants if needed
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        # Tokenize inputs and outputs
        inputs = tokenizer(
            examples["input"],
            max_length=512,
            truncation=True,
            padding="max_length"
        )
        outputs = tokenizer(
            examples["output"],
            max_length=512,
            truncation=True,
            padding="max_length"
        )
        inputs["labels"] = outputs["input_ids"]
        return inputs

    # Tokenize the dataset
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Return tokenized dataset, tokenizer, and model
    return tokenized_dataset, tokenizer, model_name

# Step 2: Configure Training Arguments
def setup_training_args(output_dir="./results"):
    """Defines the training arguments."""
    return Seq2SeqTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        save_strategy="steps",
        logging_dir=f"{output_dir}/logs",
        learning_rate=3e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        save_steps=1000,
        save_total_limit=2,
        predict_with_generate=True,
        logging_steps=500,
        report_to=["tensorboard"]
    )

# Step 3: Fine-Tune the Model
def fine_tune_t5(train_path, val_path, output_dir="./output"):
    """Fine-tunes the T5 model on the provided dataset."""
    # Prepare the dataset, tokenizer, and model
    tokenized_dataset, tokenizer, model_name = preprocess_data(train_path, val_path)

    # Load the pre-trained T5 model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Configure training arguments
    training_args = setup_training_args(output_dir)

    # Initialize the Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["val"],
        tokenizer=tokenizer,
    )

    # Train the model
    trainer.train()

    # Save the final model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


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
