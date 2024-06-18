import os
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from datasets import load_dataset

def fine_tune_model(model_name, data_path, output_dir, learning_rate, epochs, batch_size):
    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Load and process the dataset
    dataset = load_dataset('json', data_files={'train': data_path})
    def preprocess_function(examples):
        model_inputs = tokenizer(examples['text'], max_length=512, truncation=True)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples['summary'], max_length=128, truncation=True)
        model_inputs['labels'] = labels['input_ids']
        return model_inputs

    processed_dataset = dataset.map(preprocess_function, batched=True)
    
    # Data collator used for padding and converting to tensors
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=epochs,
        predict_with_generate=True,
    )

    # Initialize the trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset['train'],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Start training
    trainer.train()

    # Save the model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Fine-tune LLaVA-Med model on an augmented dataset.')
    parser.add_argument('--model-name', type=str, required=True, help='Pretrained model name or path')
    parser.add_argument('--data-path', type=str, required=True, help='Path to the training data file')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save the fine-tuned model')
    parser.add_argument('--learning-rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Training batch size')
    
    args = parser.parse_args()
    fine_tune_model(
        model_name=args.model_name,
        data_path=args.data_path,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
