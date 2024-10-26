import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import time
import onnx
import tempfile

class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train_model(model, train_loader, val_loader, device, num_epochs=3):
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=2e-5
    )

    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    best_val_accuracy = 0
    best_model_state = None

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print('-' * 10)
        
        model.train()
        total_train_loss = 0
        correct_train_preds = 0
        total_train_samples = 0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            model.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            total_train_loss += loss.item()

            preds = torch.argmax(outputs.logits, dim=1)
            correct_train_preds += (preds == labels).sum().item()
            total_train_samples += labels.size(0)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_loader)
        train_accuracy = correct_train_preds / total_train_samples

        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Training Accuracy: {train_accuracy:.4f}")

        # Validation
        model.eval()
        total_val_loss = 0
        correct_val_preds = 0
        total_val_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                total_val_loss += loss.item()

                preds = torch.argmax(outputs.logits, dim=1)
                correct_val_preds += (preds == labels).sum().item()
                total_val_samples += labels.size(0)

        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = correct_val_preds / total_val_samples

        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print()

        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict().copy()

    # Load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Using best model with validation accuracy: {best_val_accuracy:.4f}")

    return model

def convert_to_onnx(model, tokenizer, device, onnx_path="twitter_bert_tiny.onnx"):
    """Convert PyTorch model to ONNX format"""
    print("Starting ONNX conversion...")
    model.eval()
    
    # Prepare dummy input
    dummy_input = "This is a sample tweet for ONNX conversion."
    inputs = tokenizer(
        dummy_input,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    
    input_names = ["input_ids", "attention_mask"]
    output_names = ["logits"]
    
    # Move inputs to the same device as the model
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    try:
        # First try with opset 14
        print("Attempting ONNX export with opset 14...")
        torch.onnx.export(
            model,
            (inputs['input_ids'], inputs['attention_mask']),
            onnx_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={
                'input_ids': {0: 'batch_size'},
                'attention_mask': {0: 'batch_size'},
                'logits': {0: 'batch_size'}
            },
            do_constant_folding=True,
            opset_version=14,
            verbose=False
        )
    except Exception as e:
        print(f"Export with opset 14 failed: {str(e)}")
        try:
            # Try with opset 16 if 14 fails
            print("Attempting ONNX export with opset 16...")
            torch.onnx.export(
                model,
                (inputs['input_ids'], inputs['attention_mask']),
                onnx_path,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes={
                    'input_ids': {0: 'batch_size'},
                    'attention_mask': {0: 'batch_size'},
                    'logits': {0: 'batch_size'}
                },
                do_constant_folding=True,
                opset_version=16,
                verbose=False
            )
        except Exception as e:
            print(f"Export with opset 16 failed: {str(e)}")
            raise

    print(f"Model exported to {onnx_path}")
    
    # Verify the exported model
    try:
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model verification successful!")
    except Exception as e:
        print(f"ONNX model verification failed: {str(e)}")
        raise
    
    return onnx_path

def clean_data(df):
    """Clean the dataset by handling NaN values and standardizing sentiment labels"""
    print("Initial shape:", df.shape)
    
    # Drop rows with NaN values
    df = df.dropna(subset=['text', 'Sentiment'])
    print("Shape after dropping NaN:", df.shape)
    
    # Standardize sentiment labels
    df['Sentiment'] = df['Sentiment'].str.strip()
    
    print("Unique sentiment values:", df['Sentiment'].unique())
    
    df['Sentiment'] = df['Sentiment'].str.lower()
    sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    df['Sentiment'] = df['Sentiment'].map(sentiment_map)
    df = df.dropna(subset=['Sentiment'])
    
    print("Final shape after cleaning:", df.shape)
    return df

def main():
    print("Loading data...")
    df = pd.read_csv('flight_data_cleaned.csv')
    
    print("Cleaning data...")
    df = clean_data(df)
    
    print("\nClass distribution:")
    print(df['Sentiment'].value_counts())
    
    print("\nSplitting data...")
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].values, 
        df['Sentiment'].values,
        test_size=0.2,
        random_state=42,
        stratify=df['Sentiment']
    )

    print("\nInitializing model and tokenizer...")
    model_name = "prajjwal1/bert-tiny"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3  # Negative, Neutral, Positive
    )

    train_dataset = TweetDataset(train_texts, train_labels, tokenizer)
    val_dataset = TweetDataset(val_texts, val_labels, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    model.to(device)

    print("\nStarting training...")
    model = train_model(model, train_loader, val_loader, device, num_epochs=3)

    print("\nConverting to ONNX format...")
    convert_to_onnx(model, tokenizer, device)
    print("Training completed successfully!")

if __name__ == "__main__":
    main()