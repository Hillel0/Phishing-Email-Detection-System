import torch
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset as HFDataset, load_from_disk


device = torch.device("xpu" if torch.xpu.is_available() else "cpu")

# Load pre-trained DistilBERT tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
MODEL_PATH = "./saved_model"
label_encoder = LabelEncoder()

# Ensure label_encoder is fitted, even if loading from disk
def load_label_encoder(csv_file="Phishing_Email.csv"):
    df = pd.read_csv(csv_file, index_col=0)
    df['Email Type'] = df['Email Type'].astype(str)
    label_encoder.fit(df['Email Type'])  # Fit label encoder globally




def preprocess_and_save_data(csv_file, train_path="./train_dataset", test_path="./test_dataset"):
    """
    Loads, preprocesses, tokenizes, and saves the dataset to disk.
    Skips preprocessing if dataset is already saved.
    """
    if os.path.exists(train_path) and os.path.exists(test_path):
        print("✔ Preprocessed dataset found. Loading from disk...")
        load_label_encoder()
        return load_from_disk(train_path), load_from_disk(test_path)

    print("⚡ Preprocessing dataset...")

    # Load CSV file (removing extra index column)
    df = pd.read_csv(csv_file, index_col=0)

    # Ensure 'Email Text' is a string and remove NaNs
    df['Email Text'] = df['Email Text'].astype(str).fillna("")

    # Encode labels (Safe Email = 0, Phishing Email = 1)
    
    df['Label'] = label_encoder.fit_transform(df['Email Type'])

    # Split into train & test
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df['Email Text'].tolist(), df['Label'].tolist(), test_size=0.2, random_state=42
    )

    # Convert to Hugging Face dataset format
    train_dataset = HFDataset.from_dict({"text": train_texts, "label": train_labels})
    test_dataset = HFDataset.from_dict({"text": test_texts, "label": test_labels})

    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    # Tokenize dataset
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    # Save pre-tokenized dataset
    train_dataset.save_to_disk(train_path)
    test_dataset.save_to_disk(test_path)
    print("✔ Preprocessing complete. Saved dataset to disk.")

    return train_dataset, test_dataset

# ========== 1. Load Preprocessed Data ==========

csv_file = "Phishing_Email.csv"
train_dataset, test_dataset = preprocess_and_save_data(csv_file)

# ========== 2. Load or Train the Model ==========
if os.path.exists(MODEL_PATH):
    print("✔ Saved model found. Loading model from disk...")
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
else:
    print("⚡ No saved model found. Initializing new model...")
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

model.to(device)

# ========== 3. Define Training Parameters ==========
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=32, 
    per_device_eval_batch_size=32,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# ========== 4. Train or Skip If Already Saved ==========
if not os.path.exists(MODEL_PATH):
    print("⚡ Training model...")
    trainer.train()
    print("✔ Training complete. Saving model to disk...")
    model.save_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH)
else:
    print("✔ Skipping training. Using saved model.")


# ========== 5. Evaluate the Model ==========
trainer.evaluate()

# ========== 6. Make Predictions ==========
def predict_email(email_text):
    inputs = tokenizer(email_text, return_tensors="pt", truncation=True, padding="max_length", max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
    return label_encoder.inverse_transform([prediction])[0]

# Test the model with a sample email
sample_email = "URGENT: Your PayPal account has been locked. Click here to verify."
print("Prediction:", predict_email(sample_email))
