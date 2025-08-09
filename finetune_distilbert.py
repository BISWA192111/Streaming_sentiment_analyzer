import pandas as pd
from datasets import Dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import os

print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(0))
else:
    print("Using CPU only.")

if os.path.exists('./distilbert-finetuned-patient/pytorch_model.bin'):
    print("Fine-tuned model already exists. Skipping training.")
    exit(0)

# 1. Load and preprocess your data
csv_path = r'C:\\Users\\USER\\Downloads\\live transcription\\output_cleaned.csv'
df = pd.read_csv(csv_path)
df = df.dropna(subset=['statement', 'label'])
df = df[df['statement'].apply(lambda x: isinstance(x, str) and len(x.strip()) > 0)]

# 1a. Compute top 1500 words by TF-IDF
vectorizer = TfidfVectorizer(max_features=1500)
vectorizer.fit(df['statement'])
top_words = set(vectorizer.get_feature_names_out())

def filter_statement(text):
    words = text.split()
    filtered = [w for w in words if w.lower() in top_words]
    return ' '.join(filtered)

df['filtered_statement'] = df['statement'].apply(filter_statement)

# After filtering, sample 1500 rows for training
if len(df) > 5000:
    df = df.sample(n=5000, random_state=42)

# Get unique labels and create label mapping for model (convert to 0,1,2 for model)
unique_labels = sorted(df['label'].unique())  # Sort to ensure consistent order: [-1, 0, 1]
label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
id_to_label = {idx: label for label, idx in label_to_id.items()}

print(f"Dataset size: {len(df)}")
print(f"Original labels found: {unique_labels}")
print(f"Label mapping: {label_to_id}")
print(f"Label distribution: {df['label'].value_counts().sort_index().to_dict()}")

# Convert labels to integers for the model (0,1,2)
df['label_id'] = df['label'].map(label_to_id)

# 2. Split into train/validation
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label_id'], random_state=42)
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

print(f"Train size: {len(train_df)}, Validation size: {len(val_df)}")

# 3. Tokenize
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
def tokenize(batch):
    return tokenizer(batch['filtered_statement'], padding='max_length', truncation=True, max_length=128)
train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)

# 4. Set format for PyTorch
train_dataset = train_dataset.rename_column('label_id', 'labels')
val_dataset = val_dataset.rename_column('label_id', 'labels')
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# 5. Model
num_labels = len(unique_labels)
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels)

# 6. Training arguments
training_args = TrainingArguments(
    output_dir='./distilbert-finetuned-patient',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy='epoch',
    save_strategy='epoch',
    logging_dir='./logs',
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
)

# 7. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# 8. Train!
print("Starting fine-tuning...")
trainer.train()

# 9. Evaluate the fine-tuned model
print("\n" + "="*50)
print("EVALUATION RESULTS")
print("="*50)

# Get predictions on validation set
predictions = trainer.predict(val_dataset)
preds = np.argmax(predictions.predictions, axis=-1)
true_labels = val_dataset['labels']

# Calculate metrics
accuracy = accuracy_score(true_labels, preds)
precision, recall, f1, _ = precision_recall_fscore_support(true_labels, preds, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Detailed classification report with original labels
target_names = [f"Label_{id_to_label[i]}" for i in range(len(unique_labels))]
print("\nDetailed Classification Report:")
print(classification_report(true_labels, preds, target_names=target_names))

# 10. Save your model and label mapping
model.save_pretrained('./distilbert-finetuned-patient')
tokenizer.save_pretrained('./distilbert-finetuned-patient')

# Save label mapping for later use
import json
# Convert all keys and values to int or str for JSON compatibility
label_to_id_json = {str(int(k)): int(v) for k, v in label_to_id.items()}
id_to_label_json = {str(int(k)): int(v) for k, v in id_to_label.items()}
with open('./distilbert-finetuned-patient/label_mapping.json', 'w') as f:
    json.dump({'label_to_id': label_to_id_json, 'id_to_label': id_to_label_json}, f)

print('\nFine-tuning complete! Model saved to ./distilbert-finetuned-patient')
print(f"Final Model Accuracy: {accuracy:.4f}")
print(f"Original labels: {list(label_to_id.keys())}") 