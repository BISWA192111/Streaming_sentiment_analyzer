import pandas as pd
from datasets import Dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

# 1. Load and preprocess your data
df = pd.read_csv("C:\\Users\\USER\\Downloads\\live transcription\\synonyms_labeled_before_during_after.csv")  # columns: word, phase

# Convert each word to a simple sentence (or just use the word)
df['sentence'] = df['word'].astype(str)
label_map = {'before': 0, 'during': 1, 'after': 2}
df['phase'] = df['phase'].str.lower()
df['label'] = df['phase'].map(label_map)
print("Unique phase values:", df['phase'].unique())
print("Rows before dropping NaN labels:", len(df))
df = df.dropna(subset=['label'])
df['label'] = df['label'].astype(int)
print("Rows after dropping NaN labels:", len(df))
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# 2. Tokenize
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
def tokenize(batch):
    return tokenizer(batch['sentence'], padding='max_length', truncation=True, max_length=128)
train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# 3. Model
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)

# 4. Training arguments
training_args = TrainingArguments(
    output_dir='./distilbert-finetuned-phase',
    num_train_epochs=6,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy='epoch',
    save_strategy='epoch',
    logging_dir='./logs',
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
)

# 5. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# 6. Train!
trainer.train()

# 7. Save your model
model.save_pretrained('./distilbert-finetuned-phase')
tokenizer.save_pretrained('./distilbert-finetuned-phase')
print('Fine-tuned phase model saved to ./distilbert-finetuned-phase') 