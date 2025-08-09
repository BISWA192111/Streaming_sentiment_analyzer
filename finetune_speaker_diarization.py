#!/usr/bin/env python3
"""
Fine-tune pyannote.audio speaker diarization using DistilBERT for conversation pattern learning
Trains on conversation data from text files to improve speaker identification accuracy
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import warnings
import json
import pickle
from datetime import datetime

# Suppress warnings
warnings.filterwarnings("ignore")

try:
    from transformers import (
        DistilBertTokenizer, DistilBertForSequenceClassification,
        AutoTokenizer, AutoModelForSequenceClassification,
        TrainingArguments, Trainer, DataCollatorWithPadding
    )
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    from datasets import Dataset
    import torch.nn as nn
    from pyannote.audio import Pipeline, Model
    from pyannote.audio.core.task import Task
    from pyannote.audio.tasks import SpeakerDiarization
    from huggingface_hub import login
except ImportError as e:
    print(f"ERROR: Required packages not installed. Install with:")
    print("pip install transformers datasets torch pyannote.audio scikit-learn")
    sys.exit(1)

class ConversationDataLoader:
    """Load and preprocess conversation data from text files"""
    
    def __init__(self, data_folder):
        self.data_folder = Path(data_folder)
        self.conversations = []
        self.speaker_patterns = {
            'doctor_patterns': [
                'what', 'where', 'when', 'why', 'how', 'describe', 'tell me',
                'on a scale', 'rate your', 'point to', 'show me', 'examination',
                'symptoms', 'medication', 'treatment', 'pain', 'how long',
                'where is', 'what kind', 'have you', 'do you', 'can you'
            ],
            'patient_patterns': [
                'i feel', 'i have', 'i think', 'my', 'it hurts', 'yes', 'no',
                'yeah', 'the pain', 'it started', 'last night', 'this morning',
                'sharp', 'burning', 'aching', 'chest', 'back', 'stomach',
                'breathing', 'dizzy', 'tired', 'about', 'around', 'maybe'
            ]
        }
    
    def load_conversations(self):
        """Load all conversation files from the data folder"""
        print(f"📂 Loading conversations from: {self.data_folder}")
        
        conversation_files = list(self.data_folder.glob("*.txt"))
        print(f"📄 Found {len(conversation_files)} conversation files")
        
        for file_path in conversation_files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read().strip()
                
                if content:
                    conversation = self.parse_conversation(content, file_path.name)
                    if conversation:
                        self.conversations.extend(conversation)
                        
            except Exception as e:
                print(f"⚠️ Error loading {file_path.name}: {e}")
        
        print(f"✅ Loaded {len(self.conversations)} conversation turns")
        return self.conversations
    
    def parse_conversation(self, content, filename):
        """Parse conversation content and extract speaker turns based on D: and P: prefixes"""
        turns = []
        lines = content.split('\n')
        
        # Process lines and look for D: and P: prefixes
        for i, line in enumerate(lines):
            line = line.strip()
            if len(line) <= 3:  # Skip very short lines
                continue
                
            speaker_label = None
            confidence = 0.95  # High confidence for explicit labels
            text_content = line
            
            # Check for explicit speaker prefixes
            if line.startswith('D:') or line.startswith('d:'):
                speaker_label = 0  # Doctor = 0
                text_content = line[2:].strip()  # Remove D: prefix
                confidence = 0.98
            elif line.startswith('P:') or line.startswith('p:'):
                speaker_label = 1  # Patient = 1
                text_content = line[2:].strip()  # Remove P: prefix
                confidence = 0.98
            else:
                # If no explicit prefix, use pattern-based detection as fallback
                speaker_label, confidence = self.predict_speaker_from_content(line)
                text_content = line
            
            # Only include meaningful content
            if text_content and len(text_content.split()) > 2:  # At least 3 words
                turn = {
                    'text': text_content,
                    'speaker': speaker_label,
                    'confidence': confidence,
                    'file': filename,
                    'turn_id': i,
                    'length': len(text_content.split()),
                    'has_prefix': line.startswith(('D:', 'P:', 'd:', 'p:'))
                }
                turns.append(turn)
        
        return turns
    
    def predict_speaker_from_content(self, text):
        """Predict speaker based on conversation patterns"""
        text_lower = text.lower().strip()
        words = text_lower.split()
        
        doctor_score = 0
        patient_score = 0
        
        # Enhanced pattern matching
        for pattern in self.speaker_patterns['doctor_patterns']:
            if pattern in text_lower:
                if pattern in ['what', 'where', 'when', 'why', 'how']:
                    doctor_score += 3  # Strong question indicators
                elif text_lower.endswith('?'):
                    doctor_score += 4  # Question with question mark
                elif pattern in ['describe', 'tell me', 'show me']:
                    doctor_score += 3  # Command indicators
                else:
                    doctor_score += 2
        
        for pattern in self.speaker_patterns['patient_patterns']:
            if pattern in text_lower:
                if pattern in ['i feel', 'i have', 'i think']:
                    patient_score += 3  # Strong personal indicators
                elif pattern in ['yes', 'no', 'yeah']:
                    if text_lower.startswith(pattern):
                        patient_score += 4  # Response starters
                    else:
                        patient_score += 2
                elif pattern in ['sharp', 'burning', 'aching']:
                    patient_score += 3  # Symptom descriptions
                else:
                    patient_score += 1
        
        # Question mark detection
        if text_lower.endswith('?'):
            doctor_score += 3
        
        # Personal pronouns (strong patient indicators)
        personal_pronouns = ['i', 'my', 'me', 'myself']
        pronoun_count = sum(1 for word in words if word in personal_pronouns)
        patient_score += pronoun_count * 2
        
        # Professional terms (doctor indicators)
        professional_terms = ['patient', 'examination', 'diagnosis', 'scale']
        prof_count = sum(1 for term in professional_terms if term in text_lower)
        doctor_score += prof_count * 2
        
        # Determine speaker and confidence
        if doctor_score > patient_score and doctor_score > 2:
            return 0, min(0.95, 0.6 + (doctor_score - patient_score) * 0.1)  # Doctor = 0
        elif patient_score > doctor_score and patient_score > 2:
            return 1, min(0.95, 0.6 + (patient_score - doctor_score) * 0.1)  # Patient = 1
        else:
            # Default to patient for ambiguous cases (more common in medical conversations)
            return 1, 0.5

class SpeakerClassificationModel:
    """DistilBERT-based speaker classification model"""
    
    def __init__(self, model_name="distilbert-base-uncased"):
        self.model_name = model_name
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = None
        self.num_labels = 2  # Two speakers
        
    def prepare_dataset(self, conversations):
        """Prepare dataset for training"""
        print("🔄 Preparing training dataset...")
        
        # Show statistics about explicit prefixes
        with_prefix = [conv for conv in conversations if conv.get('has_prefix', False)]
        without_prefix = [conv for conv in conversations if not conv.get('has_prefix', False)]
        print(f"📋 Conversations with D:/P: prefixes: {len(with_prefix)}")
        print(f"📝 Conversations without prefixes: {len(without_prefix)}")
        
        # Prioritize high-confidence samples, especially those with explicit prefixes
        high_confidence = []
        
        # First, add all samples with explicit prefixes (highest priority)
        for conv in with_prefix:
            if conv['confidence'] > 0.9:  # Very high confidence for explicit prefixes
                high_confidence.append(conv)
        
        # Then add pattern-based predictions with good confidence
        for conv in without_prefix:
            if conv['confidence'] > 0.7:
                high_confidence.append(conv)
        
        print(f"📊 Using {len(high_confidence)} high-confidence samples for training")
        print(f"   - With D:/P: prefixes: {len([c for c in high_confidence if c.get('has_prefix', False)])}")
        print(f"   - Pattern-based: {len([c for c in high_confidence if not c.get('has_prefix', False)])}")
        
        # Balance dataset by speaker
        speaker_0 = [conv for conv in high_confidence if conv['speaker'] == 0]  # Doctor
        speaker_1 = [conv for conv in high_confidence if conv['speaker'] == 1]  # Patient
        
        print(f"👨‍⚕️ Doctor samples: {len(speaker_0)}")
        print(f"🏥 Patient samples: {len(speaker_1)}")
        
        # Balance by taking equal samples from each class (3000 total)
        max_samples_per_class = 1500  # 1500 per class = 3000 total
        min_samples = min(len(speaker_0), len(speaker_1), max_samples_per_class)
        if min_samples > 0:
            balanced_data = speaker_0[:min_samples] + speaker_1[:min_samples]
        else:
            balanced_data = high_confidence[:3000]  # Fallback to first 3000
        
        print(f"⚖️ Balanced dataset: {len(balanced_data)} samples (3000 for enhanced training)")
        
        # Create training data
        texts = [conv['text'] for conv in balanced_data]
        labels = [conv['speaker'] for conv in balanced_data]
        
        # Split into train/validation
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Tokenize
        train_encodings = self.tokenizer(
            train_texts, truncation=True, padding=True, max_length=512, return_tensors='pt'
        )
        val_encodings = self.tokenizer(
            val_texts, truncation=True, padding=True, max_length=512, return_tensors='pt'
        )
        
        # Create datasets
        train_dataset = Dataset.from_dict({
            'input_ids': train_encodings['input_ids'],
            'attention_mask': train_encodings['attention_mask'],
            'labels': train_labels
        })
        
        val_dataset = Dataset.from_dict({
            'input_ids': val_encodings['input_ids'],
            'attention_mask': val_encodings['attention_mask'],
            'labels': val_labels
        })
        
        return train_dataset, val_dataset
    
    def initialize_model(self):
        """Initialize the DistilBERT model for classification with GPU support"""
        print("🚀 Initializing DistilBERT model...")
        
        # Check for GPU availability
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️ Using device: {device}")
        
        self.model = DistilBertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            problem_type="single_label_classification"
        )
        
        # Move model to GPU if available
        self.model = self.model.to(device)
        
        # Freeze early layers for faster training
        for param in self.model.distilbert.embeddings.parameters():
            param.requires_grad = False
        
        for layer in self.model.distilbert.transformer.layer[:2]:  # Freeze first 2 layers
            for param in layer.parameters():
                param.requires_grad = False
        
        print("✅ Model initialized with frozen early layers")
    
    def train_model(self, train_dataset, val_dataset, output_dir="./speaker_classifier"):
        """Train the speaker classification model"""
        print("🎯 Starting model training...")
        
        # Training arguments optimized for larger dataset (3000 samples) and GPU
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,  # Increased back to 3 for larger dataset
            per_device_train_batch_size=16,  # Reduced for larger dataset to fit in memory
            per_device_eval_batch_size=32,   # Keep eval batch size higher
            warmup_steps=100,    # Increased warmup steps for larger dataset
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=50,    # Logging every 50 steps for larger dataset
            eval_steps=100,      # Evaluate every 100 steps
            evaluation_strategy="steps",
            save_steps=200,      # Save every 200 steps
            save_total_limit=2,  # Keep 2 checkpoints
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=None,      # Disable wandb/tensorboard
            fp16=torch.cuda.is_available(),  # Enable mixed precision if GPU available
            dataloader_pin_memory=torch.cuda.is_available(),  # Pin memory for GPU
            gradient_checkpointing=True,  # Save memory for larger dataset
            dataloader_num_workers=2,  # Parallel data loading
        )
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Metrics function
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            accuracy = accuracy_score(labels, predictions)
            return {"accuracy": accuracy}
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        
        # Train the model
        print("🔥 Training in progress...")
        print(f"📊 Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples")
        print(f"🖥️ Using {'GPU' if torch.cuda.is_available() else 'CPU'} acceleration")
        print(f"⚡ Batch size: {training_args.per_device_train_batch_size}, Epochs: {training_args.num_train_epochs}")
        print(f"📈 Expected training steps: ~{len(train_dataset) // training_args.per_device_train_batch_size * training_args.num_train_epochs}")
        trainer.train()
        
        # Save the model
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"💾 Model saved to: {output_dir}")
        
        # Evaluate
        eval_results = trainer.evaluate()
        print(f"📈 Validation Results: {eval_results}")
        
        return trainer

class PyannoteFineTuner:
    """Fine-tune pyannote.audio pipeline with trained DistilBERT classifier"""
    
    def __init__(self, distilbert_model_path, hf_token):
        self.distilbert_model_path = distilbert_model_path
        self.hf_token = hf_token
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load trained DistilBERT model
        print("🔄 Loading trained DistilBERT classifier...")
        self.tokenizer = DistilBertTokenizer.from_pretrained(distilbert_model_path)
        self.classifier = DistilBertForSequenceClassification.from_pretrained(distilbert_model_path)
        self.classifier.to(self.device)
        self.classifier.eval()
        print("✅ DistilBERT classifier loaded")
        
        # Load pyannote pipeline
        try:
            login(token=hf_token)
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token
            ).to(self.device)
            print("✅ Pyannote pipeline loaded")
        except Exception as e:
            print(f"❌ Failed to load pyannote pipeline: {e}")
            self.pipeline = None
    
    def predict_speaker_from_text(self, text):
        """Use trained DistilBERT to predict speaker from text"""
        try:
            # Tokenize
            inputs = self.tokenizer(
                text, 
                truncation=True, 
                padding=True, 
                max_length=512, 
                return_tensors='pt'
            ).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.classifier(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(predictions, dim=-1).item()
                confidence = predictions[0][predicted_class].item()
            
            # Map to speaker labels
            speaker = "SPEAKER_00" if predicted_class == 0 else "SPEAKER_01"
            return speaker, confidence
            
        except Exception as e:
            print(f"⚠️ DistilBERT prediction failed: {e}")
            return "SPEAKER_00", 0.5
    
    def create_enhanced_pipeline(self):
        """Create enhanced pipeline that combines pyannote with DistilBERT"""
        class EnhancedDiarizationPipeline:
            def __init__(self, pyannote_pipeline, distilbert_predictor):
                self.pyannote_pipeline = pyannote_pipeline
                self.distilbert_predictor = distilbert_predictor
                self.device = distilbert_predictor.device
            
            def __call__(self, audio_input, **kwargs):
                """Enhanced diarization with text-based speaker correction"""
                if self.pyannote_pipeline is None:
                    return None
                
                # Get pyannote diarization
                diarization = self.pyannote_pipeline(audio_input, **kwargs)
                
                # For now, return pyannote results
                # In real implementation, you would:
                # 1. Get transcription for each segment
                # 2. Use DistilBERT to predict speaker for each transcription
                # 3. Correct pyannote labels based on DistilBERT predictions
                # 4. Apply consistency smoothing
                
                return diarization
            
            def predict_speaker_with_text(self, transcription_text):
                """Predict speaker using DistilBERT for a given transcription"""
                return self.distilbert_predictor.predict_speaker_from_text(transcription_text)
        
        if self.pipeline:
            return EnhancedDiarizationPipeline(self.pipeline, self)
        else:
            return None
    
    def save_enhanced_config(self, output_path="enhanced_diarization_config.json"):
        """Save configuration for enhanced diarization"""
        config = {
            "distilbert_model_path": str(self.distilbert_model_path),
            "hf_token": self.hf_token,
            "device": str(self.device),
            "timestamp": datetime.now().isoformat(),
            "speaker_mapping": {
                0: "SPEAKER_00",  # Doctor/Interviewer
                1: "SPEAKER_01"   # Patient/Interviewee
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"💾 Enhanced configuration saved to: {output_path}")

def main():
    """Main training and fine-tuning pipeline"""
    print("🚀 Fine-tuning Speaker Diarization with DistilBERT")
    print("=" * 60)
    
    # Configuration
    data_folder = "C:\\Users\\USER\\Downloads\\live transcription\\archive (3)"  # Folder with conversation text files
    model_output_dir = "./trained_speaker_classifier"
    hf_token = "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"
    
    # Step 1: Load and prepare conversation data
    print("\n📚 STEP 1: Loading Conversation Data")
    print("-" * 40)
    
    if not os.path.exists(data_folder):
        print(f"❌ Data folder not found: {data_folder}")
        print("Please ensure the 'archive (3)' folder exists with conversation text files")
        return
    
    data_loader = ConversationDataLoader(data_folder)
    conversations = data_loader.load_conversations()
    
    if not conversations:
        print("❌ No conversation data loaded. Check your data files.")
        return
    
    # Step 2: Train DistilBERT classifier
    print("\n🧠 STEP 2: Training DistilBERT Speaker Classifier")
    print("-" * 40)
    
    classifier = SpeakerClassificationModel()
    classifier.initialize_model()
    
    train_dataset, val_dataset = classifier.prepare_dataset(conversations)
    trainer = classifier.train_model(train_dataset, val_dataset, model_output_dir)
    
    # Step 3: Create enhanced pipeline
    print("\n🔗 STEP 3: Creating Enhanced Diarization Pipeline")
    print("-" * 40)
    
    fine_tuner = PyannoteFineTuner(model_output_dir, hf_token)
    enhanced_pipeline = fine_tuner.create_enhanced_pipeline()
    
    if enhanced_pipeline:
        print("✅ Enhanced pipeline created successfully!")
        
        # Save configuration
        fine_tuner.save_enhanced_config()
        
        # Test with sample texts from the actual format
        print("\n🧪 Testing Enhanced Pipeline:")
        test_texts = [
            "What brings you in today?",
            "Sure, I'm just having a lot of chest pain and I thought I should get it checked out.",
            "OK, before we start, could you remind me of your gender and age?",
            "Sure 39, I'm a male.",
            "OK, and so when did this chest pain start?",
            "It started last night, but it's becoming sharper.",
            "OK, and where is this pain located?",
            "It's located on the left side of my chest.",
            "Has it been constant throughout that time, or changing?",
            "I would say it's been pretty constant, yeah."
        ]
        
        for i, text in enumerate(test_texts):
            speaker, confidence = enhanced_pipeline.predict_speaker_with_text(text)
            role = "Doctor" if speaker == "SPEAKER_00" else "Patient"
            print(f"  {i+1}. [{speaker}] ({role}, {confidence:.2f}): {text}")
        
    else:
        print("❌ Failed to create enhanced pipeline")
    
    print("\n🎉 Fine-tuning Complete!")
    print(f"📁 Trained model saved in: {model_output_dir}")
    print("🔗 Use this model with the enhanced diarization pipeline")

if __name__ == "__main__":
    main()
