#!/usr/bin/env python3
"""
Live microphone client with speaker diarization for Whisper streaming server
Records from microphone, applies speaker diarization, and sends audio in real-time to the server
"""
import socket
import numpy as np
import time
import threading
import queue
import sys
import tempfile
import os
from collections import defaultdict
import torch
import torchaudio
import warnings
import re
import json
import pandas as pd
import joblib
from datetime import datetime
import uuid

# Suppress specific warnings from pyannote
warnings.filterwarnings("ignore", message=".*detected number of speakers.*")
warnings.filterwarnings("ignore", message=".*Found only.*clusters.*")

try:
    import pyaudio
except ImportError:
    print("ERROR: pyaudio not installed. Install with: pip install pyaudio")
    sys.exit(1)

try:
    from googletrans import Translator, LANGUAGES
    print("✓ Google Translate library loaded")
except ImportError:
    print("⚠ Google Translate not available. Install with: pip install googletrans==4.0.0-rc1")
    Translator = None
    LANGUAGES = {}

try:
    from pyannote.audio import Pipeline
    from pyannote.core import Segment
    from huggingface_hub import login
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        DistilBertTokenizer, DistilBertForSequenceClassification,
        pipeline
    )
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    import torch.nn.functional as F
    import re
    import json
    import pandas as pd
    import joblib
    from datetime import datetime
    import uuid
    from textblob import TextBlob
    import nltk
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.sentiment import SentimentIntensityAnalyzer
    from sklearn.exceptions import NotFittedError
    import traceback
except ImportError:
    print("ERROR: Required packages not installed. Install with:")
    print("pip install pyannote.audio transformers scikit-learn pandas nltk textblob")
    sys.exit(1)

HOST = 'localhost'
PORT = 43007
SAMPLING_RATE = 16000
CHUNK_SIZE = 1600  # 0.1 seconds of audio
CHANNELS = 1
FORMAT = pyaudio.paInt16

# Initialize NLP tools for sentiment analysis
try:
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
except Exception as e:
    print(f" NLTK setup warning: {e}")
    lemmatizer = None
    stop_words = set()

class ComprehensiveSentimentAnalyzer:
    """Multi-model ensemble sentiment analysis system for healthcare applications"""
    
    def __init__(self, distilbert_model_dir=None):
        print("Initializing Sentiment Analysis System...")
        print("   Multi-model ensemble for healthcare applications")
        
        # Model storage
        self.vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2), stop_words='english', sublinear_tf=True)
        self.label_encoder = LabelEncoder()
        self.models = {"lr": None, "svm": None}
        self.model_accuracy = {}
        
        # Rule-based analyzers
        try:
            self.vader = SentimentIntensityAnalyzer()
            print("   VADER sentiment analyzer loaded")
        except Exception as e:
            print(f"   VADER failed to load: {e}")
            self.vader = None
        
        # Fine-tuned DistilBERT model
        distilbert_model_dir = distilbert_model_dir or './distilbert-finetuned-patient'
        self.distilbert_sentiment = None
        
        if os.path.exists(distilbert_model_dir):
            try:
                self.distilbert_sentiment = pipeline("text-classification", model=distilbert_model_dir)
                print(f"   Fine-tuned DistilBERT loaded from: {distilbert_model_dir}")
            except Exception as e:
                try:
                    self.distilbert_sentiment = pipeline("sentiment-analysis", model=distilbert_model_dir)
                    print(f"   Fine-tuned DistilBERT loaded (fallback) from: {distilbert_model_dir}")
                except Exception as e2:
                    print(f"   DistilBERT failed to load: {e2}")
                    self.distilbert_sentiment = None
        else:
            print(f"   DistilBERT model not found at: {distilbert_model_dir}")
        
        # Healthcare-specific keyword loading
        self.before_kw, self.during_kw, self.after_kw = self.load_time_keywords()
        
        # Initialize or load traditional ML models
        self.initialize_models()
        
        print("Sentiment Analysis System Ready!")
        print("   Multi-model ensemble: DistilBERT + VADER")
        print("   Patient-specific analysis configured")
    
    def initialize_models(self):
        """Initialize or load pre-trained ML models"""
        try:
            self.load_models()
            print("   Traditional ML models loaded")
        except Exception as e:
            print(f"   ML models not found, using basic configuration: {e}")
            # Initialize basic models for demonstration
            self.models['lr'] = LogisticRegression(max_iter=1000)
            self.models['svm'] = SVC(kernel='linear', probability=True)
            self.label_encoder.fit([-1, 0, 1])  # Negative, Neutral, Positive
    
    def load_models(self):
        """Load pre-trained models if they exist"""
        model_files = ['lr_model.pkl', 'svm_model.pkl', 'tfidf_vectorizer.pkl', 'label_encoder.pkl']
        models_dir = 'models'
        
        if all(os.path.exists(os.path.join(models_dir, f)) for f in model_files):
            self.models['lr'] = joblib.load(os.path.join(models_dir, 'lr_model.pkl'))
            self.models['svm'] = joblib.load(os.path.join(models_dir, 'svm_model.pkl'))
            self.vectorizer = joblib.load(os.path.join(models_dir, 'tfidf_vectorizer.pkl'))
            self.label_encoder = joblib.load(os.path.join(models_dir, 'label_encoder.pkl'))
        else:
            raise FileNotFoundError("Model files not found")
    
    def load_time_keywords(self):
        """Load temporal keywords for healthcare phase classification"""
        before_kw = ['before', 'prior', 'previous', 'initially', 'started', 'began']
        during_kw = ['during', 'while', 'treatment', 'therapy', 'medication', 'currently']
        after_kw = ['after', 'following', 'post', 'recovery', 'improved', 'better', 'worse']
        
        # Try to load from CSV if available
        csv_path = "sentiment_phases_1000.csv"
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                if 'before' in df.columns:
                    before_kw.extend([str(x).strip().lower() for x in df['before'].dropna() if str(x).strip()])
                if 'during' in df.columns:
                    during_kw.extend([str(x).strip().lower() for x in df['during'].dropna() if str(x).strip()])
                if 'after' in df.columns:
                    after_kw.extend([str(x).strip().lower() for x in df['after'].dropna() if str(x).strip()])
            except Exception as e:
                print(f"   Could not load temporal keywords: {e}")
        
        return before_kw, during_kw, after_kw
    
    def preprocess_text(self, text):
        """Advanced text preprocessing for sentiment analysis"""
        if not text or not isinstance(text, str):
            return ''
        
        text = str(text).strip()
        if len(text) < 3:
            return ''
        
        try:
            # Basic cleaning
            text = text.lower()
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            
            if lemmatizer and stop_words:
                # Tokenization and lemmatization
                tokens = word_tokenize(text)
                tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
                tokens = [lemmatizer.lemmatize(t) for t in tokens]
                return ' '.join(tokens)
            else:
                return text
        except Exception as e:
            return text.lower()
    
    def classify_time_period(self, text):
        """Classify treatment phase: disabled for now"""
        return None  # Treatment phase analysis disabled
    
    def analyze_patient_sentiment(self, text, patient_id=None):
        """Simplified sentiment analysis using DistilBERT + VADER for negation"""
        if not text or len(text.strip()) < 3:
            return {
                'text': text,
                'error': 'Text too short for analysis',
                'patient_id': patient_id or str(uuid.uuid4())
            }
        
        try:
            patient_id = patient_id or str(uuid.uuid4())
            timestamp = datetime.now().isoformat()
            
            results = {
                'patient_id': patient_id,
                'text': text,
                'timestamp': timestamp,
                'models': {}
            }
            
            # 1. Fine-tuned DistilBERT Analysis (Primary)
            if self.distilbert_sentiment:
                try:
                    distilbert_result = self.distilbert_sentiment(text)[0]
                    results['models']['distilbert'] = {
                        'sentiment': distilbert_result['label'],
                        'confidence': float(distilbert_result['score']),
                        'method': 'transformer'
                    }
                except Exception as e:
                    results['models']['distilbert'] = {
                        'sentiment': 'Unknown',
                        'confidence': 0.0,
                        'error': str(e)
                    }
            
            # 2. VADER for negation handling
            if self.vader:
                try:
                    vader_scores = self.vader.polarity_scores(text)
                    compound = vader_scores['compound']
                    
                    if compound >= 0.05:
                        vader_sentiment = "Positive"
                    elif compound <= -0.05:
                        vader_sentiment = "Negative"
                    else:
                        vader_sentiment = "Neutral"
                    
                    results['models']['vader'] = {
                        'sentiment': vader_sentiment,
                        'confidence': abs(compound),
                        'compound_score': compound,
                        'method': 'lexicon'
                    }
                except Exception as e:
                    results['models']['vader'] = {'error': str(e)}
            
            # 3. Simple Ensemble (DistilBERT primary, VADER for negation correction)
            if 'distilbert' in results['models'] and 'error' not in results['models']['distilbert']:
                primary_sentiment = results['models']['distilbert']['sentiment']
                primary_confidence = results['models']['distilbert']['confidence']
                
                # Use VADER to correct for strong negation
                if 'vader' in results['models'] and 'error' not in results['models']['vader']:
                    vader_compound = results['models']['vader']['compound_score']
                    
                    # Strong negative VADER score overrides positive DistilBERT
                    if vader_compound < -0.6 and primary_sentiment in ['Positive', 'LABEL_2']:
                        final_sentiment = "Negative"
                        final_confidence = min(0.9, primary_confidence * 0.8)
                    # Strong positive VADER score overrides negative DistilBERT
                    elif vader_compound > 0.6 and primary_sentiment in ['Negative', 'LABEL_0']:
                        final_sentiment = "Positive" 
                        final_confidence = min(0.9, primary_confidence * 0.8)
                    else:
                        # Use DistilBERT as primary
                        final_sentiment = primary_sentiment
                        final_confidence = primary_confidence
                else:
                    final_sentiment = primary_sentiment
                    final_confidence = primary_confidence
                
                results['final'] = {
                    'sentiment': final_sentiment,
                    'confidence': final_confidence,
                    'method': 'distilbert_vader_ensemble'
                }
            else:
                results['final'] = {
                    'sentiment': 'Neutral',
                    'confidence': 0.3,
                    'method': 'fallback'
                }
            
            return results
            
        except Exception as e:
            return {
                'text': text,
                'patient_id': patient_id or str(uuid.uuid4()),
                'error': f'Analysis failed: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }
    
    def label_to_string(self, label):
        """Convert numeric label to string"""
        label_map = {1: 'Positive', 0: 'Neutral', -1: 'Negative', '1': 'Positive', '0': 'Neutral', '-1': 'Negative'}
        return label_map.get(label, str(label))
CHANNELS = 1
FORMAT = pyaudio.paInt16

class TreatmentPhaseClassifier:
    """Fine-tuned DistilBERT classifier for treatment phase detection"""
    
    def __init__(self, model_path="./distilbert-finetuned-phase"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        self.phase_labels = {0: "before", 1: "during", 2: "after"}
        
        print("Loading Treatment Phase Classifier...")
        self.load_model()
    
    def load_model(self):
        """Load the fine-tuned DistilBERT model for phase classification"""
        try:
            if os.path.exists(self.model_path):
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
                self.model.eval()
                self.is_loaded = True
                print(f"   Treatment phase model loaded from: {self.model_path}")
            else:
                print(f"   Phase model not found at: {self.model_path}")
                self.is_loaded = False
        except Exception as e:
            print(f"   Failed to load phase model: {e}")
            self.is_loaded = False
    
    def predict_phase(self, text, confidence_threshold=0.6):
        """Predict treatment phase from text"""
        if not self.is_loaded or not text or len(text.strip()) < 3:
            return "unknown", 0.0
        
        try:
            # Tokenize and encode the text
            inputs = self.tokenizer(
                text.strip(),
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            
            # Get model prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Get the predicted class and confidence
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence = predictions[0][predicted_class].item()
            
            # Map to phase label
            phase = self.phase_labels.get(predicted_class, "unknown")
            
            # Apply confidence threshold
            if confidence < confidence_threshold:
                phase = "unclear"
                
            return phase, confidence
            
        except Exception as e:
            print(f"   Phase prediction error: {e}")
            return "error", 0.0
    
    def predict_batch(self, texts, confidence_threshold=0.6):
        """Predict phases for multiple texts in batch"""
        if not self.is_loaded or not texts:
            return [("unknown", 0.0) for _ in texts]
        
        try:
            # Tokenize all texts at once
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            results = []
            for i, pred in enumerate(predictions):
                predicted_class = torch.argmax(pred).item()
                confidence = pred[predicted_class].item()
                
                phase = self.phase_labels.get(predicted_class, "unknown")
                if confidence < confidence_threshold:
                    phase = "unclear"
                    
                results.append((phase, confidence))
            
            return results
            
        except Exception as e:
            print(f"   Batch phase prediction error: {e}")
            return [("error", 0.0) for _ in texts]

class FineTunedSpeakerClassifier:
    """Fine-tuned DistilBERT speaker classifier for parallel processing with pyannote.audio"""
    
    def __init__(self, model_path="./trained_speaker_classifier"):
        print(" Loading fine-tuned speaker classifier...")
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        
        # Load configuration
        self.config_path = "enhanced_diarization_config.json"
        self.speaker_mapping = {"0": "Doctor", "1": "Patient"}
        
        try:
            # Load configuration file
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    self.speaker_mapping = config.get("speaker_mapping", self.speaker_mapping)
                    print(f" Loaded speaker mapping: {self.speaker_mapping}")
            
            # Load tokenizer and model
            print(f"    Loading tokenizer from: {self.model_path}")
            self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_path)
            
            print(f"    Loading model from: {self.model_path}")
            self.model = DistilBertForSequenceClassification.from_pretrained(
                self.model_path,
                num_labels=2,  # Doctor (0) and Patient (1)
                torch_dtype=torch.float32
            ).to(self.device)
            
            self.model.eval()  # Set to evaluation mode
            self.is_loaded = True
            
            print(f" Fine-tuned classifier loaded successfully on {self.device}")
            print(f"   Model: DistilBERT with 2 speaker labels")
            print(f"   Mapping: {self.speaker_mapping}")
            
        except Exception as e:
            print(f" Failed to load fine-tuned classifier: {e}")
            print(" Falling back to pattern-based classification")
            self.is_loaded = False
    
    def predict_speaker(self, text, confidence_threshold=0.7):
        """Predict speaker using fine-tuned DistilBERT model with enhanced medical context"""
        if not self.is_loaded or not text or len(text.strip()) == 0:
            return "Doctor", 0.5  # Default fallback
        
        try:
            # Preprocess text
            text = text.strip()
            if len(text) < 3:  # Too short for meaningful prediction
                return "Doctor", 0.3
            
            # Enhanced pre-processing for medical conversations
            text_lower = text.lower()
            
            # Strong rule-based overrides for clear patterns
            # Doctor patterns (questions and medical professional language)
            strong_doctor_patterns = [
                r'what.*pain', r'where.*hurt', r'when.*start', r'how.*feel', r'rate.*pain',
                r'describe.*pain', r'tell me about', r'on a scale', r'any.*problem',
                r'have you.*', r'do you.*', r'can you.*', r'are you.*', r'will you.*',
                r'how long.*', r'any symptoms', r'medical history', r'examination'
            ]
            
            # Patient patterns (responses and personal experiences)  
            strong_patient_patterns = [
                r'^yes', r'^no', r'^yeah', r'^okay', r'^sure', r'^well',
                r'i feel.*', r'i have.*', r'i think.*', r'i am.*', r'my.*pain',
                r'it hurts.*', r'the pain.*', r'it started.*', r'last night',
                r'this morning', r'about.*hour', r'since.*', r'i noticed'
            ]
            
            import re
            
            # Check for strong doctor patterns
            doctor_score = sum(1 for pattern in strong_doctor_patterns if re.search(pattern, text_lower))
            patient_score = sum(1 for pattern in strong_patient_patterns if re.search(pattern, text_lower))
            
            # Rule-based override for very clear cases
            if doctor_score >= 2 and patient_score == 0:
                return "Doctor", 0.95
            elif patient_score >= 2 and doctor_score == 0:
                return "Patient", 0.95
            elif text_lower.endswith('?') and any(w in text_lower for w in ['what', 'where', 'when', 'how', 'why']):
                return "Doctor", 0.93  # Clear questions are usually from doctor
            
            # Use DistilBERT for ambiguous cases
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            ).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # Apply softmax to get probabilities
                probabilities = F.softmax(logits, dim=-1)
                confidence = torch.max(probabilities).item()
                predicted_label = torch.argmax(logits, dim=-1).item()
            
            # Map to speaker ID
            speaker_id = self.speaker_mapping.get(str(predicted_label), "Doctor")
            
            # Boost confidence for clear medical conversation patterns
            if text_lower.endswith('?'):
                if speaker_id == "Doctor":
                    confidence = min(0.98, confidence * 1.3)
            elif any(word in text_lower for word in ['i feel', 'i have', 'yes,', 'no,', 'it hurts', 'pain']):
                if speaker_id == "Patient":
                    confidence = min(0.98, confidence * 1.3)
            
            # Additional context-based confidence adjustments
            if doctor_score > patient_score and speaker_id == "Doctor":
                confidence = min(0.98, confidence * 1.2)
            elif patient_score > doctor_score and speaker_id == "Patient":
                confidence = min(0.98, confidence * 1.2)
            
            # Return prediction with confidence
            return speaker_id, confidence
            
        except Exception as e:
            print(f" Fine-tuned prediction failed: {e}")
            return "Doctor", 0.3  # Fallback
    
    def predict_batch(self, texts, confidence_threshold=0.7):
        """Predict speakers for multiple texts in batch for efficiency"""
        if not self.is_loaded or not texts:
            return [("Doctor", 0.5) for _ in texts]
        
        try:
            # Preprocess texts
            processed_texts = [text.strip() for text in texts if text and len(text.strip()) >= 3]
            if not processed_texts:
                return [("Doctor", 0.3) for _ in texts]
            
            # Tokenize batch
            inputs = self.tokenizer(
                processed_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # Apply softmax to get probabilities
                probabilities = F.softmax(logits, dim=-1)
                confidences = torch.max(probabilities, dim=-1)[0]
                predicted_labels = torch.argmax(logits, dim=-1)
            
            # Map to speaker IDs
            results = []
            for i, (label, confidence) in enumerate(zip(predicted_labels, confidences)):
                speaker_id = self.speaker_mapping.get(str(label.item()), "Doctor")
                results.append((speaker_id, confidence.item()))
            
            return results
            
        except Exception as e:
            print(f" Fine-tuned batch prediction failed: {e}")
            return [("Doctor", 0.3) for _ in texts]

class FastSpeakerClassifier:
    """Lightweight, real-time speaker classification using voice activity and content patterns"""
    
    def __init__(self):
        self.speaker_voice_profiles = {}  # Track voice characteristics per speaker
        self.conversation_patterns = {
            'question_indicators': ['what', 'where', 'when', 'why', 'how', 'who', 'which', 'whose', '?'],
            'response_indicators': ['yes', 'no', 'well', 'so', 'but', 'however', 'because', 'actually'],
            'professional_terms': ['patient', 'symptoms', 'pain', 'medication', 'treatment', 'examination'],
            'casual_responses': ['yeah', 'okay', 'sure', 'right', 'definitely', 'maybe']
        }
        self.speaker_patterns = defaultdict(list)  # Track speaking patterns per speaker
        self.last_speaker_features = {}
        
    def extract_audio_features(self, audio_data):
        """Extract comprehensive audio features for accurate speaker identification"""
        if len(audio_data) == 0:
            return {'energy': 0.0, 'max_amplitude': 0.0, 'zero_crossing_rate': 0.0, 'spectral_centroid': 0.0,
                   'pitch_mean': 0.0, 'pitch_std': 0.0, 'mfcc': np.zeros(13), 'is_voice_active': False}
        
        # Voice Activity Detection (VAD) using energy and zero crossing rate
        abs_audio = np.abs(audio_data)
        energy = np.mean(abs_audio)
        zcr = len(np.where(np.diff(np.signbit(audio_data)))[0]) / len(audio_data)
        
        # Enhanced VAD thresholds
        energy_threshold = 0.01  # Minimum energy for voice
        zcr_threshold = 0.1      # Maximum ZCR for voice (lower = more tonal)
        is_voice_active = energy > energy_threshold and zcr < zcr_threshold
        
        # Basic features
        features = {
            'energy': energy,
            'max_amplitude': np.max(abs_audio),
            'zero_crossing_rate': zcr,
            'is_voice_active': is_voice_active
        }
        
        if is_voice_active and len(audio_data) > 512:  # Only extract complex features for voice
            try:
                # Spectral features
                fft = np.fft.rfft(audio_data)
                magnitude = np.abs(fft)
                freqs = np.fft.rfftfreq(len(audio_data), 1/16000)
                
                # Spectral centroid (brightness indicator)
                spectral_centroid = np.sum(freqs * magnitude) / np.sum(magnitude) if np.sum(magnitude) > 0 else 0
                features['spectral_centroid'] = spectral_centroid
                
                # Pitch estimation using autocorrelation
                autocorr = np.correlate(audio_data, audio_data, mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                
                # Find pitch peaks (fundamental frequency)
                min_pitch_samples = 16000 // 300  # 300 Hz max
                max_pitch_samples = 16000 // 80   # 80 Hz min
                
                if len(autocorr) > max_pitch_samples:
                    pitch_autocorr = autocorr[min_pitch_samples:max_pitch_samples]
                    if len(pitch_autocorr) > 0:
                        pitch_peak = np.argmax(pitch_autocorr) + min_pitch_samples
                        pitch_freq = 16000 / pitch_peak if pitch_peak > 0 else 0
                        features['pitch_mean'] = pitch_freq
                        features['pitch_std'] = np.std(pitch_autocorr)
                    else:
                        features['pitch_mean'] = 0.0
                        features['pitch_std'] = 0.0
                else:
                    features['pitch_mean'] = 0.0
                    features['pitch_std'] = 0.0
                
                # Simplified MFCC-like features (mel-frequency cepstral coefficients)
                # Use log energy in different frequency bands as MFCC approximation
                n_bands = 13
                mel_features = []
                for i in range(n_bands):
                    start_freq = int(i * len(magnitude) / n_bands)
                    end_freq = int((i + 1) * len(magnitude) / n_bands)
                    band_energy = np.sum(magnitude[start_freq:end_freq])
                    mel_features.append(np.log(band_energy + 1e-10))  # Log energy
                
                features['mfcc'] = np.array(mel_features)
                
            except Exception as e:
                # Fallback to basic features if complex extraction fails
                features.update({
                    'spectral_centroid': np.mean(abs_audio),
                    'pitch_mean': 0.0,
                    'pitch_std': 0.0,
                    'mfcc': np.zeros(13)
                })
        else:
            # No voice detected or insufficient data
            features.update({
                'spectral_centroid': 0.0,
                'pitch_mean': 0.0,
                'pitch_std': 0.0,
                'mfcc': np.zeros(13)
            })
        
        return features
    
    def analyze_text_patterns(self, text):
        """Enhanced text pattern analysis with medical conversation context"""
        if not text:
            return {'role_score': 0.5, 'confidence': 0.1, 'pattern_type': 'unknown'}
            
        text_lower = text.lower().strip()
        words = text_lower.split()
        
        # Enhanced medical question patterns
        doctor_question_patterns = [
            'what', 'where', 'when', 'why', 'how', 'who', 'which', 'whose',
            'can you', 'do you', 'have you', 'are you', 'will you', 'did you', 
            'could you', 'would you', 'should you', 'any problems', 'how about',
            'tell me', 'describe', 'explain', 'show me', 'point to', 'remind me',
            'on a scale', 'rate your', 'how long', 'what kind of', 'any symptoms',
            'feeling sick', 'any pain', 'any history', 'medical history'
        ]
        
        # Enhanced patient response patterns
        patient_response_patterns = [
            'yes', 'no', 'yeah', 'nope', 'sure', 'okay', 'right', 'exactly',
            'correct', 'absolutely', 'definitely', 'maybe', 'perhaps', 'well', 'um', 'uh',
            'i feel', 'i have', 'i think', 'i believe', 'my', 'it', 'the pain',
            'about', 'around', 'probably', 'it is', 'it was', 'i am', 'i was',
            'i guess', "i don't", "i can't", 'not really', 'hurts', 'painful',
            'aches', 'feels like', 'sometimes', 'usually', 'mostly', 'it started',
            'last night', 'this morning', 'yesterday', 'since', 'been having'
        ]
        
        # Location and description patterns (typically patient responses)
        location_patterns = ['on the', 'in my', 'on my', 'near my', 'around my', 
                           'left side', 'right side', 'chest', 'back', 'stomach', 'neck', 'head']
        
        # Professional medical terms (typically doctor)
        professional_patterns = ['patient', 'symptoms', 'examination', 'medication', 
                               'treatment', 'diagnosis', 'scale', 'rate', 'medical']
        
        # Personal experience patterns (typically patient)
        personal_patterns = ['i', 'my', 'me', 'pain', 'hurt', 'feel', 'breathing', 
                           'dizzy', 'tired', 'sore']
        
        # Calculate pattern scores
        doctor_score = 0
        patient_score = 0
        professional_score = 0
        personal_score = 0
        
        # Check for question patterns (strong doctor indicators)
        if text_lower.endswith('?'):
            doctor_score += 4  # Strong question indicator
        
        if any(text_lower.startswith(pattern) for pattern in doctor_question_patterns):
            doctor_score += 4  # Question at start
        
        doctor_score += sum(2 for pattern in doctor_question_patterns if pattern in text_lower)
        
        # Check for response patterns (strong patient indicators)
        if any(text_lower.startswith(pattern) for pattern in patient_response_patterns):
            patient_score += 3  # Response at start
        
        patient_score += sum(2 for pattern in patient_response_patterns if pattern in text_lower)
        patient_score += sum(3 for pattern in location_patterns if pattern in text_lower)
        
        # Professional vs personal language
        professional_score += sum(1 for term in professional_patterns if term in text_lower)
        personal_score += sum(1 for expr in personal_patterns if expr in text_lower)
        
        # Combine scores with weights
        total_doctor_score = doctor_score + professional_score * 0.5
        total_patient_score = patient_score + personal_score * 0.5
        
        # Determine speaker with enhanced logic
        if total_doctor_score > total_patient_score + 2:  # Clear doctor
            role_score = 0.1
            confidence = min(0.98, 0.7 + (total_doctor_score - total_patient_score) * 0.05)
            pattern_type = 'doctor_question'
        elif total_patient_score > total_doctor_score + 2:  # Clear patient
            role_score = 0.9
            confidence = min(0.98, 0.7 + (total_patient_score - total_doctor_score) * 0.05)
            pattern_type = 'patient_response'
        elif total_doctor_score > total_patient_score:  # Likely doctor
            role_score = 0.3
            confidence = 0.6
            pattern_type = 'likely_doctor'
        elif total_patient_score > total_doctor_score:  # Likely patient
            role_score = 0.7
            confidence = 0.6
            pattern_type = 'likely_patient'
        else:  # Ambiguous
            role_score = 0.5
            confidence = 0.4
            pattern_type = 'ambiguous'
        
        # Length-based confidence adjustment
        word_count = len(words)
        if word_count > 8 and pattern_type in ['patient_response', 'likely_patient']:
            confidence = min(0.98, confidence + 0.1)
        elif word_count <= 4 and pattern_type in ['doctor_question', 'likely_doctor']:
            confidence = min(0.98, confidence + 0.1)
        
        return {
            'role_score': role_score,
            'confidence': confidence,
            'pattern_type': pattern_type,
            'doctor_score': total_doctor_score,
            'patient_score': total_patient_score
        }
    
    def predict_speaker(self, audio_features, text="", previous_speaker="Doctor"):
        """Advanced speaker prediction using audio features and conversation context"""
        
        # Skip if no voice activity detected
        if not audio_features.get('is_voice_active', False):
            return previous_speaker, 0.1  # Very low confidence for silence
        
        # Extract key features for comparison
        current_features = {
            'pitch': audio_features.get('pitch_mean', 0.0),
            'energy': audio_features.get('energy', 0.0),
            'spectral_centroid': audio_features.get('spectral_centroid', 0.0),
            'mfcc': audio_features.get('mfcc', np.zeros(13))
        }
        
        # If we have stored speaker profiles, compare against them
        confidence = 0.5  # Default confidence
        predicted_speaker = previous_speaker  # Default to previous
        
        # Compare with stored speaker features
        if hasattr(self, 'speaker_profiles') and len(self.speaker_profiles) > 0:
            best_similarity = -1
            best_speaker = previous_speaker
            
            for speaker_id, stored_features in self.speaker_profiles.items():
                similarity = self.calculate_speaker_similarity(current_features, stored_features)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_speaker = speaker_id
            
            # Use similarity-based prediction if confidence is high enough
            if best_similarity > 0.7:  # High similarity threshold
                predicted_speaker = best_speaker
                confidence = min(0.95, best_similarity)
            else:
                # Low similarity - might be the other speaker
                predicted_speaker = "Patient" if best_speaker == "Doctor" else "Doctor"
                confidence = 0.6
        else:
            # No stored profiles - use basic alternation with feature-based hints
            pitch = current_features['pitch']
            energy = current_features['energy']
            
            # Simple heuristics based on voice characteristics
            if pitch > 0:  # Valid pitch detected
                # Higher pitch often indicates one speaker type
                if pitch > 180:  # Higher pitch range
                    predicted_speaker = "Patient"
                    confidence = 0.7
                elif pitch < 140:  # Lower pitch range
                    predicted_speaker = "Doctor"
                    confidence = 0.7
                else:
                    # Medium pitch - use alternation
                    predicted_speaker = "Patient" if previous_speaker == "Doctor" else "Doctor"
                    confidence = 0.5
            else:
                # No pitch - use energy-based alternation
                predicted_speaker = "Patient" if previous_speaker == "Doctor" else "Doctor"
                confidence = 0.4
        
        # Update or initialize speaker profiles
        if not hasattr(self, 'speaker_profiles'):
            self.speaker_profiles = {}
        
        # Store/update speaker profile (exponential moving average)
        if predicted_speaker in self.speaker_profiles:
            # Update existing profile with exponential moving average
            alpha = 0.3  # Learning rate
            stored = self.speaker_profiles[predicted_speaker]
            stored['pitch'] = alpha * current_features['pitch'] + (1-alpha) * stored['pitch']
            stored['energy'] = alpha * current_features['energy'] + (1-alpha) * stored['energy']
            stored['spectral_centroid'] = alpha * current_features['spectral_centroid'] + (1-alpha) * stored['spectral_centroid']
            stored['mfcc'] = alpha * current_features['mfcc'] + (1-alpha) * stored['mfcc']
            stored['count'] += 1
        else:
            # Create new profile
            self.speaker_profiles[predicted_speaker] = {
                'pitch': current_features['pitch'],
                'energy': current_features['energy'],
                'spectral_centroid': current_features['spectral_centroid'],
                'mfcc': current_features['mfcc'].copy(),
                'count': 1
            }
        
        return predicted_speaker, confidence
    
    def calculate_speaker_similarity(self, features1, features2):
        """Calculate similarity between two speaker feature sets"""
        try:
            similarity_score = 0.0
            feature_count = 0
            
            # Pitch similarity (normalized difference)
            if features1['pitch'] > 0 and features2['pitch'] > 0:
                pitch_diff = abs(features1['pitch'] - features2['pitch'])
                pitch_similarity = max(0, 1.0 - (pitch_diff / 100.0))  # Normalize by 100 Hz
                similarity_score += pitch_similarity * 0.4  # 40% weight
                feature_count += 1
            
            # Energy similarity
            if features1['energy'] > 0 and features2['energy'] > 0:
                energy_ratio = min(features1['energy'], features2['energy']) / max(features1['energy'], features2['energy'])
                similarity_score += energy_ratio * 0.2  # 20% weight
                feature_count += 1
            
            # Spectral centroid similarity
            if features1['spectral_centroid'] > 0 and features2['spectral_centroid'] > 0:
                sc_ratio = min(features1['spectral_centroid'], features2['spectral_centroid']) / max(features1['spectral_centroid'], features2['spectral_centroid'])
                similarity_score += sc_ratio * 0.2  # 20% weight
                feature_count += 1
            
            # MFCC similarity using cosine similarity
            if len(features1['mfcc']) > 0 and len(features2['mfcc']) > 0:
                mfcc1_norm = np.linalg.norm(features1['mfcc'])
                mfcc2_norm = np.linalg.norm(features2['mfcc'])
                if mfcc1_norm > 0 and mfcc2_norm > 0:
                    cosine_sim = np.dot(features1['mfcc'], features2['mfcc']) / (mfcc1_norm * mfcc2_norm)
                    # Convert cosine similarity (-1 to 1) to positive similarity (0 to 1)
                    mfcc_similarity = (cosine_sim + 1) / 2
                    similarity_score += mfcc_similarity * 0.2  # 20% weight
                    feature_count += 1
            
            # Return normalized similarity
            return similarity_score / feature_count if feature_count > 0 else 0.0
            
        except Exception as e:
            return 0.0  # Return low similarity on error

class RealTimeDiarizationProcessor:
    """Advanced real-time diarization with state-of-the-art techniques from pyannote.audio 3.1"""
    
    def __init__(self):
        print(" Initializing three-stage diarization pipeline...")
        print("    Stage 1: Voice Activity Detection & Segmentation")
        print("    Stage 2: Neural Speaker Embeddings")
        print("    Stage 3: Agglomerative Clustering")
        print("    Stage 4: Treatment Phase Detection")
        
        self.fast_classifier = FastSpeakerClassifier()
        self.finetuned_classifier = FineTunedSpeakerClassifier()  # Add fine-tuned classifier
        self.phase_classifier = TreatmentPhaseClassifier()  # Add phase classifier
        self.use_acoustic_diarization = True
        self.acoustic_processor = None
        self.segmentation_model = None
        self.embedding_model = None
        
        # Advanced configuration for three-stage pipeline
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️ Using device: {self.device}")
        
        # Initialize three-stage diarization pipeline
        try:
            from pyannote.audio import Pipeline, Model
            from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
            from pyannote.core import Segment, Annotation
            from sklearn.cluster import AgglomerativeClustering
            from huggingface_hub import login
            import numpy as np
            
            # Authentication
            HF_TOKEN = "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"
            try:
                login(token=HF_TOKEN)
                print(" Hugging Face authentication successful!")
            except Exception as login_error:
                print(f" Hugging Face login failed: {login_error}")
                self.use_acoustic_diarization = False
                return
            
            print(" Loading three-stage pipeline components...")
            
            # Stage 1: Voice Activity Detection and Segmentation
            print("    Loading segmentation model...")
            self.segmentation_model = Model.from_pretrained(
                "pyannote/segmentation-3.0",
                use_auth_token=HF_TOKEN
            ).to(self.device)
            
            # Stage 2: Neural Speaker Embedding Model
            print("    Loading speaker embedding model...")
            self.embedding_model = PretrainedSpeakerEmbedding(
                "pyannote/wespeaker-voxceleb-resnet34-LM",
                use_auth_token=HF_TOKEN,
                device=self.device
            )
            
            # Stage 3: Complete pipeline with clustering
            print("    Loading complete diarization pipeline...")
            self.acoustic_processor = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=HF_TOKEN
            ).to(self.device)
            
            print(" Three-stage pipeline loaded successfully!")
            print("    Stage 1: Segmentation model ready")
            print("    Stage 2: Embedding model ready") 
            print("    Stage 3: Clustering pipeline ready")
            
            # Show fine-tuned classifier status
            if self.finetuned_classifier.is_loaded:
                print("    Stage 4: Fine-tuned DistilBERT classifier ready")
                print("    Using parallel ensemble: pyannote.audio + fine-tuned model")
            else:
                print("    Stage 4: Fine-tuned classifier unavailable - using fallback")
            
            # Three-stage processing parameters
            self.segmentation_threshold = 0.5  # Voice activity threshold
            self.embedding_window_size = 1.5   # Embedding window (seconds)
            self.embedding_step_size = 0.75    # Sliding window step
            self.clustering_threshold = 0.7    # Agglomerative clustering threshold
            self.min_cluster_size = 0.5        # Minimum cluster duration
            
            # Overlapped speech detection
            self.overlap_threshold = 0.3
            self.overlap_regions = []
            
            # RTTM output support
            self.rttm_segments = []
            self.current_rttm_file = None
            
        except Exception as e:
            print(f" Three-stage pipeline unavailable: {e}")
            print(" Falling back to basic classification")
            self.use_acoustic_diarization = False
            self.acoustic_processor = None
        
        # Quick sentence-level segmentation parameters (optimized for performance)
        self.min_segment_duration = 1.0  # Slightly longer to reduce processing load
        self.max_segment_duration = 3.0  # Longer segments for better classification  
        self.speaker_switch_threshold = 0.7  # Higher threshold to reduce switching
        self.overlap_threshold = 0.4  # Less sensitive to reduce false positives
        
        # Advanced audio preprocessing
        self.target_sample_rate = 16000  # pyannote.audio optimal sample rate
        self.chunk_duration = 2.0  # Optimal chunk size for processing
        
        # Speaker tracking improvements with actual embedding usage
        self.speaker_embeddings = {}  # Store actual speaker embeddings from pyannote
        self.speaker_confidence_history = defaultdict(list)  # Track confidence over time
        self.speaker_feature_profiles = {}  # Store acoustic feature profiles per speaker
        self.embedding_similarity_threshold = 0.85  # Threshold for embedding similarity
        
        # Quick speaker consistency tracking
        self.speaker_transition_history = []  # Track speaker transitions for pattern learning
        self.min_speaker_duration = 0.5  # Much shorter minimum duration for quick switching
        self.speaker_embedding_cache = {}  # Cache embeddings for real-time comparison
        
    def process_audio_realtime(self, audio_data, start_time, transcription_text=""):
        """Enhanced real-time processing with sophisticated ensemble approach"""
        duration = len(audio_data) / SAMPLING_RATE
        
        # Skip very short segments to reduce processing overhead
        if duration < self.min_segment_duration:
            return None
        
        # Apply Voice Activity Detection first
        if not self._detect_voice_activity(audio_data):
            print(" Silence detected - skipping diarization", end='')
            return None
            
        # Extract audio features for speaker identification
        audio_features = self.fast_classifier.extract_audio_features(audio_data)
        
        # Skip processing if no voice activity detected in features
        if not audio_features.get('is_voice_active', False):
            return None
        
        speaker_segments = []
        
        # Strategy 1: Advanced Multi-Model Ensemble (Primary method when text is available)
        if transcription_text and len(transcription_text.strip()) >= 3:
            print("", end='', flush=True)  # Ensemble indicator
            
            # Collect predictions from all available models
            ensemble_predictions = []
            
            # 1. Fine-tuned DistilBERT prediction (highest weight for conversation analysis)
            try:
                finetuned_speaker, finetuned_confidence = self.finetuned_classifier.predict_speaker(
                    transcription_text
                )
                ensemble_predictions.append((finetuned_speaker, finetuned_confidence, 0.80))  # 80% weight
                print(f"D:{finetuned_confidence:.2f}", end='', flush=True)
            except Exception as e:
                print(f"FT:{e}", end='', flush=True)
            
            # 2. Fast pattern-based classifier with audio features (reduced weight)
            try:
                fast_speaker, fast_confidence = self.fast_classifier.predict_speaker(
                    audio_features, transcription_text, "Doctor"
                )
                ensemble_predictions.append((fast_speaker, fast_confidence, 0.15))  # 15% weight
                print(f"F:{fast_confidence:.2f}", end='', flush=True)
            except Exception as e:
                print(f"FC:{e}", end='', flush=True)
            
            # 3. Acoustic analysis from pyannote.audio (lower weight but important for voice characteristics)
            acoustic_speaker = None
            acoustic_confidence = 0.3
            
            if (self.use_acoustic_diarization and self.acoustic_processor and 
                self.min_segment_duration <= duration <= self.max_segment_duration):
                try:
                    acoustic_segments = self.process_acoustic_fast(audio_data, start_time)
                    if acoustic_segments and len(acoustic_segments) > 0:
                        # Use the most confident acoustic prediction
                        best_acoustic = max(acoustic_segments, key=lambda x: x.get('confidence', 0))
                        acoustic_speaker = best_acoustic['speaker']
                        acoustic_confidence = best_acoustic.get('confidence', 0.3)
                        ensemble_predictions.append((acoustic_speaker, acoustic_confidence, 0.05))  # 5% weight
                        print(f"A:{acoustic_confidence:.2f}", end='', flush=True)
                except Exception as e:
                    print(f"AC:{e}", end='', flush=True)
            
            # 4. Enhanced conversation context analysis
            try:
                context_analysis = self._analyze_conversation_context(transcription_text, audio_features)
                if context_analysis['confidence'] > 0.5:
                    ensemble_predictions.append((
                        context_analysis['speaker'], 
                        context_analysis['confidence'], 
                        0.15  # 15% weight for context
                    ))
                    print(f"C:{context_analysis['confidence']:.2f}", end='', flush=True)
            except Exception as e:
                print(f"CX:{e}", end='', flush=True)
            
            # Apply sophisticated ensemble combining
            if ensemble_predictions:
                final_speaker, final_confidence = self._ensemble_speaker_predictions(ensemble_predictions)
                
                # Get treatment phase classification
                try:
                    phase, phase_confidence = self.phase_classifier.predict_phase(transcription_text)
                except Exception as e:
                    phase, phase_confidence = "unknown", 0.0
                    print(f"PH:{e}", end='', flush=True)
                
                # Create detailed segment result
                speaker_segments.append({
                    'start': start_time,
                    'end': start_time + duration,
                    'speaker': final_speaker,
                    'duration': duration,
                    'confidence': final_confidence,
                    'method': f'advanced_ensemble_{len(ensemble_predictions)}models',
                    'ensemble_details': {
                        'predictions': ensemble_predictions,
                        'agreement_level': self._calculate_agreement_level(ensemble_predictions),
                        'primary_method': 'distilbert' if finetuned_confidence > 0.7 else 'ensemble'
                    },
                    'finetuned_prediction': (finetuned_speaker, finetuned_confidence) if 'finetuned_speaker' in locals() else None,
                    'acoustic_prediction': (acoustic_speaker, acoustic_confidence) if acoustic_speaker else None,
                    'treatment_phase': phase,
                    'phase_confidence': phase_confidence,
                    'audio_quality': self._assess_audio_quality(audio_features)
                })
                print(f"→{final_speaker}({final_confidence:.2f})", end='', flush=True)
            else:
                # No predictions available - use default
                speaker_segments.append({
                    'start': start_time,
                    'end': start_time + duration,
                    'speaker': "Doctor",
                    'duration': duration,
                    'confidence': 0.2,
                    'method': 'no_predictions_available'
                })
        
        # Strategy 2: Acoustic-Only Analysis (when no text is available)
        elif (self.use_acoustic_diarization and self.acoustic_processor and 
              self.min_segment_duration <= duration <= self.max_segment_duration):
            
            print("🎵", end='', flush=True)  # Acoustic-only indicator
            try:
                acoustic_segments = self.process_acoustic_fast(audio_data, start_time)
                if acoustic_segments:
                    # Enhance acoustic segments with audio quality assessment
                    for segment in acoustic_segments:
                        segment['audio_quality'] = self._assess_audio_quality(audio_features)
                        segment['method'] += '_acoustic_only'
                    speaker_segments.extend(acoustic_segments)
                    print(f"→{len(acoustic_segments)}segs", end='', flush=True)
            except Exception as e:
                print(f" Acoustic processing failed: {e}")
                # Fallback to audio feature-based classification
                speaker, confidence = self.fast_classifier.predict_speaker(
                    audio_features, "", "Doctor"
                )
                speaker_segments.append({
                    'start': start_time,
                    'end': start_time + duration,
                    'speaker': speaker,
                    'duration': duration,
                    'confidence': confidence,
                    'method': 'audio_features_fallback',
                    'audio_quality': self._assess_audio_quality(audio_features)
                })
        
        # Strategy 3: Simple fallback for edge cases
        else:
            print("", end='', flush=True)  # Fallback indicator
            speaker_segments.append({
                'start': start_time,
                'end': start_time + duration,
                'speaker': "Doctor",  # Conservative default
                'duration': duration,
                'confidence': 0.3,
                'method': 'conservative_default'
            })
        
        return speaker_segments
    
    def _analyze_conversation_context(self, text, audio_features):
        """Advanced conversation context analysis for medical dialogues"""
        try:
            text_lower = text.lower().strip()
            
            # Medical conversation patterns with confidence scoring
            context_scores = {
                'doctor_indicators': 0,
                'patient_indicators': 0,
                'medical_professional': 0,
                'personal_experience': 0,
                'question_pattern': 0,
                'response_pattern': 0
            }
            
            # Enhanced doctor patterns with medical context
            doctor_patterns = [
                (r'(what|how|where|when|why|which).*\?', 3),  # Questions
                (r'(can you|could you|would you|do you|have you|are you)', 3),
                (r'(tell me|describe|explain|show me)', 2),
                (r'(examination|assess|evaluate|check)', 3),
                (r'(medication|treatment|therapy|diagnosis)', 2),
                (r'(on a scale|rate your|how would you)', 3),
                (r'(medical history|any problems|symptoms)', 2),
                (r'(let.*check|let.*see|let.*examine)', 3)
            ]
            
            # Enhanced patient patterns with personal context
            patient_patterns = [
                (r'^(yes|no|yeah|okay|well|um|uh)', 2),  # Response starters
                (r'(i feel|i have|i think|i am|i was)', 3),
                (r'(my|the pain|it hurts|aches)', 3),
                (r'(started|began|been having|since)', 2),
                (r'(last night|this morning|yesterday)', 2),
                (r'(about.*hours?|around.*days?)', 2),
                (r'(breathing|dizzy|tired|sore|nauseous)', 3),
                (r'(better|worse|same|different)', 2)
            ]
            
            # Calculate pattern scores
            for pattern, weight in doctor_patterns:
                if re.search(pattern, text_lower):
                    context_scores['doctor_indicators'] += weight
                    if '?' in pattern:
                        context_scores['question_pattern'] += weight
            
            for pattern, weight in patient_patterns:
                if re.search(pattern, text_lower):
                    context_scores['patient_indicators'] += weight
                    if pattern.startswith(r'^(yes|no'):
                        context_scores['response_pattern'] += weight
            
            # Professional vs personal language analysis
            professional_terms = ['patient', 'symptoms', 'examination', 'medication', 'treatment', 'diagnosis', 'assess', 'evaluate']
            personal_terms = ['i', 'my', 'me', 'pain', 'hurt', 'feel', 'tired', 'sick']
            
            context_scores['medical_professional'] = sum(2 for term in professional_terms if term in text_lower)
            context_scores['personal_experience'] = sum(1 for term in personal_terms if term in text_lower)
            
            # Audio context integration
            if audio_features:
                pitch = audio_features.get('pitch_mean', 0)
                energy = audio_features.get('energy', 0)
                
                # Pitch-based context adjustment
                if pitch > 0:
                    if pitch > 180:  # Higher pitch might indicate patient stress/discomfort
                        context_scores['patient_indicators'] += 1
                    elif pitch < 140:  # Lower pitch might indicate doctor authority
                        context_scores['doctor_indicators'] += 1
                
                # Energy-based context adjustment
                if energy > 0.05:  # High energy might indicate doctor explaining
                    context_scores['doctor_indicators'] += 0.5
            
            # Calculate final speaker determination
            total_doctor_score = (context_scores['doctor_indicators'] + 
                                context_scores['medical_professional'] * 0.5 + 
                                context_scores['question_pattern'] * 1.5)
            
            total_patient_score = (context_scores['patient_indicators'] + 
                                 context_scores['personal_experience'] * 0.8 + 
                                 context_scores['response_pattern'] * 1.2)
            
            # Determine speaker with confidence
            if total_doctor_score > total_patient_score + 1.5:
                speaker = "Doctor"
                confidence = min(0.95, 0.6 + (total_doctor_score - total_patient_score) * 0.08)
            elif total_patient_score > total_doctor_score + 1.5:
                speaker = "Patient"
                confidence = min(0.95, 0.6 + (total_patient_score - total_doctor_score) * 0.08)
            else:
                # Ambiguous - use slight doctor bias for medical conversations
                speaker = "Doctor"
                confidence = 0.4 + abs(total_doctor_score - total_patient_score) * 0.1
            
            return {
                'speaker': speaker,
                'confidence': confidence,
                'context_scores': context_scores,
                'total_doctor_score': total_doctor_score,
                'total_patient_score': total_patient_score
            }
            
        except Exception as e:
            return {'speaker': 'Doctor', 'confidence': 0.3, 'error': str(e)}
    
    def _calculate_agreement_level(self, predictions):
        """Calculate agreement level among ensemble predictions"""
        try:
            if len(predictions) <= 1:
                return 1.0  # Perfect agreement with single prediction
            
            speakers = [pred[0] for pred in predictions]
            unique_speakers = set(speakers)
            
            # Perfect agreement
            if len(unique_speakers) == 1:
                return 1.0
            
            # Calculate weighted agreement
            total_weight = sum(pred[2] for pred in predictions)
            speaker_weights = defaultdict(float)
            
            for speaker, confidence, weight in predictions:
                speaker_weights[speaker] += weight * confidence
            
            # Highest weighted agreement ratio
            max_weighted_agreement = max(speaker_weights.values()) / total_weight if total_weight > 0 else 0.5
            
            return max_weighted_agreement
            
        except Exception as e:
            return 0.5  # Moderate agreement on error
    
    def _assess_audio_quality(self, audio_features):
        """Assess audio quality for confidence adjustment"""
        try:
            quality_score = 0.5  # Base quality
            
            if audio_features:
                # Voice activity detection
                if audio_features.get('is_voice_active', False):
                    quality_score += 0.2
                
                # Energy level assessment
                energy = audio_features.get('energy', 0)
                if 0.01 <= energy <= 0.5:  # Good energy range
                    quality_score += 0.2
                elif energy > 0.5:  # Too loud
                    quality_score -= 0.1
                
                # Pitch clarity
                pitch = audio_features.get('pitch_mean', 0)
                if pitch > 0:  # Valid pitch detected
                    quality_score += 0.1
                
                # Signal clarity (low zero crossing rate for voice)
                zcr = audio_features.get('zero_crossing_rate', 0)
                if 0.02 <= zcr <= 0.15:  # Good ZCR range for speech
                    quality_score += 0.1
            
            return max(0.1, min(1.0, quality_score))
            
        except Exception as e:
            return 0.5  # Default moderate quality
    
    def _ensemble_speaker_predictions(self, predictions):
        """Advanced ensemble combining DistilBERT, pyannote.audio, and conversation patterns"""
        try:
            if not predictions:
                return "Doctor", 0.5
            
            # Normalize speaker labels to standard format
            normalized_predictions = []
            for speaker, confidence, weight in predictions:
                normalized_speaker = self._normalize_speaker_label(speaker)
                normalized_predictions.append((normalized_speaker, confidence, weight))
            
            # Enhanced ensemble with adaptive weights based on prediction quality
            speaker_votes = defaultdict(list)  # speaker -> list of (confidence, weight, method)
            
            for speaker, confidence, weight in normalized_predictions:
                # Adaptive weight adjustment based on confidence level
                adjusted_weight = weight
                if confidence > 0.9:  # Very high confidence predictions get boosted
                    adjusted_weight *= 1.3
                elif confidence < 0.4:  # Low confidence predictions get reduced weight
                    adjusted_weight *= 0.6
                
                speaker_votes[speaker].append((confidence, adjusted_weight, weight))
            
            # Calculate sophisticated scoring with disagreement penalty
            speaker_scores = {}
            total_disagreement = 0
            
            for speaker, votes in speaker_votes.items():
                # Weighted average with confidence consideration
                weighted_sum = sum(conf * weight for conf, weight, _ in votes)
                weight_sum = sum(weight for _, weight, _ in votes)
                
                if weight_sum > 0:
                    base_score = weighted_sum / weight_sum
                    
                    # Confidence variance penalty (more consistent = better)
                    confidences = [conf for conf, _, _ in votes]
                    if len(confidences) > 1:
                        conf_std = np.std(confidences)
                        consistency_bonus = max(0, 0.1 * (1 - conf_std))  # Up to 10% bonus for consistency
                        base_score += consistency_bonus
                    
                    speaker_scores[speaker] = base_score
            
            # Calculate disagreement level across all models
            all_speakers = list(speaker_votes.keys())
            if len(all_speakers) > 1:
                total_disagreement = len(all_speakers) - 1  # More speakers = more disagreement
            
            if not speaker_scores:
                return "Doctor", 0.5
            
            # Select best speaker with sophisticated confidence calculation
            best_speaker = max(speaker_scores.keys(), key=lambda s: speaker_scores[s])
            best_confidence = speaker_scores[best_speaker]
            
            # Agreement and disagreement adjustments
            unique_predictions = set(pred[0] for pred in normalized_predictions)
            
            if len(unique_predictions) == 1:  # Perfect agreement
                # Strong confidence boost for unanimous decisions
                best_confidence = min(0.98, best_confidence * 1.25)
                method_tag = "unanimous"
                
            elif len(unique_predictions) == 2:  # Split decision
                # Moderate confidence for split decisions, favor higher-weighted models
                best_confidence = min(0.85, best_confidence * 0.9)
                method_tag = "split_decision"
                
                # Additional logic for medical conversation patterns
                # If DistilBERT (higher weight) disagrees with others, trust DistilBERT more
                distilbert_prediction = None
                acoustic_prediction = None
                
                for speaker, confidence, weight in normalized_predictions:
                    if weight >= 0.5:  # DistilBERT typically has weight 0.55+
                        distilbert_prediction = speaker
                    elif weight <= 0.3:  # Acoustic typically has weight 0.2-0.25
                        acoustic_prediction = speaker
                
                # If DistilBERT has high confidence and disagrees, trust it
                if distilbert_prediction and distilbert_prediction != acoustic_prediction:
                    for speaker, confidence, weight in normalized_predictions:
                        if speaker == distilbert_prediction and weight >= 0.5 and confidence > 0.8:
                            best_speaker = distilbert_prediction
                            best_confidence = min(0.95, confidence * 1.1)
                            method_tag = "distilbert_override"
                            break
            else:
                # High disagreement - use conservative confidence
                best_confidence = min(0.7, best_confidence * 0.8)
                method_tag = "high_disagreement"
            
            # Medical context boost for very clear patterns
            text_based_predictions = [pred for pred in normalized_predictions if pred[2] >= 0.4]  # High-weight predictions (text-based)
            if text_based_predictions:
                avg_text_confidence = sum(pred[1] for pred in text_based_predictions) / len(text_based_predictions)
                if avg_text_confidence > 0.9:  # Very confident text-based prediction
                    best_confidence = min(0.98, best_confidence * 1.15)
                    method_tag += "_text_boost"
            
            # Final confidence bounds
            best_confidence = max(0.2, min(0.98, best_confidence))
            
            return best_speaker, best_confidence
            
        except Exception as e:
            print(f" Enhanced ensemble prediction failed: {e}")
            return "Doctor", 0.5
    
    def _normalize_speaker_label(self, speaker):
        """Normalize speaker labels to standard format"""
        if isinstance(speaker, str):
            speaker_lower = speaker.lower()
            if speaker_lower in ['speaker_00', 'speaker_0', '0', 'doctor']:
                return "Doctor"
            elif speaker_lower in ['speaker_01', 'speaker_1', '1', 'patient']:
                return "Patient"
            elif 'doctor' in speaker_lower:
                return "Doctor"
            elif 'patient' in speaker_lower:
                return "Patient"
        return "Doctor"  # Default fallback
    
    def process_transcription_batch(self, transcriptions):
        """Enhanced batch processing with sophisticated ensemble for multiple transcriptions"""
        if not transcriptions:
            return []
        
        try:
            # Extract texts and prepare for batch processing
            texts = [t.get('text', '') for t in transcriptions if t.get('text', '').strip()]
            if not texts:
                return []
            
            print(f" Processing {len(texts)} transcriptions in batch...", end='', flush=True)
            
            # Get batch predictions from fine-tuned classifier (most efficient)
            finetuned_results = []
            try:
                finetuned_results = self.finetuned_classifier.predict_batch(texts)
                print("D", end='', flush=True)
            except Exception as e:
                print(f"FT_batch:{e}", end='', flush=True)
                finetuned_results = [("Doctor", 0.3) for _ in texts]
            
            # Get batch phase predictions
            phase_results = []
            try:
                phase_results = self.phase_classifier.predict_batch(texts)
                print("P", end='', flush=True)
            except Exception as e:
                print(f"PH_batch:{e}", end='', flush=True)
                phase_results = [("unknown", 0.0) for _ in texts]
            
            # Process each transcription with enhanced ensemble
            results = []
            for i, transcription in enumerate(transcriptions):
                text = transcription.get('text', '').strip()
                if len(text) < 3:  # Skip very short texts
                    continue
                
                try:
                    # Prepare ensemble predictions
                    ensemble_predictions = []
                    
                    # 1. Fine-tuned DistilBERT prediction (primary)
                    if i < len(finetuned_results):
                        ft_speaker, ft_confidence = finetuned_results[i]
                        ensemble_predictions.append((ft_speaker, ft_confidence, 0.80))  # 80% weight in batch
                    
                    # 2. Fast pattern analysis (secondary)
                    try:
                        fast_speaker, fast_confidence = self.fast_classifier.predict_speaker(
                            {}, text, transcription.get('previous_speaker', 'Doctor')
                        )
                        ensemble_predictions.append((fast_speaker, fast_confidence, 0.15))  # 15% weight
                    except Exception as fast_error:
                        print(f"F{i}:{fast_error}", end='', flush=True)
                    
                    # 3. Enhanced context analysis (tertiary)
                    try:
                        context_analysis = self._analyze_conversation_context(text, None)
                        if context_analysis['confidence'] > 0.4:
                            ensemble_predictions.append((
                                context_analysis['speaker'],
                                context_analysis['confidence'],
                                0.05  # 5% weight for context in batch
                            ))
                    except Exception as context_error:
                        print(f"C{i}:{context_error}", end='', flush=True)
                    
                    # Apply ensemble combining
                    if ensemble_predictions:
                        final_speaker, final_confidence = self._ensemble_speaker_predictions(ensemble_predictions)
                    else:
                        final_speaker, final_confidence = "Doctor", 0.3
                    
                    # Get phase information
                    phase, phase_confidence = ("unknown", 0.0)
                    if i < len(phase_results):
                        phase, phase_confidence = phase_results[i]
                    
                    # Create enhanced result
                    result = {
                        'text': text,
                        'speaker': final_speaker,
                        'confidence': final_confidence,
                        'method': f'batch_ensemble_{len(ensemble_predictions)}models',
                        'timestamp': transcription.get('timestamp', time.time()),
                        'treatment_phase': phase,
                        'phase_confidence': phase_confidence,
                        'ensemble_details': {
                            'predictions': ensemble_predictions,
                            'agreement_level': self._calculate_agreement_level(ensemble_predictions),
                            'processing_mode': 'batch'
                        },
                        'batch_index': i,
                        'batch_size': len(texts)
                    }
                    
                    # Add individual model results for debugging
                    if i < len(finetuned_results):
                        result['finetuned_prediction'] = finetuned_results[i]
                    
                    results.append(result)
                    
                except Exception as item_error:
                    print(f"Item{i}:{item_error}", end='', flush=True)
                    # Fallback result
                    results.append({
                        'text': text,
                        'speaker': 'Doctor',
                        'confidence': 0.2,
                        'method': 'batch_fallback',
                        'timestamp': transcription.get('timestamp', time.time()),
                        'error': str(item_error),
                        'batch_index': i
                    })
            
            print(f"→{len(results)} processed", end='', flush=True)
            return results
            
        except Exception as e:
            print(f" Enhanced batch processing failed: {e}")
            # Fallback to simple processing
            return [{
                'text': t.get('text', ''),
                'speaker': 'Doctor',
                'confidence': 0.2,
                'method': 'batch_error_fallback',
                'timestamp': t.get('timestamp', time.time()),
                'error': str(e)
            } for t in transcriptions if t.get('text', '').strip()]
    
    def classify_transcription_with_ensemble(self, text, previous_speaker="Doctor"):
        """Classify a single transcription using both classifiers with ensemble"""
        if not text or len(text.strip()) < 3:
            return previous_speaker, 0.3, "too_short"
        
        # Pre-process long segments with mixed content
        text_cleaned = self._preprocess_mixed_content(text)
        
        try:
            # Get predictions from both classifiers
            fast_speaker, fast_confidence = self.fast_classifier.predict_speaker(
                {}, text_cleaned, previous_speaker
            )
            
            finetuned_speaker, finetuned_confidence = self.finetuned_classifier.predict_speaker(text_cleaned)
            
            # Use ensemble method with higher weight for fine-tuned model
            final_speaker, final_confidence = self._ensemble_speaker_predictions([
                (fast_speaker, fast_confidence, 0.2),       # 20% weight for pattern-based
                (finetuned_speaker, finetuned_confidence, 0.8)  # 80% weight for fine-tuned
            ])
            
            # Enhanced medical conversation logic with stronger pattern detection
            text_lower = text_cleaned.lower().strip()
            
            # Strong question patterns (definitely doctor)
            strong_question_patterns = [
                'what', 'where', 'when', 'why', 'how', 'describe', 'tell me',
                'can you', 'do you', 'have you', 'are you', 'will you',
                'on a scale', 'rate your', 'point to', 'show me', 'any problems',
                'have you had', 'how about', 'any history', 'feeling sick'
            ]
            
            # Strong answer patterns (definitely patient)
            strong_answer_patterns = [
                'i feel', 'i have', 'i think', 'i am', 'my pain', 'it hurts',
                'yes', 'no', 'yeah', 'okay', 'sure', 'it started', 'about',
                'i guess', 'i don\'t', 'i can\'t', 'i was', 'i will'
            ]
            
            # Additional medical conversation patterns
            doctor_phrases = [
                'any symptoms', 'medical history', 'examination', 'treatment',
                'diagnosis', 'medication', 'hospital', 'check', 'test'
            ]
            
            patient_phrases = [
                'i\'m having', 'it feels', 'the pain', 'i\'m not', 'it\'s been',
                'since', 'last night', 'this morning', 'i noticed'
            ]
            
            # Check for mixed content (both questions and answers in same text)
            has_question = (text_lower.endswith('?') or 
                          any(pattern in text_lower for pattern in strong_question_patterns))
            has_answer = any(pattern in text_lower for pattern in strong_answer_patterns)
            
            # Override with high confidence if clear patterns
            if has_question and not has_answer:  # Pure question
                final_speaker = "Doctor"
                final_confidence = min(0.98, final_confidence * 1.4)
            elif has_answer and not has_question:  # Pure answer
                final_speaker = "Patient"  
                final_confidence = min(0.98, final_confidence * 1.4)
            elif has_question and has_answer:  # Mixed content - use context
                # Count question vs answer indicators
                question_count = sum(1 for p in strong_question_patterns if p in text_lower)
                answer_count = sum(1 for p in strong_answer_patterns if p in text_lower)
                
                if question_count > answer_count:
                    final_speaker = "Doctor"
                    final_confidence = min(0.95, final_confidence * 1.2)
                else:
                    final_speaker = "Patient"
                    final_confidence = min(0.95, final_confidence * 1.2)
            elif any(phrase in text_lower for phrase in doctor_phrases):
                final_speaker = "Doctor"
                final_confidence = min(0.92, final_confidence * 1.1)
            elif any(phrase in text_lower for phrase in patient_phrases):
                final_speaker = "Patient"
                final_confidence = min(0.92, final_confidence * 1.1)
            
            method = f"ensemble(ft:{finetuned_confidence:.2f},fast:{fast_confidence:.2f})"
            
            return final_speaker, final_confidence, method
            
        except Exception as e:
            print(f" Ensemble classification failed: {e}")
            return previous_speaker, 0.3, "fallback"
    
    def _preprocess_mixed_content(self, text):
        """Preprocess text to handle mixed question-answer content"""
        try:
            # Clean and normalize the text
            text = text.strip()
            if not text:
                return text
            
            # Split on sentence boundaries more intelligently
            import re
            
            # Split on question marks, periods, and exclamation marks
            sentences = re.split(r'[.?!]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if len(sentences) <= 1:
                return text
            
            # Analyze each sentence and find the most significant one
            best_sentence = ""
            best_score = 0
            
            for sentence in sentences:
                if len(sentence) < 3:
                    continue
                    
                s_lower = sentence.lower().strip()
                score = 0
                
                # Strong question indicators (Doctor patterns)
                question_patterns = ['what', 'where', 'when', 'why', 'how', 'do you', 'have you', 'can you', 'are you', 'will you']
                if any(q in s_lower for q in question_patterns):
                    score += 3
                
                # Question endings
                if s_lower.endswith('?') or '?' in sentence:
                    score += 2
                
                # Strong answer indicators (Patient patterns)
                answer_patterns = ['yes', 'no', 'i feel', 'i have', 'i think', 'i am', 'it hurts', 'the pain']
                if any(a in s_lower for a in answer_patterns):
                    score += 3
                
                # Medical context
                medical_patterns = ['pain', 'hurt', 'ache', 'feel', 'started', 'night', 'morning']
                if any(m in s_lower for m in medical_patterns):
                    score += 1
                
                # Prefer longer, more complete sentences
                if len(sentence) > 10:
                    score += 1
                
                if score > best_score:
                    best_score = score
                    best_sentence = sentence
            
            # Return the most significant sentence, or original text if no clear winner
            return best_sentence if best_sentence and best_score > 2 else text
            
        except Exception as e:
            return text
    
    def _detect_voice_activity(self, audio_data):
        """Advanced Voice Activity Detection to filter out silence and noise"""
        if len(audio_data) == 0:
            return False
        
        try:
            # Calculate multiple VAD features
            abs_audio = np.abs(audio_data)
            
            # 1. Energy-based detection
            energy = np.mean(abs_audio)
            energy_threshold = 0.01
            
            # 2. Zero crossing rate
            zcr = len(np.where(np.diff(np.signbit(audio_data)))[0]) / len(audio_data)
            zcr_threshold = 0.15  # Voice typically has lower ZCR
            
            # 3. Spectral analysis for voice characteristics
            if len(audio_data) > 512:
                fft = np.fft.rfft(audio_data)
                magnitude = np.abs(fft)
                
                # Check for spectral energy in voice frequency range (80-300 Hz)
                freqs = np.fft.rfftfreq(len(audio_data), 1/16000)
                voice_range_mask = (freqs >= 80) & (freqs <= 300)
                voice_energy = np.sum(magnitude[voice_range_mask])
                total_energy = np.sum(magnitude)
                
                voice_ratio = voice_energy / (total_energy + 1e-10)
                voice_ratio_threshold = 0.1
                
                # Combined VAD decision
                is_voice = (energy > energy_threshold and 
                           zcr < zcr_threshold and 
                           voice_ratio > voice_ratio_threshold)
            else:
                # Simple VAD for short segments
                is_voice = energy > energy_threshold and zcr < zcr_threshold
            
            return is_voice
            
        except Exception as e:
            # Fallback to simple energy-based detection
            energy = np.mean(np.abs(audio_data))
            return energy > 0.01
    
    def process_acoustic_fast(self, audio_data, start_time):
        """Three-stage processing: segmentation, embeddings, clustering with overlap detection"""
        if not self.acoustic_processor:
            return None
            
        try:
            # Prepare mono 16kHz audio tensor as required by pyannote
            audio_tensor = torch.from_numpy(audio_data.astype(np.float32))
            if len(audio_tensor.shape) == 1:
                audio_tensor = audio_tensor.unsqueeze(0)  # Add batch dimension
            
            # Ensure mono audio for pyannote
            if audio_tensor.shape[0] > 1:
                audio_tensor = torch.mean(audio_tensor, dim=0, keepdim=True)
            
            # Move tensor to the same device as the models (CRITICAL FIX)
            audio_tensor = audio_tensor.to(self.device)
            
            # Preprocess for optimal diarization
            audio_tensor = self._preprocess_audio_for_diarization(audio_tensor)
            
            # === STAGE 1: Voice Activity Detection & Segmentation ===
            print("", end='', flush=True)  # Segmentation indicator
            
            voice_activity = None
            speaker_changes = []
            
            if self.segmentation_model:
                try:
                    # Get voice activity detection and speaker change points
                    segmentation_scores = self.segmentation_model(audio_tensor.unsqueeze(0))
                    
                    # Extract voice activity and speaker boundaries - safe bounds checking
                    if segmentation_scores.shape[0] > 0 and segmentation_scores.shape[2] > 0:
                        voice_activity = segmentation_scores[0, :, 0] > self.segmentation_threshold
                        speaker_changes = self._detect_speaker_changes(segmentation_scores[0])
                    else:
                        print(" Invalid segmentation scores shape")
                        voice_activity = torch.ones(max(1, audio_tensor.shape[1] // 160))
                        speaker_changes = []
                    
                except Exception as e:
                    print(f" Segmentation failed: {e}")
                    # Safe fallback with bounds checking
                    fallback_size = max(1, audio_tensor.shape[1] // 160) if audio_tensor.shape[1] > 0 else 1
                    voice_activity = torch.ones(fallback_size)  # Fallback: assume all voice
                    speaker_changes = []
            else:
                # Safe fallback with bounds checking
                fallback_size = max(1, audio_tensor.shape[1] // 160) if audio_tensor.shape[1] > 0 else 1
                voice_activity = torch.ones(fallback_size)
                speaker_changes = []
            
            # === STAGE 2: Neural Speaker Embeddings ===
            print("", end='', flush=True)  # Embedding indicator
            
            embeddings = []
            embedding_segments = []
            
            if self.embedding_model:
                try:
                    # Create sliding windows for embedding extraction
                    window_samples = int(self.embedding_window_size * self.target_sample_rate)
                    step_samples = int(self.embedding_step_size * self.target_sample_rate)
                    
                    # Ensure audio tensor is large enough for windowing
                    if audio_tensor.shape[1] < window_samples:
                        print("⚠️ Audio segment too short for embedding extraction")
                        embeddings = []
                        embedding_segments = []
                    else:
                        max_range = max(0, audio_tensor.shape[1] - window_samples)
                        for i in range(0, max_range, step_samples):
                            window_start = i / self.target_sample_rate + start_time
                            window_end = (i + window_samples) / self.target_sample_rate + start_time
                            
                            # Check if this window has voice activity - safe bounds checking
                            window_start_frame = max(0, i // 160)
                            window_end_frame = min(len(voice_activity), (i + window_samples) // 160)
                            
                            # Ensure valid frame range
                            if (window_start_frame < len(voice_activity) and 
                                window_end_frame <= len(voice_activity) and
                                window_start_frame < window_end_frame):
                                
                                window_voice_ratio = voice_activity[window_start_frame:window_end_frame].float().mean()
                                
                                if window_voice_ratio > 0.5:  # At least 50% voice activity
                                    # Extract speaker embedding for this window - safe bounds
                                    end_idx = min(i + window_samples, audio_tensor.shape[1])
                                    audio_window = audio_tensor[:, i:end_idx]
                                    
                                    if audio_window.shape[1] == window_samples:
                                        try:
                                            embedding = self.embedding_model(audio_window)
                                            embeddings.append(embedding.cpu().numpy().flatten())
                                            embedding_segments.append({
                                                'start': window_start,
                                                'end': window_end,
                                                'embedding': embedding.cpu().numpy().flatten()
                                            })
                                        except Exception as e:
                                            print(f"⚠️ Embedding extraction failed: {e}")
                                            continue
                
                except Exception as e:
                    print(f" Embedding stage failed: {e}")
                    embeddings = []
                    embedding_segments = []
            
            # === STAGE 3: Agglomerative Clustering ===
            print("", end='', flush=True)  # Clustering indicator
            
            if len(embeddings) >= 2:
                try:
                    from sklearn.cluster import AgglomerativeClustering
                    
                    # Perform agglomerative clustering on embeddings
                    embeddings_array = np.array(embeddings)
                    
                    # Use 2 clusters for speaker diarization
                    clustering = AgglomerativeClustering(
                        n_clusters=2,
                        linkage='ward',
                        metric='euclidean'
                    )
                    
                    cluster_labels = clustering.fit_predict(embeddings_array)
                    
                    # Map cluster labels to speaker segments
                    segments = []
                    current_speaker = None
                    current_start = None
                    
                    for i, (segment, cluster_label) in enumerate(zip(embedding_segments, cluster_labels)):
                        speaker_label = f"Doctor" if cluster_label == 0 else f"Patient"
                        
                        # Check for speaker change or overlapped speech
                        if current_speaker != speaker_label:
                            if current_speaker is not None and current_start is not None:
                                # Close previous segment
                                segments.append({
                                    'start': current_start,
                                    'end': embedding_segments[i-1]['end'],
                                    'speaker': current_speaker,
                                    'duration': embedding_segments[i-1]['end'] - current_start,
                                    'confidence': 0.9,
                                    'method': 'three_stage_clustering',
                                    'embedding': embeddings_array[i-1]
                                })
                            
                            current_speaker = speaker_label
                            current_start = segment['start']
                    
                    # Close final segment
                    if current_speaker is not None and current_start is not None:
                        segments.append({
                            'start': current_start,
                            'end': embedding_segments[-1]['end'],
                            'speaker': current_speaker,
                            'duration': embedding_segments[-1]['end'] - current_start,
                            'confidence': 0.9,
                            'method': 'three_stage_clustering',
                            'embedding': embeddings_array[-1]
                        })
                    
                    # Detect overlapped speech regions
                    overlap_segments = self._detect_overlapped_speech(segments, embeddings_array, cluster_labels)
                    segments.extend(overlap_segments)
                    
                    # Generate RTTM format output
                    self._add_to_rttm(segments, start_time)
                    
                    print(f" → {len(segments)} segments", end='', flush=True)
                    return segments
                    
                except Exception as e:
                    print(f" Clustering failed: {e}")
                    # Fallback to simple alternating speakers
                    segments = self._fallback_speaker_assignment(embedding_segments, start_time)
                    return segments
            
            else:
                # Fallback: use full pipeline if individual stages fail
                print("", end='', flush=True)  # Fallback indicator
                return self._fallback_full_pipeline(audio_tensor, start_time)
                
        except Exception as e:
            print(f" Three-stage processing failed: {e}")
            return None
    
    def _detect_speaker_changes(self, segmentation_scores):
        """Detect speaker change points from segmentation scores"""
        try:
            # Safe bounds checking for segmentation scores
            if (segmentation_scores is None or 
                segmentation_scores.shape[0] < 2 or 
                segmentation_scores.shape[1] < 2):
                return []
            
            # Find peaks in speaker change scores (assuming multi-class segmentation)
            if segmentation_scores.shape[1] > 1:
                # Calculate speaker change likelihood with safe operations
                try:
                    change_scores = torch.diff(segmentation_scores, dim=0).abs().sum(dim=1)
                    if change_scores.numel() == 0:  # Check if tensor is empty
                        return []
                    
                    # Find significant changes above threshold
                    score_mean = change_scores.mean()
                    score_std = change_scores.std()
                    change_threshold = score_mean + score_std
                    change_points = torch.where(change_scores > change_threshold)[0]
                    
                    if change_points.numel() == 0:  # No change points found
                        return []
                    
                    return change_points.cpu().numpy() * 160 / self.target_sample_rate  # Convert to seconds
                except Exception as calc_error:
                    print(f" Speaker change calculation failed: {calc_error}")
                    return []
            else:
                return []
        except Exception as e:
            print(f"Speaker change detection failed: {e}")
            return []
    
    def _preprocess_audio_for_diarization(self, audio_tensor):
        """Advanced audio preprocessing for better diarization accuracy"""
        try:
            # Apply a simple high-pass filter to remove low-frequency noise
            # This helps with voice clarity for diarization
            audio_np = audio_tensor.squeeze().numpy()
            
            # Simple high-pass filter using difference
            if len(audio_np) > 1:
                filtered = np.diff(audio_np, prepend=audio_np[0])
                audio_np = 0.7 * audio_np + 0.3 * filtered  # Blend original with filtered
            
            # Normalize audio to optimal range for pyannote
            max_val = np.max(np.abs(audio_np))
            if max_val > 0:
                audio_np = audio_np / max_val * 0.8  # Normalize to 80% of max range
            
            # Apply gentle windowing to reduce edge artifacts - safe bounds checking
            window_size = min(320, len(audio_np) // 10)  # 20ms window or 10% of signal
            if window_size > 10 and len(audio_np) >= window_size * 2:
                try:
                    window = np.hanning(window_size * 2)
                    # Safe bounds checking for fade operations
                    if len(audio_np) >= window_size and len(window) >= window_size:
                        # Apply fade-in - safe slicing
                        audio_np[:window_size] *= window[:window_size]
                        # Apply fade-out - safe slicing
                        if len(audio_np) >= window_size:
                            audio_np[-window_size:] *= window[window_size:window_size*2]
                except Exception as window_error:
                    print(f"⚠️ Windowing failed: {window_error}")
                    # Skip windowing if it fails
            
            return torch.from_numpy(audio_np).unsqueeze(0).to(self.device)
            
        except Exception as e:
            # Return original audio if preprocessing fails - ensure correct device
            return audio_tensor.to(self.device)
    
    def _detect_overlapped_speech(self, segments, embeddings, cluster_labels):
        """Detect overlapped speech regions using embedding similarity"""
        overlap_segments = []
        try:
            # Look for temporal overlaps with different speakers
            for i, seg1 in enumerate(segments):
                for j, seg2 in enumerate(segments[i+1:], i+1):
                    # Check for temporal overlap
                    overlap_start = max(seg1['start'], seg2['start'])
                    overlap_end = min(seg1['end'], seg2['end'])
                    
                    if overlap_start < overlap_end and seg1['speaker'] != seg2['speaker']:
                        # Check embedding similarity to confirm different speakers
                        if 'embedding' in seg1 and 'embedding' in seg2:
                            similarity = self._calculate_embedding_similarity(
                                seg1['embedding'], seg2['embedding']
                            )
                            
                            # If embeddings are sufficiently different, mark as overlap
                            if similarity < self.overlap_threshold:
                                overlap_segments.append({
                                    'start': overlap_start,
                                    'end': overlap_end,
                                    'speaker': 'OVERLAP',
                                    'duration': overlap_end - overlap_start,
                                    'confidence': 0.8,
                                    'method': 'overlap_detection',
                                    'speakers': [seg1['speaker'], seg2['speaker']]
                                })
                                
                                # Store for RTTM output
                                self.overlap_regions.append({
                                    'start': overlap_start,
                                    'end': overlap_end,
                                    'speakers': [seg1['speaker'], seg2['speaker']]
                                })
            
            return overlap_segments
            
        except Exception as e:
            print(f" Overlap detection failed: {e}")
            return []
    
    def _add_to_rttm(self, segments, start_time):
        """Add segments to RTTM format output"""
        try:
            for segment in segments:
                # RTTM format: SPEAKER <filename> <channel> <start> <duration> <ortho> <stype> <name> <conf> <slat>
                rttm_entry = {
                    'type': 'SPEAKER',
                    'filename': f'audio_{int(start_time)}',
                    'channel': 1,
                    'start': segment['start'],
                    'duration': segment['duration'],
                    'ortho': '<NA>',
                    'stype': '<NA>',
                    'speaker': segment['speaker'],
                    'confidence': segment.get('confidence', 0.9),
                    'slat': '<NA>'
                }
                self.rttm_segments.append(rttm_entry)
                
        except Exception as e:
            print(f" RTTM generation failed: {e}")
    
    def _fallback_speaker_assignment(self, embedding_segments, start_time):
        """Fallback speaker assignment when clustering fails"""
        segments = []
        try:
            # Simple alternating speaker assignment
            for i, segment in enumerate(embedding_segments):
                speaker_id = i % 2  # Alternate between 0 and 1
                segments.append({
                    'start': segment['start'],
                    'end': segment['end'],
                    'speaker': f'Doctor' if speaker_id == 0 else f'Patient',
                    'duration': segment['end'] - segment['start'],
                    'confidence': 0.7,  # Lower confidence for fallback
                    'method': 'fallback_alternating',
                    'embedding': segment.get('embedding', None)
                })
            return segments
        except Exception as e:
            print(f" Fallback assignment failed: {e}")
            return []
    
    def _fallback_full_pipeline(self, audio_tensor, start_time):
        """Fallback to complete diarization pipeline"""
        try:
            # Use the complete pipeline as fallback
            audio_dict = {
                "waveform": audio_tensor.to(self.device),  # Ensure correct device
                "sample_rate": self.target_sample_rate
            }
            
            diarization = self.acoustic_processor(audio_dict, num_speakers=2)
            
            # Convert to our segment format
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append({
                    'start': turn.start + start_time,
                    'end': turn.end + start_time,
                    'speaker': speaker,
                    'duration': turn.end - turn.start,
                    'confidence': 0.85,
                    'method': 'full_pipeline_fallback'
                })
            
            return segments
            
        except Exception as e:
            print(f" Full pipeline fallback failed: {e}")
            return []
    
    def export_rttm(self, filename=None):
        """Export diarization results in RTTM format"""
        if filename is None:
            filename = f"diarization_{int(time.time())}.rttm"
        
        try:
            with open(filename, 'w') as f:
                for entry in self.rttm_segments:
                    # RTTM format line
                    line = f"SPEAKER {entry['filename']} {entry['channel']} " \
                           f"{entry['start']:.3f} {entry['duration']:.3f} " \
                           f"{entry['ortho']} {entry['stype']} {entry['speaker']} " \
                           f"{entry['confidence']:.3f} {entry['slat']}\n"
                    f.write(line)
            
            print(f" RTTM file exported: {filename}")
            return filename
            
        except Exception as e:
            print(f" RTTM export failed: {e}")
            return None
    
    def _process_segments_with_overlap_handling(self, diarization, start_time, audio_tensor):
        """Process diarization segments with advanced overlap detection"""
        segments = []
        previous_end = 0.0
        
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            # Detect potential overlapping speech
            overlap_score = 0.0
            if previous_end > turn.start:
                overlap_duration = previous_end - turn.start
                overlap_score = min(1.0, overlap_duration / turn.duration) if turn.duration > 0 else 0.0
            
            # Extract speaker embedding for consistency tracking
            try:
                # Get speaker embedding from the diarization result
                segment_audio = audio_tensor[:, int(turn.start * SAMPLING_RATE):int(turn.end * SAMPLING_RATE)]
                if segment_audio.shape[1] > 1600:  # Minimum samples for embedding
                    # Use pyannote's embedding model if available
                    if hasattr(self.acoustic_processor, '_segmentation') and hasattr(self.acoustic_processor._segmentation, 'model'):
                        try:
                            # Extract embedding using the segmentation model
                            embedding_input = {
                                "waveform": segment_audio,
                                "sample_rate": SAMPLING_RATE
                            }
                            # Get embedding from the model
                            with torch.no_grad():
                                embedding = self.acoustic_processor._segmentation.model(embedding_input)
                                if isinstance(embedding, torch.Tensor):
                                    embedding_vector = embedding.cpu().numpy().flatten()
                                    
                                    # Store embedding for speaker consistency
                                    speaker_key = f"{speaker}_{turn.start:.1f}"
                                    self.speaker_embedding_cache[speaker_key] = embedding_vector
                                else:
                                    embedding_vector = None
                        except Exception as emb_error:
                            embedding_vector = None
                    else:
                        embedding_vector = None
                else:
                    embedding_vector = None
            except Exception as e:
                embedding_vector = None
            
            # Advanced speaker mapping with consistency tracking
            mapped_speaker = self._map_speaker_with_consistency(speaker, turn)
            
            # Validate speaker assignment using embedding similarity if available
            if embedding_vector is not None:
                validated_speaker = self._validate_speaker_with_embedding(mapped_speaker, embedding_vector)
                if validated_speaker != mapped_speaker:
                    print(f" Embedding-based speaker correction: {mapped_speaker} -> {validated_speaker}")
                    mapped_speaker = validated_speaker
            
            # Adjust confidence based on overlap
            base_confidence = 0.95
            if overlap_score > 0.3:  # Significant overlap
                base_confidence *= (1.0 - overlap_score * 0.3)  # Reduce confidence
            
            segment = {
                'start': turn.start + start_time,
                'end': turn.end + start_time,
                'speaker': mapped_speaker,
                'duration': turn.end - turn.start,
                'confidence': base_confidence,
                'method': 'advanced_segmentation',
                'overlap_score': overlap_score,
                'embedding': embedding_vector  # Store embedding for future use
            }
            segments.append(segment)
            previous_end = turn.end
        
        return segments
    
    def _validate_speaker_with_embedding(self, suggested_speaker, new_embedding):
        """Validate speaker assignment using embedding similarity"""
        if new_embedding is None or len(new_embedding) == 0:
            return suggested_speaker
        
        try:
            # Check if we have stored embeddings for both speakers
            if suggested_speaker in self.speaker_embeddings:
                stored_embedding = self.speaker_embeddings[suggested_speaker]
                similarity = self._calculate_embedding_similarity(new_embedding, stored_embedding)
                
                # If similarity is too low, check the other speaker
                if similarity < self.embedding_similarity_threshold:
                    other_speaker = "Patient" if suggested_speaker == "Doctor" else "Doctor"
                    if other_speaker in self.speaker_embeddings:
                        other_similarity = self._calculate_embedding_similarity(new_embedding, self.speaker_embeddings[other_speaker])
                        
                        # If other speaker has higher similarity, switch
                        if other_similarity > similarity and other_similarity > self.embedding_similarity_threshold:
                            return other_speaker
                
                # Update stored embedding with exponential moving average
                alpha = 0.2  # Learning rate for embedding update
                self.speaker_embeddings[suggested_speaker] = (
                    alpha * new_embedding + (1 - alpha) * stored_embedding
                )
            else:
                # Store new embedding for this speaker
                self.speaker_embeddings[suggested_speaker] = new_embedding.copy()
            
            return suggested_speaker
            
        except Exception as e:
            print(f" Embedding validation error: {e}")
            return suggested_speaker
    
    def _calculate_embedding_similarity(self, emb1, emb2):
        """Calculate cosine similarity between two embeddings"""
        try:
            # Validate inputs
            if emb1 is None or emb2 is None:
                return 0.0
            
            # Ensure embeddings are numpy arrays with safe conversion
            emb1 = np.array(emb1, dtype=np.float32).flatten()
            emb2 = np.array(emb2, dtype=np.float32).flatten()
            
            # Check if arrays are empty or have different sizes
            if emb1.size == 0 or emb2.size == 0:
                return 0.0
            
            # Ensure same dimensions (pad with zeros if needed)
            if emb1.size != emb2.size:
                max_size = max(emb1.size, emb2.size)
                emb1_padded = np.zeros(max_size, dtype=np.float32)
                emb2_padded = np.zeros(max_size, dtype=np.float32)
                emb1_padded[:emb1.size] = emb1
                emb2_padded[:emb2.size] = emb2
                emb1, emb2 = emb1_padded, emb2_padded
            
            # Calculate cosine similarity with safe division
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            
            # Check for zero norms to prevent division by zero
            if norm1 > 1e-10 and norm2 > 1e-10:
                similarity = np.dot(emb1, emb2) / (norm1 * norm2)
                # Clamp to valid range and convert to 0-1 range
                similarity = np.clip(similarity, -1.0, 1.0)
                return (similarity + 1) / 2
            else:
                return 0.0
                
        except Exception as e:
            print(f" Embedding similarity calculation error: {e}")
            return 0.0

    def _map_speaker_with_consistency(self, speaker_label, turn):
        """Advanced speaker mapping with embedding consistency tracking"""
        # Map pyannote speaker labels to medical roles
        if speaker_label in ["SPEAKER_00", "SPEAKER_0"] or speaker_label.endswith("_0"):
            mapped_speaker = "Doctor"
        else:
            mapped_speaker = "Patient"
        
        # Track speaker confidence for consistency
        confidence = getattr(turn, 'confidence', 0.9)
        self.speaker_confidence_history[mapped_speaker].append(confidence)
        
        # Keep only recent confidence scores (last 10 turns)
        if len(self.speaker_confidence_history[mapped_speaker]) > 10:
            self.speaker_confidence_history[mapped_speaker] = \
                self.speaker_confidence_history[mapped_speaker][-10:]
        
        return mapped_speaker
    
    def _process_with_file_fallback(self, audio_tensor, start_time):
        """Fallback file-based processing for compatibility"""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            try:
                torchaudio.save(tmp_file.name, audio_tensor, sample_rate=SAMPLING_RATE)
                
                # Simplified file-based processing
                diarization_result = self.acoustic_processor(
                    tmp_file.name,
                    num_speakers=2
                )
                
                segments = []
                if diarization_result:
                    for turn, _, speaker in diarization_result.itertracks(yield_label=True):
                        mapped_speaker = "Doctor" if speaker.endswith("_0") else "Patient"
                        segments.append({
                            'start': turn.start + start_time,
                            'end': turn.end + start_time,
                            'speaker': mapped_speaker,
                            'duration': turn.end - turn.start,
                            'confidence': 0.85,
                            'method': 'file_fallback'
                        })
                
                return segments
                        
            finally:
                try:
                    os.unlink(tmp_file.name)
                except:
                    pass

class MicrophoneStreamerWithDiarization:
    def scrub_phi(self, text):
        """Remove PHI (Protected Health Information) from transcriptions."""
        if not text:
            return text
        # Example patterns for PHI (names, dates, phone numbers, addresses, MRNs, emails)
        # These can be expanded for more robust scrubbing
        import re
        # Remove names (simple placeholder, real implementation should use NER)
        text = re.sub(r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b', '[REDACTED NAME]', text)
        # Remove dates
        text = re.sub(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', '[REDACTED DATE]', text)
        text = re.sub(r'\b\d{4}-\d{2}-\d{2}\b', '[REDACTED DATE]', text)
        # Remove phone numbers
        text = re.sub(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', '[REDACTED PHONE]', text)
        # Remove email addresses
        text = re.sub(r'[\w\.-]+@[\w\.-]+', '[REDACTED EMAIL]', text)
        # Remove medical record numbers (MRN)
        text = re.sub(r'\bMRN[:\s]?\d+\b', '[REDACTED MRN]', text)
        # Remove addresses (very basic)
        text = re.sub(r'\d+ [A-Za-z ]+,? [A-Za-z ]+', '[REDACTED ADDRESS]', text)
        return text
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.transcription_queue = queue.Queue()
        self.sentiment_queue = queue.Queue()  # New queue for sentiment analysis
        self.is_recording = False
        self.socket = None
        self.audio_level_threshold = 50
        
        # Initialize comprehensive sentiment analyzer
        print("Initializing Patient Sentiment Analysis System...")
        self.sentiment_analyzer = ComprehensiveSentimentAnalyzer()
        self.sentiment_enabled = True
        self.patient_segments_count = 0
        self.total_segments_count = 0
        
        # Diarization components - audio-based (tone and silence detection)
        self.diarization_processor = RealTimeDiarizationProcessor()
        self.audio_buffer_for_diarization = []
        self.buffer_start_time = 0
        self.speaker_history = defaultdict(list)  # Track speaker segments
        self.current_speaker = "Doctor"  # Default to first speaker
        self.last_speaker_change_time = 0  # Track when speaker last changed
        self.speaker_change_threshold = 0.1  # Very quick threshold for instant switching
        self.detected_speakers = set()  # Keep track of all detected speakers
        self.last_transcription_time = 0  # Track timing for fallback alternation
        self.silence_threshold = 0.3  # Quick silence detection for speaker changes
        
        # Three-stage diarization parameters
        self.segmentation_threshold = 0.5   # Voice activity detection threshold
        self.embedding_window_size = 1.5    # Window for speaker embeddings
        self.embedding_step_size = 0.75     # Step size for sliding windows
        self.clustering_threshold = 0.7     # Agglomerative clustering threshold
        self.min_cluster_size = 0.5         # Minimum cluster duration
        
        # RTTM format output support
        self.rttm_output_file = None
        self.enable_rttm_export = True
        
        # Enhanced conversation tracking - simplified
        self.recent_transcriptions = []  # Store recent transcriptions
        self.conversation_turn_count = 0  # Track conversation flow
        self.last_transcription_time = 0  # Track timing for speaker alternation
        
        # Quick sentence output - no consolidation
        self.pending_fragments = defaultdict(str)  # Store current sentence per speaker
        self.pending_original_fragments = defaultdict(str)  # Store original (pre-translation) text
        self.pending_translation_info = defaultdict(lambda: {'was_translated': False, 'language': 'English'})
        self.last_output_time = defaultdict(float)  # Track last output per speaker
        
        # Audio feature tracking for segmentation (fix segmentation fault)
        self.last_audio_features = None  # Store previous audio features for comparison
        self.tone_change_threshold = 0.2  # Threshold for detecting tone changes
        self.min_speaker_duration = 1.0   # Minimum duration before speaker switch
        
        # Threading locks
        self.speaker_lock = threading.Lock()
        self.sentiment_lock = threading.Lock()
        
        # Sentiment analysis threading
        self.sentiment_thread = None
        self.sentiment_running = False
        
        # Enhanced metrics tracking for sentiment, DER, and latency
        self.sentiment_history = []
        self.patient_sentiments = []
        self.diarization_errors = 0
        self.total_diarization_segments = 0
        self.latency_measurements = []
        self.processing_start_times = {}  # Track processing latency
        self.session_start_time = time.time()
        
        # Initialize healthcare sentiment analyzer
        try:
            from healthcare_sentiment_analyzer import HealthcareSentimentAnalyzer
            self.healthcare_sentiment = HealthcareSentimentAnalyzer()
            print(" Healthcare sentiment analyzer loaded")
        except ImportError as ie:
            print(f" Healthcare sentiment analyzer module not found: {ie}")
            self.healthcare_sentiment = None
        except Exception as e:
            print(f" Healthcare sentiment analyzer failed to load: {e}")
            self.healthcare_sentiment = None
        
        # Initialize Google Translate
        try:
            if Translator:
                self.translator = Translator()
                self.translation_enabled = True
                print("Google Translate initialized")
                print("   Auto-translation for non-English text enabled")
            else:
                self.translator = None
                self.translation_enabled = False
                print(" Google Translate not available")
        except Exception as e:
            print(f" Google Translate initialization failed: {e}")
            self.translator = None
            self.translation_enabled = False
    
    def detect_and_translate_text(self, text):
        """
        Detect language and translate to English if needed
        Returns: (translated_text, original_language, was_translated)
        """
        if not self.translation_enabled or not self.translator or not text.strip():
            return text, 'en', False
        
        try:
            # Detect language
            detection = self.translator.detect(text)
            detected_language = detection.lang
            confidence = detection.confidence
            
            # Get language name
            language_name = LANGUAGES.get(detected_language, detected_language).title()
            
            # Only translate if not English and confidence is reasonable
            if detected_language != 'en' and confidence > 0.5:
                # Translate to English
                translation = self.translator.translate(text, dest='en', src=detected_language)
                translated_text = translation.text
                
                return translated_text, detected_language, True
            else:
                return text, detected_language, False
                
        except Exception as e:
            # If translation fails, return original text
            print(f" Translation error: {e}")
            return text, 'unknown', False
    
    def calculate_patient_satisfaction_score(self):
        """
        Calculate patient satisfaction using Weighted Average Aggregation 
        with Dampened Negative Impact methodology
        """
        try:
            if not self.patient_sentiments:
                return {
                    'satisfaction_score': 50.0,
                    'classification': 'Unknown',
                    'details': 'No patient sentiment data available'
                }
            
            # Step 1: Extract sentiment scores and prepare weights
            weighted_scores = []
            total_weight = 0
            
            for sentiment_data in self.patient_sentiments:
                # Extract numerical sentiment score (-3 to +3 scale)
                sentiment_score = sentiment_data.get('score', 0)
                confidence = sentiment_data.get('confidence', 0.5)
                phase = sentiment_data.get('phase', 'during')
                
                # Step 1: Phase importance weights
                phase_weights = {
                    'before': 0.8,  # Baseline expectations
                    'during': 1.2,  # Most important - treatment experience  
                    'after': 1.0    # Outcome satisfaction
                }
                phase_weight = phase_weights.get(phase, 1.0)
                
                # Confidence-based weight scaling
                confidence_weight = 0.5 + 0.5 * confidence
                
                # Combined weight
                final_weight = phase_weight * confidence_weight
                
                # Step 2: Apply dampening function to negative scores
                alpha = 0.5  # Dampening factor for negatives
                dampened_score = sentiment_score if sentiment_score >= 0 else alpha * sentiment_score
                
                weighted_scores.append((dampened_score, final_weight))
                total_weight += final_weight
            
            # Step 3: Weighted Average Aggregation
            if total_weight == 0:
                weighted_average = 0
            else:
                weighted_sum = sum(score * weight for score, weight in weighted_scores)
                weighted_average = weighted_sum / total_weight
            
            # Step 4: Normalization & Scaling
            k = 1.0  # Scaling constant
            normalized_score = np.tanh(k * weighted_average)  # Constrain to [-1, 1]
            
            # Map to 0-100 scale
            satisfaction_score = 50 + (normalized_score * 50)
            satisfaction_score = max(0, min(100, satisfaction_score))
            
            # Step 5: Classification
            if satisfaction_score >= 80:
                classification = "Excellent"
            elif satisfaction_score >= 65:
                classification = "Good"
            elif satisfaction_score >= 45:
                classification = "Satisfactory"
            elif satisfaction_score >= 30:
                classification = "Poor"
            else:
                classification = "Very Poor"
            
            # Detailed breakdown
            details = {
                'total_sentiments': len(self.patient_sentiments),
                'weighted_average': weighted_average,
                'normalized_score': normalized_score,
                'dampening_factor': alpha,
                'phase_distribution': {},
                'confidence_stats': {
                    'mean_confidence': np.mean([s.get('confidence', 0.5) for s in self.patient_sentiments]),
                    'high_confidence_count': sum(1 for s in self.patient_sentiments if s.get('confidence', 0) > 0.8)
                }
            }
            
            # Calculate phase distribution
            for phase in ['before', 'during', 'after']:
                phase_count = sum(1 for s in self.patient_sentiments if s.get('phase') == phase)
                details['phase_distribution'][phase] = phase_count
            
            return {
                'satisfaction_score': round(satisfaction_score, 1),
                'classification': classification,
                'details': details
            }
            
        except Exception as e:
            return {
                'satisfaction_score': 50.0,
                'classification': 'Error',
                'details': f'Calculation error: {str(e)}'
            }
    
    def calculate_diarization_error_rate(self):
        """Calculate Diarization Error Rate (DER)"""
        try:
            if self.total_diarization_segments == 0:
                return 0.0
            
            der = (self.diarization_errors / self.total_diarization_segments) * 100
            return round(der, 2)
        except:
            return 0.0
    
    def calculate_average_latency(self):
        """Calculate average processing latency in milliseconds"""
        try:
            if not self.latency_measurements:
                return 0.0
            
            return round(np.mean(self.latency_measurements), 1)
        except:
            return 0.0

    def detect_audio_features(self, audio_data):
        """Quick audio feature detection for tone and silence analysis"""
        try:
            if len(audio_data) == 0:
                return {'energy': 0, 'pitch_estimate': 0, 'is_silence': True, 'tone_change': False}
            
            # Calculate energy (volume level)
            energy = np.mean(np.abs(audio_data))
            
            # Simple pitch estimation using zero crossing rate
            zero_crossings = len(np.where(np.diff(np.signbit(audio_data)))[0])
            pitch_estimate = zero_crossings / len(audio_data) if len(audio_data) > 0 else 0
            
            # Silence detection
            is_silence = energy < self.silence_threshold
            
            # Tone change detection (compare with previous)
            tone_change = False
            if self.last_audio_features:
                pitch_diff = abs(pitch_estimate - self.last_audio_features['pitch_estimate'])
                energy_diff = abs(energy - self.last_audio_features['energy'])
                tone_change = (pitch_diff > self.tone_change_threshold or 
                             energy_diff > self.tone_change_threshold)
            
            features = {
                'energy': energy,
                'pitch_estimate': pitch_estimate,
                'is_silence': is_silence,
                'tone_change': tone_change
            }
            
            self.last_audio_features = features
            return features
            
        except Exception as e:
            return {'energy': 0, 'pitch_estimate': 0, 'is_silence': True, 'tone_change': False}
        
    def audio_callback(self, in_data, frame_count, time_info, status):
        """Callback function for PyAudio stream with aggressive flow control"""
        if self.is_recording:
            # Convert bytes to numpy array
            audio_data = np.frombuffer(in_data, dtype=np.int16)
            
            # BOOST the audio by 10x to compensate for low microphone volume
            audio_data = audio_data.astype(np.float32)
            audio_data = audio_data * 10.0  # 10x amplification
            
            # Clip to prevent overflow and convert back to int16
            audio_data = np.clip(audio_data, -32767, 32767).astype(np.int16)
            
            # Very aggressive queue management
            queue_size = self.audio_queue.qsize()
            if queue_size > 3:  # Much lower threshold
                try:
                    # Drop ALL old chunks to prevent server overload
                    while not self.audio_queue.empty():
                        try:
                            self.audio_queue.get_nowait()
                        except queue.Empty:
                            break
                except:
                    pass
            
            # Only add audio if queue is nearly empty
            if queue_size < 2:
                self.audio_queue.put(audio_data)
            # Else: drop this chunk to prevent overload
            
            # Store audio for diarization processing (very limited buffer)
            if len(self.audio_buffer_for_diarization) > 6:  # Smaller limit for quicker processing
                self.audio_buffer_for_diarization = self.audio_buffer_for_diarization[-3:]  # Keep only last 3 chunks
            
            # Detect audio features for quick speaker switching
            normalized_audio = audio_data.astype(np.float32) / 32767.0
            audio_features = self.detect_audio_features(normalized_audio)
            
            # Quick speaker change detection based on tone changes (more conservative)
            if audio_features['tone_change'] and not audio_features['is_silence']:
                current_time = time.time()
                # Require longer duration before switching AND significant energy change
                if (current_time - self.last_speaker_change_time > self.min_speaker_duration and
                    audio_features['energy'] > 0.1):  # Ensure it's actual speech, not noise
                    
                    # Additional check: only switch if energy change is substantial
                    if (self.last_audio_features and 
                        abs(audio_features['energy'] - self.last_audio_features['energy']) > 0.3):
                        # Switch speaker based on audio characteristics
                        self.current_speaker = "Patient" if self.current_speaker == "Doctor" else "Doctor"
                        self.last_speaker_change_time = current_time
            
            self.audio_buffer_for_diarization.append(normalized_audio)  # Store normalized audio
                
        return (in_data, pyaudio.paContinue)
    
    def connect_to_server(self):
        """Connect to the Whisper server with retry logic"""
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                if self.socket:
                    self.socket.close()
                    
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.settimeout(10.0)  # 10 second connection timeout
                self.socket.connect((HOST, PORT))
                self.socket.settimeout(None)  # Remove timeout after connection
                print(f" Connected to Whisper server at {HOST}:{PORT}")
                return True
                
            except ConnectionRefusedError:
                print(f" Connection refused (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    print(f" Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    
            except socket.timeout:
                print(f" Connection timeout (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    print(f" Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    
            except Exception as e:
                print(f" Failed to connect to server: {e} (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    print(f" Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
        
        print(" Failed to connect after all attempts")
        return False
    
    def send_audio_worker(self):
        """Worker thread to send audio data to server with improved load balancing"""
        audio_buffer = []
        last_send_time = time.time()
        send_interval = 0.4  # Slightly slower for better stability
        max_buffer_size = 1  # Only 1 chunk at a time initially
        consecutive_timeouts = 0
        max_consecutive_timeouts = 2  # Faster adaptation
        adaptive_delay = 0.08  # Start with larger delay for stability
        
        while self.is_recording:
            try:
                # Get audio data from queue with adaptive timeout
                audio_data = self.audio_queue.get(timeout=adaptive_delay)
                audio_buffer.append(audio_data)
                
                current_time = time.time()
                
                # Adaptive sending based on server load
                should_send = (len(audio_buffer) >= max_buffer_size or 
                              (current_time - last_send_time >= send_interval and len(audio_buffer) > 0))
                
                if should_send:
                    # Check server overload and adapt more aggressively
                    if consecutive_timeouts >= max_consecutive_timeouts:
                        # Server is overloaded - more aggressive adaptation
                        adaptive_delay = min(2.0, adaptive_delay * 1.5)  # Larger increases
                        send_interval = min(2.0, send_interval * 1.3)   # Larger increases
                        max_buffer_size = 1  # Keep minimal chunks
                        
                        # Drop more audio to reduce backlog
                        if len(audio_buffer) > 1:
                            audio_buffer = audio_buffer[-1:]  # Keep only latest chunk
                        
                        # Reduce frequency of overload messages - only show every 10th time
                        if not hasattr(self, 'overload_message_count'):
                            self.overload_message_count = 0
                        self.overload_message_count += 1
                        if self.overload_message_count % 10 == 1:  # Show message every 10th overload
                            print(f"\n Server overloaded ({self.overload_message_count}), adapting...", end='')
                        consecutive_timeouts = 0
                        continue
                    
                    combined_audio = np.concatenate(audio_buffer)
                    
                    # Send audio to server with reasonable timeout
                    if self.socket:
                        try:
                            self.socket.settimeout(0.5)  # Slightly longer timeout
                            self.socket.sendall(combined_audio.tobytes())
                            last_send_time = current_time
                            consecutive_timeouts = 0  # Reset on success
                            
                            # Success - gradually improve performance
                            if adaptive_delay > 0.02:
                                adaptive_delay = max(0.02, adaptive_delay * 0.98)
                            if send_interval > 0.2:
                                send_interval = max(0.2, send_interval * 0.99)
                            
                            # Reduced frequency status indicators - only every 5th success
                            if not hasattr(self, 'success_count'):
                                self.success_count = 0
                            self.success_count += 1
                            if self.success_count % 5 == 0:  # Show dot every 5th success
                                print(".", end='', flush=True)
                            
                        except socket.timeout:
                            consecutive_timeouts += 1
                            if consecutive_timeouts % 3 == 1:  # Show timeout every 3rd occurrence
                                print(f"T", end='', flush=True)
                            
                        except ConnectionResetError:
                            print(f"\n Connection lost - attempting to reconnect...")
                            consecutive_timeouts = 0
                            if self.connect_to_server():
                                print("✓ Reconnected")
                        except Exception as e:
                            consecutive_timeouts += 1
                            if consecutive_timeouts % 2 == 1:  # Show error every 2nd occurrence
                                print(f"E", end='', flush=True)
                        finally:
                            self.socket.settimeout(None)
                    
                    # Always clear buffer
                    audio_buffer = []
                    
            except queue.Empty:
                # No audio available - this is normal
                continue
            except Exception as e:
                print(f"\nSend worker error: {e}")
                break
    
    def receive_transcription_worker(self):
        """Worker thread to receive transcription from server with preserved quality"""
        consecutive_timeouts = 0
        max_consecutive_timeouts = 15  # More tolerance for timeouts
        sentence_buffer = ""  # Buffer to accumulate partial sentences
        last_transcription_time = time.time()
        
        while self.is_recording:
            try:
                if self.socket:
                    self.socket.settimeout(0.05)  # Slightly longer timeout for stability
                    response = self.socket.recv(512)  # Larger buffer for better performance
                    if response:
                        consecutive_timeouts = 0  # Reset timeout counter
                        transcription = response.decode().strip()
                        if transcription:
                            # Parse transcription (format: "timestamp_start timestamp_end text")
                            parts = transcription.split(' ', 2)
                            if len(parts) >= 3:
                                try:
                                    start_time = float(parts[0]) / 1000.0  # Convert ms to seconds
                                    end_time = float(parts[1]) / 1000.0
                                    text = parts[2].strip()
                                    
                                    # Preserve original text quality - minimal processing
                                    if text:
                                        # Accumulate text into sentence buffer (preserve spacing)
                                        if sentence_buffer:
                                            sentence_buffer += " " + text
                                        else:
                                            sentence_buffer = text
                                        
                                        current_time = time.time()
                                        time_since_last = current_time - last_transcription_time
                                        
                                        # Check for natural sentence completion
                                        sentence_ends = ['.', '!', '?']
                                        has_sentence_end = any(sentence_buffer.strip().endswith(end) for end in sentence_ends)
                                        
                                        # Send complete sentences or timeout-based chunks (preserve quality)
                                        if (has_sentence_end or 
                                            time_since_last > 1.2 or  # Shorter delay for more responsive output
                                            len(sentence_buffer.split()) > 8):  # Smaller chunks
                                            
                                            if sentence_buffer.strip():
                                                # Minimal cleanup - preserve original transcription
                                                clean_sentence = sentence_buffer.strip()
                                                
                                                # Add to transcription queue for speaker assignment
                                                self.transcription_queue.put({
                                                    'start': start_time,
                                                    'end': end_time,
                                                    'text': clean_sentence,  # Preserve original quality
                                                    'timestamp': current_time
                                                })
                                                
                                                sentence_buffer = ""  # Reset buffer
                                                last_transcription_time = current_time
                                    
                                except ValueError:
                                    # If parsing fails, preserve the original text
                                    if transcription.strip():
                                        sentence_buffer = transcription if not sentence_buffer else sentence_buffer + " " + transcription
                            else:
                                # Add unparsed text to buffer
                                sentence_buffer += " " + transcription if sentence_buffer else transcription
                    else:
                        # Empty response - increment timeout counter
                        consecutive_timeouts += 1
                        if consecutive_timeouts >= max_consecutive_timeouts:
                            # Send any buffered content
                            if sentence_buffer.strip():
                                current_time = time.time()
                                self.transcription_queue.put({
                                    'start': current_time - 0.5,
                                    'end': current_time,
                                    'text': sentence_buffer.strip(),
                                    'timestamp': current_time
                                })
                                sentence_buffer = ""
                            consecutive_timeouts = 0
                            
            except socket.timeout:
                consecutive_timeouts += 1
                # Send buffered content on timeout less frequently
                if consecutive_timeouts >= max_consecutive_timeouts and sentence_buffer.strip():
                    current_time = time.time()
                    self.transcription_queue.put({
                        'start': current_time - 0.5,
                        'end': current_time,
                        'text': sentence_buffer.strip(),
                        'timestamp': current_time
                    })
                    sentence_buffer = ""
                    consecutive_timeouts = 0
                continue
            except ConnectionResetError:
                if self.is_recording:
                    print(f"\n Server connection lost - attempting to reconnect...")
                    if self.connect_to_server():
                        print(" Reconnected successfully")
                    else:
                        print(" Reconnection failed")
                break
            except Exception as e:
                if self.is_recording:  # Only print error if we're still recording
                    print(f"\n Receive error (non-fatal): {e}")
                    time.sleep(0.1)  # Brief pause on error
                continue
    
    def diarization_worker(self):
        """Quick audio-based diarization worker focused on tone and silence detection"""
        print(" Starting three-stage diarization worker...")
        print(f" Processing on: {self.diarization_processor.device}")
        print("  Stage 1: Voice Activity Detection & Segmentation")
        print("  Stage 2: Neural Speaker Embeddings") 
        print("  Stage 3: Agglomerative Clustering")

        last_process_time = time.time()
        processing_stats = {'success': 0, 'timeout': 0, 'error': 0}
        
        while self.is_recording:
            try:
                current_time = time.time()
                
                # Quick processing timing: every 0.8 seconds for faster response
                if current_time - last_process_time >= 0.8 and len(self.audio_buffer_for_diarization) > 0:
                    
                    # Get recent audio data for quick analysis
                    with self.speaker_lock:
                        # Use shorter audio segments (2 seconds) for quicker processing
                        max_samples = int(SAMPLING_RATE * 2.0)  # Shorter for quick response
                        if len(self.audio_buffer_for_diarization) > 0:
                            try:
                                # Safe concatenation with shape validation
                                valid_buffers = []
                                for buf in self.audio_buffer_for_diarization:
                                    if isinstance(buf, np.ndarray) and buf.size > 0:
                                        # Ensure 1D array
                                        buf_flat = buf.flatten()
                                        if buf_flat.size > 0:
                                            valid_buffers.append(buf_flat)
                                
                                if valid_buffers:
                                    audio_segment = np.concatenate(valid_buffers)
                                    
                                    # Keep recent audio for quick analysis
                                    if len(audio_segment) > max_samples:
                                        audio_segment = audio_segment[-max_samples:]
                                    
                                    segment_start_time = current_time - (len(audio_segment) / SAMPLING_RATE)
                                else:
                                    audio_segment = np.array([])  # Empty array
                                    segment_start_time = current_time
                                    
                            except Exception as concat_error:
                                print(f" Buffer concatenation error: {concat_error}")
                                audio_segment = np.array([])  # Empty array fallback
                                segment_start_time = current_time
                            
                            # Quick buffer management - safe slicing
                            try:
                                if len(self.audio_buffer_for_diarization) > 8:
                                    self.audio_buffer_for_diarization = self.audio_buffer_for_diarization[-8:]
                            except Exception as slice_error:
                                print(f" Buffer slicing error: {slice_error}")
                                self.audio_buffer_for_diarization = []  # Reset buffer
                        else:
                            audio_segment = np.array([])  # Empty array
                            segment_start_time = current_time
                    
                    # Process if we have minimum audio (0.8 seconds for quick processing)
                    if len(audio_segment) >= SAMPLING_RATE * 0.8:
                        try:
                            # Use quick acoustic analysis for speaker detection
                            if self.diarization_processor.use_acoustic_diarization and self.diarization_processor.acoustic_processor:
                                print("", end='', flush=True)  # Quick indicator
                                
                                # Quick processing with performance monitoring
                                start_processing = time.time()
                                segments = self.diarization_processor.process_acoustic_fast(audio_segment, segment_start_time)
                                processing_time = time.time() - start_processing
                                
                                if segments:
                                    processing_stats['success'] += 1
                                    
                                    # Advanced speaker change detection with confidence weighting
                                    latest_segment = max(segments, key=lambda x: x['end'])
                                    avg_confidence = np.mean([s['confidence'] for s in segments])
                                    
                                    # Only change speaker if confidence is high enough
                                    if (latest_segment['speaker'] != self.current_speaker and 
                                        avg_confidence >= self.diarization_processor.speaker_switch_threshold):
                                        
                                        self.current_speaker = latest_segment['speaker']
                                        self.last_speaker_change_time = current_time
                                        self.conversation_turn_count += 1
                                        
                                        confidence_indicator = "●●●" if avg_confidence > 0.9 else "●●"
                                        print(f" -> {self.current_speaker} {confidence_indicator} ({processing_time:.2f}s)")
                                    else:
                                        print(f" -> same speaker ● ({processing_time:.2f}s)")
                                else:
                                    print(" -> no segments detected")
                                
                                # Adaptive processing interval based on performance
                                if processing_time > 1.0:
                                    # If processing is slow, increase interval
                                    last_process_time = current_time + 0.5
                                else:
                                    last_process_time = current_time
                                
                        except Exception as e:
                            processing_stats['error'] += 1
                            print(f" Advanced segmentation failed: {e}")
                            # Increase interval on error to prevent spam
                            last_process_time = current_time + 1.0
                    
                    else:
                        last_process_time = current_time
                
                # Show processing statistics every 30 seconds
                if current_time % 30 < 0.3:  # Approximate every 30 seconds
                    total_attempts = sum(processing_stats.values())
                    if total_attempts > 0:
                        success_rate = (processing_stats['success'] / total_attempts) * 100
                        print(f"\n Diarization stats: {success_rate:.1f}% success rate")
                
                time.sleep(0.2)  # Optimized check interval
                
            except Exception as e:
                print(f"Advanced diarization error: {e}")
                time.sleep(1.0)
    
    def clean_transcription_text(self, text):
        """Minimal transcription cleaning - preserve original quality"""
        if not text:
            return text
        
        # Only do very minimal, safe cleaning
        import re
        
        # Remove only obvious artifacts without corrupting words
        text = re.sub(r'\bTT\b', '', text)  # Remove "TT" fragments
        text = re.sub(r'\s+', ' ', text)    # Fix multiple spaces
        text = re.sub(r'([.!?])\1+', r'\1', text)  # Fix double punctuation like "??"
        
        # Only fix very common and safe transcription errors
        safe_word_fixes = {
            'okayy': 'okay',
            'yeeah': 'yeah', 
            'wwell': 'well',
            'thee': 'the',
            'tto': 'to',
            'aand': 'and',
            'oof': 'of',
            'inn': 'in',
            'iss': 'is',
            'att': 'at'
        }
        
        # Apply only safe word fixes (very conservative)
        words = text.split()
        cleaned_words = []
        for word in words:
            # Remove punctuation for matching but be very conservative
            clean_word = word.lower().strip('.,!?;:')
            if clean_word in safe_word_fixes and len(clean_word) <= 4:  # Only fix very short words
                # Preserve original punctuation
                punctuation = ''.join(c for c in word if c in '.,!?;:')
                cleaned_words.append(safe_word_fixes[clean_word] + punctuation)
            else:
                cleaned_words.append(word)  # Keep original word
        
        text = ' '.join(cleaned_words)
        
        # Only fix very obvious phrase errors
        safe_phrase_fixes = {
            'check that book': 'check up',
            'stress pain': 'chest pain'
        }
        
        for wrong_phrase, correct_phrase in safe_phrase_fixes.items():
            text = text.replace(wrong_phrase, correct_phrase)
        
        # Basic capitalization - only first letter of first word
        text = text.strip()
        if text and len(text) > 1:
            text = text[0].upper() + text[1:]
        
        return text

    def transcription_assignment_worker(self):
        """Enhanced transcription assignment with preserved text quality"""
        while self.is_recording:
            try:
                # Get transcription from queue
                transcription = self.transcription_queue.get(timeout=0.1)
                
                # Quick processing - minimal buffering
                current_time = time.time()
                text = transcription['text'].strip()
                
                # Minimal cleaning - preserve original transcription quality
                text = self.clean_transcription_text(text)
                # PHI scrubbing for sensitive medical data
                text = self.scrub_phi(text)
                
                # Language detection and translation
                original_text = text
                translated_text, detected_language, was_translated = self.detect_and_translate_text(text)
                
                # Use translated text for processing if translation occurred
                if was_translated:
                    text = translated_text
                    language_name = LANGUAGES.get(detected_language, detected_language).title()
                else:
                    language_name = 'English'
                
                # Skip very short transcriptions EXCEPT for common short responses
                word_count = len(text.split())
                short_responses = ['no', 'yes', 'yeah', 'okay', 'ok', 'sure', 'maybe', 'right', 'good', 'bad', 'well', 'great', 'fine', 'none', 'nothing', 'never', 'always', 'sometimes']
                # Also detect phrases that are clearly responses
                response_phrases = ['no rashes', 'up to date', 'not really', 'i think', 'i feel', 'i have', 'i dont', 'i do', 'i am', 'im not', 'thats right', 'thats correct']
                
                is_short_response = (
                    any(word.lower() in short_responses for word in text.lower().split()[:3]) or  # Check first 3 words
                    any(phrase in text.lower() for phrase in response_phrases) or
                    (word_count <= 3 and any(word.lower() in ['no', 'yes', 'yeah', 'ok', 'okay'] for word in text.split()))
                )
                
                if word_count < 2 and not is_short_response:  # Allow short responses to pass through
                    continue
                
                # Enhanced speaker assignment using ensemble of classifiers
                time_since_last_change = current_time - self.last_speaker_change_time
                
                # Use ensemble classifier for more accurate speaker assignment
                ensemble_speaker, ensemble_confidence, ensemble_method = self.diarization_processor.classify_transcription_with_ensemble(
                    text, self.current_speaker
                )
                
                # Check if current speaker has incomplete sentence that needs to be finished
                current_speaker_has_incomplete_sentence = False
                if self.current_speaker in self.pending_fragments and self.pending_fragments[self.current_speaker]:
                    incomplete_text = self.pending_fragments[self.current_speaker].strip()
                    
                    # Enhanced incomplete sentence detection
                    if len(incomplete_text) > 0:
                        # Consider sentence incomplete if it doesn't end with proper punctuation
                        lacks_punctuation = not incomplete_text.endswith(('.', '!', '?', ':', ';'))
                        
                        # Check for clearly incomplete phrases
                        incomplete_phrases = [
                            'makes the', 'besides i', 'and i', 'but i', 'so i', 'then i',
                            'worse besides', 'pain besides', 'anything else that', 'there is',
                            'i am', 'i have', 'i feel', 'i think', 'i was', 'i will', 'i would',
                            'it is', 'it was', 'that is', 'this is', 'when i', 'where i'
                        ]
                        
                        text_lower = incomplete_text.lower()
                        has_incomplete_phrase = any(phrase in text_lower for phrase in incomplete_phrases)
                        
                        # Check if it ends with words that typically continue
                        continuation_words = ['and', 'or', 'but', 'so', 'because', 'besides', 'when', 'where', 'that', 'the', 'a', 'an']
                        ends_with_continuation = any(text_lower.endswith(' ' + word) for word in continuation_words)
                        
                        # Consider incomplete if lacks punctuation AND has substantive content AND suggests continuation
                        current_speaker_has_incomplete_sentence = (
                            lacks_punctuation and 
                            len(incomplete_text.split()) > 1 and  # More than one word
                            (has_incomplete_phrase or ends_with_continuation or len(incomplete_text.split()) > 3)
                        )
                
                # Check if current speaker is Doctor and has a complete question
                doctor_completed_question = False
                if (self.current_speaker == "Doctor" and 
                    self.current_speaker in self.pending_fragments and 
                    self.pending_fragments[self.current_speaker]):
                    
                    doctor_text = self.pending_fragments[self.current_speaker].strip()
                    # Check if doctor's accumulated text ends with a question
                    if doctor_text.endswith('?') and len(doctor_text.split()) > 2:
                        doctor_completed_question = True
                
                # Check if this new text fragment contains a question completion
                text_completes_question = text.strip().endswith('?') and len(text.split()) > 1
                
                # Combine ensemble prediction with audio-based detection for robustness
                if time_since_last_change < 1.5 and ensemble_confidence < 0.85:
                    # Recent audio detection + low text confidence = trust audio
                    assigned_speaker = self.current_speaker
                    method = f"audio(conf:{ensemble_confidence:.2f})"
                elif doctor_completed_question or (self.current_speaker == "Doctor" and text_completes_question):
                    # Doctor just completed a question - prepare to switch to Patient for response
                    assigned_speaker = self.current_speaker  # Keep doctor for this question completion
                    method = f"doctor-question-complete"
                    # Set flag to switch to patient for next transcription
                    if not hasattr(self, 'expect_patient_response'):
                        self.expect_patient_response = False
                    self.expect_patient_response = True
                elif hasattr(self, 'expect_patient_response') and self.expect_patient_response and not text_completes_question:
                    # Doctor completed question, now switch to patient for response
                    assigned_speaker = "Patient"
                    method = f"question-response-switch"
                    self.current_speaker = "Patient"
                    self.last_speaker_change_time = current_time
                    self.expect_patient_response = False  # Reset flag
                elif ensemble_confidence > 0.90 and not current_speaker_has_incomplete_sentence:
                    # High confidence from ensemble = trust text classification, BUT only if current sentence is complete
                    assigned_speaker = ensemble_speaker
                    method = f"{ensemble_method}"
                    
                    # Update current speaker if ensemble is very confident AND no incomplete sentence
                    if ensemble_confidence > 0.95 and assigned_speaker != self.current_speaker:
                        self.current_speaker = assigned_speaker
                        self.last_speaker_change_time = current_time
                elif ensemble_confidence > 0.90 and current_speaker_has_incomplete_sentence:
                    # High confidence but current speaker has incomplete sentence - wait for completion
                    assigned_speaker = self.current_speaker  # Keep current speaker until sentence completes
                    method = f"{ensemble_method}⏳wait-for-completion"
                elif is_short_response and not current_speaker_has_incomplete_sentence:
                    # For short responses, use ensemble but validate with simple patterns, only if no incomplete sentence
                    if text.lower().strip().startswith(('yes', 'no', 'yeah', 'okay', 'sure')):
                        assigned_speaker = "Patient"  # Responses are typically patient
                    elif text.lower().endswith('?'):
                        assigned_speaker = "Doctor"   # Questions are typically doctor
                    else:
                        assigned_speaker = ensemble_speaker
                    
                    if assigned_speaker != self.current_speaker:
                        self.current_speaker = assigned_speaker
                        self.last_speaker_change_time = current_time
                    method = f"short({ensemble_confidence:.2f})"
                elif is_short_response and current_speaker_has_incomplete_sentence:
                    # Short response but incomplete sentence - keep current speaker
                    assigned_speaker = self.current_speaker
                    method = f"short({ensemble_confidence:.2f})⏳wait"
                else:
                    # Medium confidence - use ensemble but with validation, respect incomplete sentences
                    if current_speaker_has_incomplete_sentence:
                        assigned_speaker = self.current_speaker
                        method = f"{ensemble_method}⏳incomplete"
                    else:
                        assigned_speaker = ensemble_speaker
                        method = f"{ensemble_method}"
                
                self.last_transcription_time = current_time
                
                # Quick output - preserve original text quality
                if text:
                    # Add to current sentence for this speaker (both translated and original)
                    if self.pending_fragments[assigned_speaker]:
                        self.pending_fragments[assigned_speaker] += " " + text
                        if was_translated:
                            self.pending_original_fragments[assigned_speaker] += " " + original_text
                    else:
                        self.pending_fragments[assigned_speaker] = text
                        if was_translated:
                            self.pending_original_fragments[assigned_speaker] = original_text
                    
                    # Track translation info for this speaker
                    if was_translated:
                        self.pending_translation_info[assigned_speaker]['was_translated'] = True
                        self.pending_translation_info[assigned_speaker]['language'] = language_name
                    
                    # Output immediately if sentence seems complete or after delay
                    should_output = False
                    
                    # Check if current accumulated text appears complete
                    accumulated_text = self.pending_fragments[assigned_speaker].strip()
                    
                    # Enhanced sentence completion detection
                    def is_sentence_complete(text):
                        """Check if text represents a complete sentence or thought"""
                        if not text:
                            return False
                        
                        text = text.strip()
                        
                        # Check for proper sentence endings - questions are always complete
                        if text.endswith(('.', '!', '?')):
                            return True
                        
                        # Check if it ends with incomplete punctuation that suggests more is coming
                        if text.endswith((',', ';', ':', 'and', 'or', 'but', 'so', 'because', 'besides')):
                            return False
                        
                        # Check for incomplete phrases that clearly need continuation
                        incomplete_endings = [
                            'i am', 'i have', 'i feel', 'i think', 'i was', 'i will', 'i would',
                            'there is', 'there are', 'it is', 'it was', 'that is', 'this is',
                            'makes the', 'besides i', 'and i', 'but i', 'so i',
                            'worse besides', 'pain besides', 'anything else that'
                        ]
                        
                        text_lower = text.lower()
                        for ending in incomplete_endings:
                            if text_lower.endswith(ending):
                                return False
                        
                        # Special handling for doctor questions - if it contains question words, likely complete
                        question_indicators = ['what', 'when', 'where', 'why', 'how', 'do you', 'are you', 'can you', 'have you', 'did you']
                        if any(indicator in text_lower for indicator in question_indicators) and len(text.split()) >= 3:
                            return True
                        
                        # If text doesn't end with proper punctuation but seems like a complete thought
                        words = text.split()
                        if len(words) >= 3:
                            # Common complete response patterns
                            complete_patterns = ['yes', 'no', 'okay', 'sure', 'right', 'exactly', 'correct', 'wrong', 'never', 'always', 'sometimes']
                            if any(pattern in text_lower for pattern in complete_patterns):
                                return True
                        
                        return False
                    
                    # Output logic - prioritize natural sentence completion and question completion
                    if is_sentence_complete(accumulated_text) and len(accumulated_text.split()) > 1:
                        should_output = True
                    
                    # Force output for completed doctor questions to enable immediate patient response
                    elif (assigned_speaker == "Doctor" and accumulated_text.endswith('?') and 
                          len(accumulated_text.split()) > 2):
                        should_output = True
                    
                    # Special case for clear short responses (but still check completeness)
                    elif is_short_response and any(word in accumulated_text.lower() for word in ['yes', 'no', 'okay', 'sure', 'right']):
                        if is_sentence_complete(accumulated_text) or len(accumulated_text.split()) <= 3:  # Allow short responses
                            should_output = True
                    
                    # Only output after delay if sentence appears complete OR delay is very long
                    elif (assigned_speaker in self.last_output_time):
                        delay = 1.5 if is_short_response else 4.0  # Increased delay for better completion
                        time_since_last = current_time - self.last_output_time[assigned_speaker]
                        
                        if time_since_last > delay:
                            # Check if sentence seems complete before outputting on timeout
                            if is_sentence_complete(accumulated_text):
                                should_output = True
                            elif time_since_last > delay * 2:  # Very long delay - force output to prevent hanging
                                should_output = True
                    
                    # Or if this is the first output for this speaker (but still check completeness)
                    elif assigned_speaker not in self.last_output_time:
                        if is_sentence_complete(accumulated_text) or len(accumulated_text.split()) <= 2:
                            should_output = True
                    
                    # Output the sentence with preserved quality
                    if should_output:
                        full_text = self.pending_fragments[assigned_speaker].strip()
                        original_full_text = self.pending_original_fragments[assigned_speaker].strip()
                        translation_info = self.pending_translation_info[assigned_speaker]
                        
                        # Clear buffers
                        self.pending_fragments[assigned_speaker] = ""
                        self.pending_original_fragments[assigned_speaker] = ""
                        self.pending_translation_info[assigned_speaker] = {'was_translated': False, 'language': 'English'}
                        self.last_output_time[assigned_speaker] = current_time
                        
                        # Clean output formatting - show translation info if applicable
                        if translation_info['was_translated']:
                            print(f"\n{assigned_speaker}: {full_text}", end="")
                            print(f" | Translated from {translation_info['language']}: \"{original_full_text}\"", end="")
                        else:
                            print(f"\n{assigned_speaker}: {full_text}", end="")
                        
                        # Add treatment phase detection for all segments
                        if self.diarization_processor.phase_classifier.is_loaded:
                            try:
                                phase, phase_conf = self.diarization_processor.phase_classifier.predict_phase(full_text)
                                if phase != "unknown" and phase != "error" and phase_conf > 0.5:
                                    print(f" | Phase: {phase} ({phase_conf:.2f})", end="")
                            except:
                                pass  # Skip phase detection errors
                        
                        # Trigger sentiment analysis for Patient segments only
                        if assigned_speaker == "Patient" and self.sentiment_enabled:
                            # Get immediate sentiment for inline display
                            try:
                                # Record processing start time for latency measurement
                                if hasattr(self, 'processing_start_times'):
                                    self.processing_start_times['sentiment'] = time.time()
                                
                                # Use available sentiment analyzer
                                if hasattr(self, 'healthcare_sentiment') and self.healthcare_sentiment:
                                    try:
                                        sentiment_result = self.healthcare_sentiment.analyze_text(
                                            full_text, patient_id=f"patient_{current_time}", 
                                            include_individual_models=False
                                        )
                                        # Extract from healthcare analyzer result format
                                        sentiment_label = sentiment_result.get('ensemble_prediction', 'Neutral')
                                        sentiment_confidence = sentiment_result.get('ensemble_confidence', 0.5)
                                        
                                        # Map healthcare analyzer labels to numerical scores (-3 to +3 scale)
                                        if sentiment_label in ['Negative', 'NEGATIVE', 'negative']:
                                            sentiment_score = -1.5 * sentiment_confidence  # Scale by confidence
                                        elif sentiment_label in ['Positive', 'POSITIVE', 'positive']:
                                            sentiment_score = 1.5 * sentiment_confidence   # Scale by confidence  
                                        else:  # Neutral
                                            sentiment_score = 0.0
                                            
                                        # Ensure score is in valid range (-3 to +3)
                                        sentiment_score = max(-3.0, min(3.0, sentiment_score))
                                            
                                    except Exception as healthcare_error:
                                        # Fallback if healthcare sentiment analyzer fails
                                        print(f" | Healthcare analyzer error", end="")
                                        sentiment_result = self.sentiment_analyzer.analyze_patient_sentiment(full_text)
                                        final_result = sentiment_result.get('final', {})
                                        sentiment_score = final_result.get('numerical_score', 0)
                                        sentiment_label = final_result.get('sentiment', 'Neutral')
                                        sentiment_confidence = final_result.get('confidence', 0.5)
                                else:
                                    # Fallback to built-in sentiment analyzer
                                    try:
                                        sentiment_result = self.sentiment_analyzer.analyze_patient_sentiment(full_text)
                                        # Extract numerical score from the result
                                        final_result = sentiment_result.get('final', {})
                                        sentiment_score = final_result.get('numerical_score', 0)
                                        sentiment_label = final_result.get('sentiment', 'Neutral')
                                        sentiment_confidence = final_result.get('confidence', 0.5)
                                        
                                        # Convert label format and assign proper numerical scores if needed
                                        if sentiment_label == 'LABEL_0':
                                            sentiment_label = 'Negative'
                                            sentiment_score = -1.5 if sentiment_score == 0 else sentiment_score
                                        elif sentiment_label == 'LABEL_1':
                                            sentiment_label = 'Neutral'
                                            sentiment_score = 0.0 if sentiment_score == 0 else sentiment_score
                                        elif sentiment_label == 'LABEL_2':
                                            sentiment_label = 'Positive'
                                            sentiment_score = 1.5 if sentiment_score == 0 else sentiment_score
                                        
                                        # Ensure score is in valid range (-3 to +3)
                                        sentiment_score = max(-3.0, min(3.0, sentiment_score))
                                        
                                    except Exception as fallback_error:
                                        # Last resort - assign neutral sentiment
                                        print(f" | Fallback error", end="")
                                        sentiment_score = 0.0
                                        sentiment_label = 'Neutral'
                                        sentiment_confidence = 0.5
                                
                                # Get treatment phase for this segment
                                phase = 'during'  # default
                                if hasattr(self.diarization_processor, 'phase_classifier') and self.diarization_processor.phase_classifier.is_loaded:
                                    try:
                                        phase_result, phase_conf = self.diarization_processor.phase_classifier.predict_phase(full_text)
                                        if phase_result != "unknown" and phase_result != "error" and phase_conf > 0.5:
                                            phase = phase_result
                                    except:
                                        pass
                                
                                # Store patient sentiment for satisfaction calculation
                                self.patient_sentiments.append({
                                    'text': full_text,
                                    'score': sentiment_score,
                                    'label': sentiment_label,
                                    'confidence': sentiment_confidence,
                                    'phase': phase,
                                    'timestamp': current_time
                                })
                                
                                # Format sentiment display with color and emoji
                                if sentiment_score > 0.1:
                                    sentiment_color = "\033[32m"  # Green for positive
                                    sentiment_symbol = ""
                                elif sentiment_score < -0.1:
                                    sentiment_color = "\033[31m"  # Red for negative  
                                    sentiment_symbol = ""
                                else:
                                    sentiment_color = "\033[33m"  # Yellow for neutral
                                    sentiment_symbol = ""
                                
                                sentiment_display = f" | {sentiment_symbol}{sentiment_color}Sentiment: {sentiment_label} ({sentiment_score:+.2f})\033[0m"
                                print(sentiment_display, end="")
                                
                                # Calculate processing latency
                                if 'sentiment' in self.processing_start_times:
                                    latency_ms = (time.time() - self.processing_start_times['sentiment']) * 1000
                                    self.latency_measurements.append(latency_ms)
                                    # Keep only last 100 measurements for rolling average
                                    if len(self.latency_measurements) > 100:
                                        self.latency_measurements.pop(0)
                                
                            except Exception as e:
                                print(f" | Sentiment: Error ({str(e)})", end="")
                            
                            self.patient_segments_count += 1
                        else:
                            # Just add newline for Doctor segments  
                            print()
                        
                        # Track diarization segments for DER calculation
                        self.total_diarization_segments += 1
                        
                        # Show simplified metrics (confidence and method only during conversation)
                        metrics_line = f"   Confidence: {ensemble_confidence:.2f} | Method: {method}"
                        print(metrics_line)
                        
                        self.total_segments_count += 1
                        
                        # Update conversation tracking
                        self.conversation_turn_count += 1
                        
                        # Track in speaker history
                        self.speaker_history[assigned_speaker].append({
                            'text': full_text,
                            'timestamp': current_time,
                            'confidence': ensemble_confidence,
                            'method': method
                        })
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f" Transcription assignment error: {e}")
    
    def sentiment_analysis_worker(self):
        """Sentiment analysis now handled inline - this worker is disabled"""
        pass
    
    def display_sentiment_results(self, sentiment_results, segment_data):
        """Display simplified sentiment analysis results without emojis"""
        try:
            text = sentiment_results.get('text', '')
            final_result = sentiment_results.get('final', {})
            
            # Simple display format
            if final_result:
                sentiment = final_result.get('sentiment', 'Unknown')
                confidence = final_result.get('confidence', 0.0)
                
                # Convert model labels to readable format
                if sentiment == 'LABEL_0':
                    sentiment = 'Negative'
                elif sentiment == 'LABEL_1':
                    sentiment = 'Neutral'  
                elif sentiment == 'LABEL_2':
                    sentiment = 'Positive'
                
                print(f" | Sentiment: {sentiment} ({confidence:.2f})")
            else:
                print(f" | Sentiment: Error in analysis")
            
        except Exception as e:
            print(f" | Sentiment: Analysis failed")
    
    def get_current_speaker_from_audio(self, current_time):
        """Get current speaker from recent audio analysis"""
        with self.speaker_lock:
            if not self.speaker_history:
                return None
                
            # Get the most recent diarization result
            most_recent_time = max(self.speaker_history.keys()) if self.speaker_history else 0
            if most_recent_time == 0:
                return None
                
            recent_segments = self.speaker_history[most_recent_time]
            
            if recent_segments:
                # Return the speaker from the most recent segment
                return recent_segments[-1]['speaker']
        
        return None
    
    def start_streaming(self):
        """Start live microphone streaming with diarization"""
        # Connect to server
        if not self.connect_to_server():
            return
        
        p = None
        try:
            # Initialize PyAudio
            p = pyaudio.PyAudio()
            # List available audio devices
            print("\nAvailable audio devices:")
            input_devices = []
            for i in range(p.get_device_count()):
                info = p.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    input_devices.append((i, info['name']))
                    print(f"  {i}: {info['name']} (inputs: {info['maxInputChannels']})")
            # Let user choose device or use default
            print(f"\nPress Enter to use default device, or type device number (0-{len(input_devices)-1}):")
            choice = input().strip()
            if choice.isdigit() and 0 <= int(choice) < p.get_device_count():
                device_index = int(choice)
                device_info = p.get_device_info_by_index(device_index)
                print(f"Using device: {device_info['name']}")
            else:
                # Try to find the default input device
                default_device = p.get_default_input_device_info()
                device_index = default_device['index']
                print(f"Using default device: {default_device['name']}")

            # Open audio stream
            stream = p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=SAMPLING_RATE,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=CHUNK_SIZE,
                stream_callback=self.audio_callback
            )

            print("\n Starting live transcription with three-stage diarization...")
            print(" Speak into your microphone!")
            print(" Output format: SPEAKER_X: sentences with timestamps")
            print(" Stage 1: Voice Activity Detection & Segmentation")
            print(" Stage 2: Neural Speaker Embeddings (1.5s windows)")
            print(" Stage 3: Agglomerative Clustering (2 speakers)")
            print(" Stage 4: Patient Sentiment Analysis (parallel)")
            if self.translation_enabled:
                print("Stage 5: Auto-Translation (non-English → English)")
            print(" Features: Overlap detection, RTTM format support")
            print("  Doctor (blue) |  Patient (green) | OVERLAP (red)")
            print(" Status:=segment, =embed, =cluster, =fallback")
            if self.translation_enabled:
                print(" Translation: Auto-detects language and translates to English")
            print(" Sentiment: Only patient segments analyzed in real-time")
            print("Press Ctrl+C to stop\n")

            # Start recording
            self.is_recording = True
            self.sentiment_running = True
            self.buffer_start_time = time.time()
            stream.start_stream()

            # Start worker threads
            send_thread = threading.Thread(target=self.send_audio_worker, daemon=True)
            receive_thread = threading.Thread(target=self.receive_transcription_worker, daemon=True)
            diarization_thread = threading.Thread(target=self.diarization_worker, daemon=True)
            assignment_thread = threading.Thread(target=self.transcription_assignment_worker, daemon=True)
            sentiment_thread = threading.Thread(target=self.sentiment_analysis_worker, daemon=True)

            send_thread.start()
            receive_thread.start()
            diarization_thread.start()
            assignment_thread.start()
            sentiment_thread.start()

            print(" All worker threads started:")
            print("    Audio streaming")
            print("    Transcription receiving")
            print("    Speaker diarization")
            print("   Transcription assignment")
            print("    Patient sentiment analysis")
            print("")

            # Keep main thread alive
            try:
                while stream.is_active() and self.is_recording:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("\n Stopping transcription...")
                print(f" Enhanced Session Summary:")
                print(f"   Total segments: {self.total_segments_count}")
                print(f"    Patient segments analyzed: {self.patient_segments_count}")
                if self.patient_segments_count > 0:
                    print(f"    Sentiment analysis coverage: {(self.patient_segments_count/self.total_segments_count)*100:.1f}%")
                    # Calculate final metrics - only show satisfaction and latency at end
                    satisfaction_data = self.calculate_patient_satisfaction_score()
                    avg_latency = self.calculate_average_latency()
                    print(f"\n Final Results:")
                    print(f"    Patient Satisfaction Score: {satisfaction_data['satisfaction_score']:.1f}% ({satisfaction_data['classification']})")
                    print(f"    Average Processing Latency: {avg_latency:.1f}ms")
                    # Show detailed sentiment distribution
                    if len(self.patient_sentiments) > 0:
                        positive_count = sum(1 for s in self.patient_sentiments if s.get('score', 0) > 0.1)
                        negative_count = sum(1 for s in self.patient_sentiments if s.get('score', 0) < -0.1)
                        neutral_count = len(self.patient_sentiments) - positive_count - negative_count
                        print(f"\n Sentiment Score Breakdown:")
                        print(f"    Positive sentiments: {positive_count} ({positive_count/len(self.patient_sentiments)*100:.1f}%)")
                        print(f"    Negative sentiments: {negative_count} ({negative_count/len(self.patient_sentiments)*100:.1f}%)")
                        print(f"    Neutral sentiments: {neutral_count} ({neutral_count/len(self.patient_sentiments)*100:.1f}%)")
                        # Show individual sentiment scores
                        print(f"\n Individual Patient Sentiment Scores:")
                        for i, sentiment in enumerate(self.patient_sentiments[:10]):  # Show first 10
                            score = sentiment.get('score', 0)
                            label = sentiment.get('label', 'Unknown')
                            phase = sentiment.get('phase', 'during')
                            text_preview = sentiment.get('text', '')[:50] + "..." if len(sentiment.get('text', '')) > 50 else sentiment.get('text', '')
                            print(f"   {i+1:2d}. Score: {score:+.2f} | {label:8s} | Phase: {phase:6s} | \"{text_preview}\"")
                        if len(self.patient_sentiments) > 10:
                            print(f"   ... and {len(self.patient_sentiments) - 10} more sentiments")
                else:
                    print(f"    No patient sentiment data available for satisfaction analysis")
                # Clean up
                self.is_recording = False
                self.sentiment_running = False
                stream.stop_stream()
                stream.close()
                # Export RTTM file if enabled
                if (self.enable_rttm_export and 
                    hasattr(self.diarization_processor, 'rttm_segments') and 
                    len(self.diarization_processor.rttm_segments) > 0):
                    rttm_file = self.diarization_processor.export_rttm()
                    if rttm_file:
                        print(f" Diarization results saved to: {rttm_file}")
                # Wait for threads to finish
                send_thread.join(timeout=1.0)
                receive_thread.join(timeout=1.0)
                diarization_thread.join(timeout=1.0)
                assignment_thread.join(timeout=1.0)
                sentiment_thread.join(timeout=1.0)
                assignment_thread.join(timeout=1.0)
        except Exception as e:
            print(f"Error with audio stream: {e}")
            print("\nTroubleshooting tips:")
            print("1. Make sure your microphone is connected and working")
            print("2. Check Windows audio settings")
            print("3. Try running as administrator")
        finally:
            if p is not None:
                p.terminate()
            if self.socket:
                self.socket.close()
                print("✓ Disconnected from server")
        # ...existing code...

def test_microphone():
    """Test if microphone is working"""
    print("Testing microphone...")
    p = pyaudio.PyAudio()
    
    try:
        # Try to open a stream
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLING_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE
        )
        
        print("✓ Microphone test successful!")
        stream.close()
        return True
        
    except Exception as e:
        print(f"✗ Microphone test failed: {e}")
        return False
    finally:
        p.terminate()

if __name__ == "__main__":
    print("Live Whisper Transcription with Speaker Diarization")
    print("Enhanced with Fine-Tuned DistilBERT Classifiers")
    print("Treatment Phase Detection and Patient Sentiment Analysis")
    print("=" * 60)
    
    # Test microphone first
    if not test_microphone():
        print("\nPlease fix microphone issues before continuing.")
        sys.exit(1)
    
    print("\nSpeaker Classification System:")
    print("   • pyannote.audio 3.1 (acoustic analysis)")
    print("   • Fine-tuned DistilBERT (conversation patterns)")
    print("   • Real-time speaker identification")
    
    print("\nTreatment Phase Detection:")
    print("   • Fine-tuned DistilBERT phase model")
    print("   • Before/During/After classification")
    print("   • Applied to all segments")
    print("   • Real-time phase analysis")
    
    print("\nPatient Sentiment Analysis:")
    print("   • Fine-tuned DistilBERT model")
    print("   • VADER for negation handling")
    print("   • Patient segments only")
    print("   • Inline sentiment display")
    print("")
    
    # Start streaming
    streamer = MicrophoneStreamerWithDiarization()
    streamer.start_streaming()
