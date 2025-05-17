import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from neattext.functions import clean_text
import joblib
import os

class EmotionDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.model = LogisticRegression(max_iter=1000)
        self.emotions = ['happy', 'sad', 'angry']
        
    def preprocess_text(self, text):
        """Clean and preprocess the input text"""
        return clean_text(text, 
                         puncts=True, 
                         urls=True, 
                         emails=True, 
                         numbers=True, 
                         special_char=True).lower()
    
    def prepare_sample_data(self):
        """Create sample training data"""
        # Sample data for demonstration
        texts = [
            "I am so happy today! Everything is going great!",
            "I feel wonderful and excited about the future!",
            "This is the best day ever!",
            "I'm feeling really down and sad today",
            "Everything seems so gloomy and depressing",
            "I can't stop crying, I'm so sad",
            "I'm so angry right now!",
            "This makes me furious!",
            "I can't believe this happened, I'm so mad!",
        ]
        
        labels = ['happy', 'happy', 'happy', 
                 'sad', 'sad', 'sad',
                 'angry', 'angry', 'angry']
        
        return texts, labels
    
    def train(self):
        """Train the emotion detection model"""
        # Get sample data
        texts, labels = self.prepare_sample_data()
        
        # Preprocess texts
        cleaned_texts = [self.preprocess_text(text) for text in texts]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            cleaned_texts, labels, test_size=0.2, random_state=42
        )
        
        # Vectorize text
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        # Train model
        self.model.fit(X_train_vec, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_vec)
        print("\nModel Evaluation:")
        print(classification_report(y_test, y_pred))
        
        # Plot confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.emotions,
                    yticklabels=self.emotions)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
    
    def predict_emotion(self, text):
        """Predict emotion from input text"""
        # Preprocess text
        cleaned_text = self.preprocess_text(text)
        
        # Vectorize text
        text_vec = self.vectorizer.transform([cleaned_text])
        
        # Predict emotion
        prediction = self.model.predict(text_vec)[0]
        probabilities = self.model.predict_proba(text_vec)[0]
        
        # Get confidence scores for each emotion
        emotion_scores = dict(zip(self.emotions, probabilities))
        
        return {
            'predicted_emotion': prediction,
            'confidence_scores': emotion_scores
        }
    
    def save_model(self, model_path='emotion_model.joblib'):
        """Save the trained model and vectorizer"""
        model_data = {
            'vectorizer': self.vectorizer,
            'model': self.model
        }
        joblib.dump(model_data, model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path='emotion_model.joblib'):
        """Load a trained model and vectorizer"""
        if os.path.exists(model_path):
            model_data = joblib.load(model_path)
            self.vectorizer = model_data['vectorizer']
            self.model = model_data['model']
            print(f"Model loaded from {model_path}")
        else:
            print(f"No model found at {model_path}") 