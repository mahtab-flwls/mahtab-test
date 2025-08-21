import librosa
import numpy as np
import soundfile as sf
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import os
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

class AudioSeparator:
    def __init__(self):
        self.sample_rate = 22050
        self.hop_length = 512
        self.n_mels = 128
        self.classifier = None
        self.scaler = None
        self.model_path = "speech_singing_classifier.pkl"
        self.scaler_path = "feature_scaler.pkl"
        
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file and return audio data and sample rate."""
        try:
            audio, sr = librosa.load(file_path, sr=self.sample_rate)
            return audio, sr
        except Exception as e:
            raise ValueError(f"Error loading audio file: {str(e)}")
    
    def extract_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract audio features for classification."""
        # Extract various audio features
        features = []
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
        features.extend([np.mean(spectral_centroids), np.std(spectral_centroids)])
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)[0]
        features.extend([np.mean(spectral_rolloff), np.std(spectral_rolloff)])
        
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate)[0]
        features.extend([np.mean(spectral_bandwidth), np.std(spectral_bandwidth)])
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        features.extend([np.mean(zcr), np.std(zcr)])
        
        # MFCC features
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
        for i in range(13):
            features.extend([np.mean(mfccs[i]), np.std(mfccs[i])])
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=audio, sr=self.sample_rate)
        features.extend([np.mean(chroma), np.std(chroma)])
        
        # Tempo
        tempo, _ = librosa.beat.beat_track(y=audio, sr=self.sample_rate)
        features.append(tempo)
        
        # RMS energy
        rms = librosa.feature.rms(y=audio)[0]
        features.extend([np.mean(rms), np.std(rms)])
        
        return np.array(features)
    
    def segment_audio(self, audio: np.ndarray, segment_length: float = 2.0) -> List[np.ndarray]:
        """Segment audio into smaller chunks for analysis."""
        segment_samples = int(segment_length * self.sample_rate)
        segments = []
        
        for i in range(0, len(audio), segment_samples):
            segment = audio[i:i + segment_samples]
            if len(segment) >= segment_samples // 2:  # Keep segments that are at least half the target length
                # Pad short segments
                if len(segment) < segment_samples:
                    segment = np.pad(segment, (0, segment_samples - len(segment)), mode='constant')
                segments.append(segment)
        
        return segments
    
    def train_classifier(self, speech_files: List[str], singing_files: List[str]):
        """Train a classifier to distinguish between speech and singing."""
        print("Training classifier...")
        
        X = []
        y = []
        
        # Process speech files
        for file_path in speech_files:
            try:
                audio, _ = self.load_audio(file_path)
                segments = self.segment_audio(audio)
                for segment in segments:
                    features = self.extract_features(segment)
                    X.append(features)
                    y.append(0)  # 0 for speech
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        # Process singing files
        for file_path in singing_files:
            try:
                audio, _ = self.load_audio(file_path)
                segments = self.segment_audio(audio)
                for segment in segments:
                    features = self.extract_features(segment)
                    X.append(features)
                    y.append(1)  # 1 for singing
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        X = np.array(X)
        y = np.array(y)
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train classifier
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.classifier.fit(X_scaled, y)
        
        # Save models
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.classifier, f)
        with open(self.scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print(f"Classifier trained with {len(X)} samples")
        print(f"Speech samples: {np.sum(y == 0)}, Singing samples: {np.sum(y == 1)}")
    
    def load_classifier(self):
        """Load pre-trained classifier and scaler."""
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            with open(self.model_path, 'rb') as f:
                self.classifier = pickle.load(f)
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            return True
        return False
    
    def create_default_classifier(self):
        """Create a simple default classifier based on audio characteristics."""
        print("Creating default classifier based on audio characteristics...")
        
        # This is a simplified approach that uses basic audio features
        # In a real-world scenario, you would train with actual data
        class SimpleClassifier:
            def predict(self, X):
                predictions = []
                for features in X:
                    # Simple heuristic: singing typically has more stable pitch and higher spectral centroid
                    spectral_centroid_mean = features[0]
                    spectral_centroid_std = features[1]
                    zcr_mean = features[6]
                    
                    # If spectral centroid is high and stable, and ZCR is low, likely singing
                    if spectral_centroid_mean > 2000 and spectral_centroid_std < 1000 and zcr_mean < 0.1:
                        predictions.append(1)  # singing
                    else:
                        predictions.append(0)  # speech
                
                return np.array(predictions)
        
        self.classifier = SimpleClassifier()
        self.scaler = StandardScaler()
        # Fit scaler with dummy data
        dummy_data = np.random.randn(100, 37)  # 37 features
        self.scaler.fit(dummy_data)
    
    def classify_segments(self, audio: np.ndarray) -> List[Tuple[int, int, str]]:
        """Classify audio segments as speech or singing."""
        if self.classifier is None:
            if not self.load_classifier():
                self.create_default_classifier()
        
        segments = self.segment_audio(audio)
        classifications = []
        
        segment_samples = int(2.0 * self.sample_rate)
        
        for i, segment in enumerate(segments):
            features = self.extract_features(segment)
            features_scaled = self.scaler.transform([features])
            prediction = self.classifier.predict(features_scaled)[0]
            
            start_time = i * segment_samples
            end_time = start_time + len(segment)
            label = "singing" if prediction == 1 else "speech"
            
            classifications.append((start_time, end_time, label))
        
        return classifications
    
    def separate_audio(self, file_path: str, output_dir: str = "output") -> Tuple[str, str]:
        """Separate speech and singing from audio file."""
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load audio
        audio, sr = self.load_audio(file_path)
        
        # Classify segments
        classifications = self.classify_segments(audio)
        
        # Separate audio
        speech_segments = []
        singing_segments = []
        
        for start, end, label in classifications:
            segment = audio[start:end]
            if label == "speech":
                speech_segments.append(segment)
            else:
                singing_segments.append(segment)
        
        # Combine segments
        speech_audio = np.concatenate(speech_segments) if speech_segments else np.array([])
        singing_audio = np.concatenate(singing_segments) if singing_segments else np.array([])
        
        # Save separated audio
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        speech_path = os.path.join(output_dir, f"{base_name}_speech.wav")
        singing_path = os.path.join(output_dir, f"{base_name}_singing.wav")
        
        if len(speech_audio) > 0:
            sf.write(speech_path, speech_audio, sr)
        else:
            # Create empty file
            sf.write(speech_path, np.array([0.0]), sr)
            
        if len(singing_audio) > 0:
            sf.write(singing_path, singing_audio, sr)
        else:
            # Create empty file
            sf.write(singing_path, np.array([0.0]), sr)
        
        return speech_path, singing_path
    
    def analyze_audio(self, file_path: str) -> dict:
        """Analyze audio file and return detailed information."""
        audio, sr = self.load_audio(file_path)
        classifications = self.classify_segments(audio)
        
        total_duration = len(audio) / sr
        speech_duration = sum((end - start) / sr for start, end, label in classifications if label == "speech")
        singing_duration = sum((end - start) / sr for start, end, label in classifications if label == "singing")
        
        return {
            "total_duration": total_duration,
            "speech_duration": speech_duration,
            "singing_duration": singing_duration,
            "speech_percentage": (speech_duration / total_duration) * 100 if total_duration > 0 else 0,
            "singing_percentage": (singing_duration / total_duration) * 100 if total_duration > 0 else 0,
            "segments": classifications
        }