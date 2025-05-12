"""
Audio and text processing utilities for the audio-text model.
"""

import torch
import numpy as np
import librosa
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoFeatureExtractor
import logging

logger = logging.getLogger(__name__)

class AudioTextProcessor:
    """Handles audio and text processing for the audio-text model."""
    
    def __init__(
        self, 
        text_model_name="sentence-transformers/all-roberta-large-v1",
        audio_model_name="facebook/w2v-bert-2.0", 
        device=None,
        max_text_length=256,
        sampling_rate=16000,
        max_audio_length=480000  # 10 seconds at 16kHz
    ):
        """Initialize processor with tokenizer and feature extractor."""
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_text_length = max_text_length
        self.sampling_rate = sampling_rate
        self.max_audio_length = max_audio_length
        
        logger.info(f"Loading tokenizer from {text_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        
        logger.info(f"Loading feature extractor from {audio_model_name}")
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(audio_model_name)
        
        # Check feature extractor keys
        dummy_audio = np.zeros(1000, dtype=np.float32)
        dummy_features = self.feature_extractor(
            dummy_audio, 
            sampling_rate=self.sampling_rate,
            return_tensors="pt"
        )
        logger.info(f"Feature extractor output keys: {list(dummy_features.keys())}")
    
    def process_text(self, text):
        """Process text for embedding."""
        logger.info(f"Processing text: {text[:50]}{'...' if len(text) > 50 else ''}")
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            max_length=self.max_text_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Move to device
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
    
    def process_audio_file(self, audio_path):
        """Process an audio file for embedding."""
        logger.info(f"Processing audio file: {audio_path}")
        
        # Load audio file
        audio_array, orig_sr = librosa.load(audio_path, sr=None)
        logger.info(f"Original audio: sampling rate={orig_sr}, length={len(audio_array)}")
        
        return self.process_audio_array(audio_array, orig_sr)
    
    def process_audio_array(self, audio_array, orig_sr):
        """Process an audio array for embedding."""
        # Resample if needed
        if orig_sr != self.sampling_rate:
            logger.info(f"Resampling from {orig_sr}Hz to {self.sampling_rate}Hz")
            audio_array = librosa.resample(audio_array, orig_sr=orig_sr, target_sr=self.sampling_rate)
            logger.info(f"After resampling: length={len(audio_array)}")
        
        # Convert to float32
        audio_array = audio_array.astype(np.float32)
        
        # Normalize
        if np.abs(audio_array).max() > 1.0:
            audio_array = audio_array / np.abs(audio_array).max()
        
        # Trim for maximum length
        if len(audio_array) > self.max_audio_length:
            logger.info(f"Trimming audio from {len(audio_array)} to {self.max_audio_length} samples")
            audio_array = audio_array[:self.max_audio_length]
        
        # Extract features
        logger.info("Extracting audio features...")
        audio_features = self.feature_extractor(
            audio_array,
            sampling_rate=self.sampling_rate,
            return_tensors="pt"
        )
        
        # Determine the correct key for input features
        if "input_features" in audio_features:
            input_features_key = "input_features"
        elif "input_values" in audio_features:
            input_features_key = "input_values"
        else:
            raise ValueError(f"Neither 'input_features' nor 'input_values' found in audio features")
        
        logger.info(f"Using '{input_features_key}' for audio features")
        
        # Move to device
        input_features = audio_features[input_features_key].to(self.device)
        attention_mask_audio = audio_features.get("attention_mask")
        if attention_mask_audio is not None:
            attention_mask_audio = attention_mask_audio.to(self.device)
        
        return {
            "input_features": input_features,
            "attention_mask_audio": attention_mask_audio
        }
    
    def get_text_embedding(self, model, text_input):
        """Get embeddings for processed text input."""
        with torch.no_grad():
            text_embedding, _ = model.encode_text(
                text_input["input_ids"], 
                text_input["attention_mask"]
            )
            text_embedding = F.normalize(text_embedding, p=2, dim=1)
        return text_embedding
    
    def get_audio_embedding(self, model, audio_input):
        """Get embeddings for processed audio input."""
        with torch.no_grad():
            audio_embedding, _ = model.encode_audio(
                audio_input["input_features"],
                audio_input["attention_mask_audio"]
            )
            audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
        return audio_embedding

    def compute_similarity(self, embedding1, embedding2):
        """Compute cosine similarity between embeddings."""
        # Ensure embeddings are normalized
        if not torch.allclose(torch.norm(embedding1, p=2, dim=1), torch.ones(1, device=self.device), atol=1e-4):
            embedding1 = F.normalize(embedding1, p=2, dim=1)
        
        if not torch.allclose(torch.norm(embedding2, p=2, dim=1), torch.ones(1, device=self.device), atol=1e-4):
            embedding2 = F.normalize(embedding2, p=2, dim=1)
        
        # Compute similarity
        similarity = torch.sum(embedding1 * embedding2, dim=1).cpu().numpy()
        return similarity