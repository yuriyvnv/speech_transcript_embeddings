#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced Audio-Text Embedding Model Trainer with Word-Level Alignment
This implementation addresses issues with high similarity for incorrect transcripts
by adding word-level alignment mechanisms.
"""

import os
import torch
import logging
import numpy as np
import time
import random
import json
import argparse
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, AutoFeatureExtractor
from transformers import get_linear_schedule_with_warmup
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from datasets import load_dataset, Audio
from info_nce import InfoNCE
import matplotlib.pyplot as plt


# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

# Configure environment settings
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"


# ===== ENHANCED COMPONENTS =====

class EnhancedProjection(nn.Module):
    """
    Enhanced projection layer with multiple linear transformations and non-linearities.
    """
    def __init__(
        self,
        input_dim, 
        projection_dim, 
        hidden_dim=None, 
        dropout=0.1,
        activation="gelu"
    ):
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = projection_dim * 2
        
        if activation == "gelu":
            activation_fn = nn.GELU()
        elif activation == "relu":
            activation_fn = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
            
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation_fn,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, projection_dim),
            nn.LayerNorm(projection_dim)
        )
    
    def forward(self, x):
        return self.projection(x)


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention mechanism to capture relationships between modalities.
    """
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Multi-head attention components
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        # Initialization
        nn.init.xavier_uniform_(self.query.weight)
        nn.init.xavier_uniform_(self.key.weight)
        nn.init.xavier_uniform_(self.value.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
    
    def forward(self, x, context, attention_mask=None):
        """
        Apply cross-modal attention from x to context.
        
        Args:
            x: Query tensor [batch_size, seq_len_q, dim]
            context: Key/Value tensor [batch_size, seq_len_kv, dim]
            attention_mask: Optional mask [batch_size, seq_len_q, seq_len_kv]
        
        Returns:
            Tensor with same shape as x after attention with context
        """
        batch_size = x.shape[0]
        
        # Project and reshape for multi-head attention
        q = self.query(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(context).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(context).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Calculate attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided
        if attention_mask is not None:
            # Reshape mask for broadcasting
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, seq_len_kv]
            # Convert 0s to -inf before softmax
            attn_weights = attn_weights.masked_fill(attention_mask == 0, -1e9)
        
        # Apply softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, v)
        
        # Reshape back to original dimensions
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        
        # Final projection
        output = self.out_proj(output)
        
        return output


class AttentivePooling(nn.Module):
    """
    Attentive pooling for sequence data.
    Computes a weighted average of the sequence based on learned attention weights.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, hidden_states, attention_mask=None):
        """
        Apply attentive pooling to sequence.
        
        Args:
            hidden_states: Sequence to pool [batch_size, seq_len, hidden_size]
            attention_mask: Optional mask [batch_size, seq_len] (1=keep, 0=mask)
            
        Returns:
            Pooled representation [batch_size, hidden_size]
        """
        # Calculate attention scores
        attention_scores = self.attention(hidden_states).squeeze(-1)
        
        # Apply mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e9)
        
        # Apply softmax to get weights
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Apply attention weights
        pooled_output = torch.bmm(
            attention_weights.unsqueeze(1),  # [batch_size, 1, seq_len] 
            hidden_states                    # [batch_size, seq_len, hidden_size]
        ).squeeze(1)  # [batch_size, hidden_size]
        
        return pooled_output


class WordLevelAlignmentModule(nn.Module):
    """
    Module to align word-level representations from text with temporal segments in audio.
    Uses attention mechanism to create a soft alignment between words and audio frames.
    """
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Projection layers to create query (text) and key/value (audio) representations
        self.text_projection = nn.Linear(hidden_dim, hidden_dim)
        self.audio_projection = nn.Linear(hidden_dim, hidden_dim)
        
        # Multi-head attention for alignment
        self.alignment_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection and layer norm
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Alignment confidence scorer (predicts how well each word aligns with audio)
        self.alignment_confidence = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, text_hidden_states, audio_hidden_states, 
                text_attention_mask=None, audio_attention_mask=None):
        """
        Compute alignment between text words and audio frames.
        
        Args:
            text_hidden_states: Text token representations [batch_size, text_len, hidden_dim]
            audio_hidden_states: Audio frame representations [batch_size, audio_len, hidden_dim]
            text_attention_mask: Text mask [batch_size, text_len]
            audio_attention_mask: Audio mask [batch_size, audio_len]
            
        Returns:
            aligned_representations: Text representations aligned with audio [batch_size, text_len, hidden_dim]
            alignment_scores: Word-level alignment scores [batch_size, text_len]
            alignment_matrix: Full alignment matrix [batch_size, text_len, audio_len]
        """
        batch_size, text_len, _ = text_hidden_states.shape
        _, audio_len, _ = audio_hidden_states.shape
        
        # Project text and audio
        text_proj = self.text_projection(text_hidden_states)
        audio_proj = self.audio_projection(audio_hidden_states)
        
        # Convert masks to format needed by MultiheadAttention
        if text_attention_mask is not None:
            text_key_padding_mask = (1.0 - text_attention_mask).bool()
        else:
            text_key_padding_mask = None
            
        if audio_attention_mask is not None:
            audio_key_padding_mask = (1.0 - audio_attention_mask).bool()
        else:
            audio_key_padding_mask = None
        
        # Compute text-to-audio attention (words attending to relevant audio frames)
        aligned_representations, alignment_weights = self.alignment_attention(
            query=text_proj,
            key=audio_proj,
            value=audio_proj,
            key_padding_mask=audio_key_padding_mask,
            need_weights=True,
            average_attn_weights=False  # Return all attention heads
        )
        
        # Average attention weights across heads to get final alignment matrix
        alignment_matrix = alignment_weights.mean(dim=1)
        
        # Apply residual connection and layer norm
        aligned_representations = self.layer_norm(
            text_hidden_states + self.output_projection(aligned_representations)
        )
        
        # Compute confidence score for each word alignment
        # Higher score = more confident that the word aligns with some part of the audio
        alignment_scores = self.alignment_confidence(aligned_representations).squeeze(-1)
        
        # Mask out padding tokens
        if text_attention_mask is not None:
            alignment_scores = alignment_scores * text_attention_mask
        
        return aligned_representations, alignment_scores, alignment_matrix


# ===== ENHANCED AUDIO-TEXT MODEL =====

class EnhancedAudioTextModel(nn.Module):
    """
    Enhanced Audio-Text multimodal embedding model with:
    - Improved projection layers
    - Cross-modal attention
    - Attentive pooling
    - Word-level alignment
    """
    def __init__(
        self,
        text_model_name="sentence-transformers/all-roberta-large-v1",
        audio_model_name="facebook/w2v-bert-2.0",
        projection_dim=1024,
        text_embedding_dim=1024,
        audio_embedding_dim=1024,
        dropout=0.1,
        use_cross_modal=True,
        use_attentive_pooling=True,
        use_word_alignment=True,  # New parameter
        freeze_encoders=False,    # Default to False as recommended
    ):
        super().__init__()
        
        # Load pre-trained models
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.audio_encoder = AutoModel.from_pretrained(audio_model_name)
        
        # Save configuration
        self.projection_dim = projection_dim
        self.use_cross_modal = use_cross_modal
        self.use_attentive_pooling = use_attentive_pooling
        self.use_word_alignment = use_word_alignment
        
        # Freeze encoders if requested
        if freeze_encoders:
            logger.info("Freezing text and audio encoders")
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            for param in self.audio_encoder.parameters():
                param.requires_grad = False
        
        # Enhanced projection heads
        self.text_projection = EnhancedProjection(
            input_dim=text_embedding_dim,
            projection_dim=projection_dim,
            dropout=dropout
        )
        
        self.audio_projection = EnhancedProjection(
            input_dim=audio_embedding_dim,
            projection_dim=projection_dim,
            dropout=dropout
        )
        
        # Cross-modal attention (optional)
        if use_cross_modal:
            self.text_to_audio_attention = CrossModalAttention(
                dim=projection_dim, 
                dropout=dropout
            )
            self.audio_to_text_attention = CrossModalAttention(
                dim=projection_dim, 
                dropout=dropout
            )
            
            # Fusion layers to combine original and cross-attended features
            self.text_fusion = nn.Sequential(
                nn.Linear(projection_dim * 2, projection_dim),
                nn.LayerNorm(projection_dim)
            )
            self.audio_fusion = nn.Sequential(
                nn.Linear(projection_dim * 2, projection_dim),
                nn.LayerNorm(projection_dim)
            )
        
        # Attentive pooling (optional)
        if use_attentive_pooling:
            self.text_pooling = AttentivePooling(text_embedding_dim)
            self.audio_pooling = AttentivePooling(audio_embedding_dim)
        
        # Add word-level alignment module (NEW)
        if use_word_alignment:
            self.word_level_alignment = WordLevelAlignmentModule(
                hidden_dim=projection_dim,
                dropout=dropout
            )
            
            # Add a layer to incorporate alignment scores into final similarity
            self.alignment_weighting = nn.Sequential(
                nn.Linear(projection_dim, 1),
                nn.Sigmoid()
            )
        
        # Log number of trainable parameters
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"Model initialized with {trainable_params:,} trainable parameters out of {total_params:,} total")
    
    def encode_text(self, input_ids, attention_mask=None):
        """
        Encode text inputs with enhanced processing.
        """
        # Get text embeddings from encoder
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Choose pooling method
        if self.use_attentive_pooling:
            # Apply attentive pooling
            text_embedding = self.text_pooling(outputs.last_hidden_state, attention_mask)
        else:
            # Use CLS token embedding (BERT-style) or mean pooling
            text_embedding = outputs.last_hidden_state[:, 0, :]  # CLS token
        
        # Project to shared space
        projected = self.text_projection(text_embedding)
        
        return projected, outputs.last_hidden_state
    
    def encode_audio(self, input_features, attention_mask=None):
        """
        Encode audio inputs with enhanced processing.
        """
        # Adapt input name based on the model's expected input
        try:
            # First try with input_values (older models)
            outputs = self.audio_encoder(
                input_values=input_features,
                attention_mask=attention_mask
            )
        except TypeError:
            try:
                # Then try with input_features (newer models)
                outputs = self.audio_encoder(
                    input_features=input_features,
                    attention_mask=attention_mask
                )
            except TypeError as e:
                # Fallback to a generic approach
                logger.warning(f"Using fallback approach for audio encoder: {e}")
                outputs = self.audio_encoder(input_features)
        
        # Handle different output formats
        if hasattr(outputs, 'last_hidden_state'):
            hidden_states = outputs.last_hidden_state
        else:
            # Assume the first element is the hidden states
            hidden_states = outputs[0]
        
        # Choose pooling method
        if self.use_attentive_pooling:
            # Apply attentive pooling
            audio_embedding = self.audio_pooling(hidden_states, attention_mask)
        else:
            # Apply masking for proper mean calculation
            if attention_mask is not None:
                # Expand mask for feature dimension
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
                # Mask the hidden states
                masked_hidden = hidden_states * mask_expanded
                # Sum and normalize by the number of actual tokens
                sum_embeddings = torch.sum(masked_hidden, dim=1)
                sum_mask = torch.sum(mask_expanded, dim=1)
                # Avoid division by zero
                sum_mask = torch.clamp(sum_mask, min=1e-9)
                audio_embedding = sum_embeddings / sum_mask
            else:
                # If no mask, just average over time dimension
                audio_embedding = torch.mean(hidden_states, dim=1)
        
        # Project to shared space
        projected = self.audio_projection(audio_embedding)
        
        return projected, hidden_states
    
    def apply_cross_modal_attention(self, 
                                   text_projected, text_hidden, text_mask,
                                   audio_projected, audio_hidden, audio_mask):
        """
        Apply cross-modal attention between text and audio.
        """
        if not self.use_cross_modal:
            return text_projected, audio_projected
        
        # Calculate cross-modal attentions
        text_attended = self.text_to_audio_attention(
            text_projected.unsqueeze(1),  # Add sequence dimension [B, 1, D]
            audio_hidden,                 # [B, S_a, D]
            audio_mask                    # [B, S_a]
        ).squeeze(1)  # Remove sequence dimension [B, D]
        
        audio_attended = self.audio_to_text_attention(
            audio_projected.unsqueeze(1), # Add sequence dimension [B, 1, D]
            text_hidden,                 # [B, S_t, D]
            text_mask                    # [B, S_t]
        ).squeeze(1)  # Remove sequence dimension [B, D]
        
        # Fuse original and attended features
        text_fused = self.text_fusion(torch.cat([text_projected, text_attended], dim=1))
        audio_fused = self.audio_fusion(torch.cat([audio_projected, audio_attended], dim=1))
        
        return text_fused, audio_fused
    
    def forward(self, batch):
        """
        Forward pass with enhanced processing and word-level alignment.
        """
        # Get base embeddings
        text_projected, text_hidden = self.encode_text(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )
        
        audio_projected, audio_hidden = self.encode_audio(
            input_features=batch["input_features"],
            attention_mask=batch["attention_mask_audio"]
        )
        
        # Apply cross-modal attention if enabled
        if self.use_cross_modal:
            text_embeddings, audio_embeddings = self.apply_cross_modal_attention(
                text_projected, text_hidden, batch["attention_mask"],
                audio_projected, audio_hidden, batch["attention_mask_audio"]
            )
        else:
            text_embeddings, audio_embeddings = text_projected, audio_projected
        
        # Store alignment info for loss calculation and visualization
        self.last_alignment_scores = None
        self.last_alignment_matrix = None
        
        # Apply word-level alignment if enabled
        if self.use_word_alignment:
            # Process hidden states through word-level alignment module
            aligned_text, alignment_scores, alignment_matrix = self.word_level_alignment(
                text_hidden_states=text_hidden,
                audio_hidden_states=audio_hidden,
                text_attention_mask=batch["attention_mask"],
                audio_attention_mask=batch["attention_mask_audio"]
            )
            
            # Store alignment info for loss calculation and visualization
            self.last_alignment_scores = alignment_scores
            self.last_alignment_matrix = alignment_matrix
            
            
            
            # Use a small residual connection to incorporate alignment information
            # This ensures that misaligned pairs get lower similarity
            #previous
            #alignment_factor = self.alignment_weighting(aligned_text.mean(dim=1))
            if batch["attention_mask"] is not None:
                # Set padding tokens to 1.0 (perfect alignment) so they don't affect min
                masked_scores = alignment_scores * batch["attention_mask"] + (1.0 - batch["attention_mask"])
                min_alignment = torch.min(masked_scores, dim=1)[0]
            else:
                min_alignment = torch.min(alignment_scores, dim=1)[0]

            # Scale the factor more aggressively
            alignment_factor = torch.sigmoid(min_alignment * 2).unsqueeze(-1)
            # Apply subtle adjustment based on alignment - this makes the model more
            # sensitive to misalignments between audio and text
            text_embeddings = text_embeddings * (0.1 + 0.9 * alignment_factor)
        
        # Normalize embeddings
        text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
        audio_embeddings = F.normalize(audio_embeddings, p=2, dim=1)
        
        return text_embeddings, audio_embeddings


# ===== ALIGNMENT-AWARE LOSS FUNCTION =====

class AlignmentAwareInfoNCE(nn.Module):
    """
    InfoNCE loss that incorporates word-level alignment quality to make 
    the model more sensitive to semantic mismatches.
    """
    def __init__(self, temperature=0.07, alignment_weight=0.3):
        super().__init__()
        self.temperature = temperature
        self.alignment_weight = alignment_weight
        self.base_loss = InfoNCE(temperature=temperature)
        
    def forward(self, text_embeddings, audio_embeddings, alignment_scores=None):
        # Standard InfoNCE loss
        base_loss = self.base_loss(text_embeddings, audio_embeddings)
        
        # If alignment scores are provided, use them to weight the loss
        if alignment_scores is not None:
            # Calculate mean alignment score per sample
            mean_alignment = alignment_scores.mean(dim=1)
            
            # Scale alignment scores to [0, 1]
            scaled_alignment = torch.sigmoid(mean_alignment)
            
            # Higher alignment should reduce loss (well-aligned samples are easier)
            # Lower alignment should increase loss (poorly-aligned samples need more focus)
            alignment_factor = 1.0 - (scaled_alignment * self.alignment_weight)
            
            # Apply the factor to the base loss
            weighted_loss = base_loss * alignment_factor.mean()
            return weighted_loss
        else:
            return base_loss


# ===== COMMON VOICE DATASET CLASS =====

class CommonVoiceDataset(Dataset):
    """
    Dataset for Common Voice audio-text pairs.
    Directly uses the dataset object from Hugging Face datasets.
    """
    def __init__(
        self,
        dataset,
        tokenizer,
        feature_extractor,
        max_text_length=128,
        sampling_rate=16000,
        max_audio_length=160000,  # 10 seconds at 16kHz
        add_corrupted_examples=True,  # NEW: Add corrupted text examples as hard negatives
        corruption_probability=0.2    # Chance of corruption per sample
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.max_text_length = max_text_length
        self.sampling_rate = sampling_rate
        self.max_audio_length = max_audio_length
        self.add_corrupted_examples = add_corrupted_examples
        self.corruption_probability = corruption_probability
        
        # Debug output the feature extractor to check its keys
        dummy_audio = np.zeros(1000, dtype=np.float32)
        dummy_features = self.feature_extractor(
            dummy_audio, 
            sampling_rate=self.sampling_rate,
            return_tensors="pt"
        )
        logger.info(f"Feature extractor output keys: {list(dummy_features.keys())}")
    
    def __len__(self):
        return len(self.dataset)
    
    def create_corrupted_transcript(self, text):
        """
        Create a version of the transcript with random corruptions.
        This helps the model learn to distinguish incorrect transcripts.
        """
        words = text.split()
        if len(words) <= 1:
            return text  # Don't corrupt very short texts
            
        # Various corruption strategies
        corruption_type = random.choice(["replace", "shuffle", "drop", "add", "partial"])
        
        if corruption_type == "replace":
            # Replace a random word with a different word
            replace_idx = random.randint(0, len(words) - 1)
            random_word = random.choice(["sim", "não", "e", "o", "de", "um", "uma", "tua", "qualquer", "coisa", "deveria", "gostaria", "imaginemos"])
            words[replace_idx] = random_word
            
        elif corruption_type == "shuffle":
            # Shuffle words (if there are enough)
            if len(words) > 2:
                shuffle_start = random.randint(0, len(words) - 2)
                shuffle_end = random.randint(shuffle_start + 1, len(words) - 1)
                shuffle_segment = words[shuffle_start:shuffle_end+1]
                random.shuffle(shuffle_segment)
                words[shuffle_start:shuffle_end+1] = shuffle_segment
                
        elif corruption_type == "drop":
            # Drop a random word
            drop_idx = random.randint(0, len(words) - 1)
            words.pop(drop_idx)
            
        elif corruption_type == "add":
            # Add a random word
            insert_idx = random.randint(0, len(words))
            random_word = random.choice(["sim", "não", "e", "o", "de", "um", "uma"])
            words.insert(insert_idx, random_word)
            
        elif corruption_type == "partial":
            # Only keep part of the transcript (first or last half)
            if random.random() < 0.5:
                words = words[:len(words)//2]
            else:
                words = words[len(words)//2:]
        
        return " ".join(words)
    
    def __getitem__(self, idx):
        try:
            # Get data from the dataset
            item = self.dataset[idx]
            text = item["sentence"]
            
            # Decide whether to corrupt this example
            use_corrupted = self.add_corrupted_examples and random.random() < self.corruption_probability
            
            if use_corrupted:
                # Create a corrupted version of the transcript
                orig_text = text
                text = self.create_corrupted_transcript(text)
                # Store both versions for the model to learn the difference
                is_corrupted = 1.0
            else:
                is_corrupted = 0.0
            
            # Process text
            encoding = self.tokenizer(
                text,
                max_length=self.max_text_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            # Process audio
            try:
                # Get audio array directly from the dataset
                speech_array = item["audio"]["array"]
                
                # Validation checks for audio
                if speech_array is None or len(speech_array) == 0:
                    logger.warning(f"Empty audio array for item {idx}")
                    return None
                
                # Ensure correct data type
                if speech_array.dtype != np.float32:
                    speech_array = speech_array.astype(np.float32)
                
                # Normalize if needed
                if np.abs(speech_array).max() > 1.0:
                    speech_array = speech_array / np.abs(speech_array).max()
                
                # Limit maximum audio length to avoid memory issues
                if len(speech_array) > self.max_audio_length:
                    logger.debug(f"Truncating long audio in item {idx} from {len(speech_array)} to {self.max_audio_length}")
                    speech_array = speech_array[:self.max_audio_length]
                
                # Process with feature extractor
                audio_features = self.feature_extractor(
                    speech_array,
                    sampling_rate=self.sampling_rate,
                    return_tensors="pt"
                )
                
                # Handle different key naming depending on the feature extractor
                if "input_features" in audio_features:
                    input_features_key = "input_features"
                elif "input_values" in audio_features:
                    input_features_key = "input_values"
                else:
                    logger.warning(f"Neither 'input_features' nor 'input_values' found in audio_features. Keys: {audio_features.keys()}")
                    return None
                
                # Remove batch dimension from features
                features = audio_features[input_features_key].squeeze(0)
                
                return {
                    "input_ids": encoding["input_ids"].squeeze(),
                    "attention_mask": encoding["attention_mask"].squeeze(),
                    "input_features": features,
                    "attention_mask_audio": audio_features.get("attention_mask", None),
                    "is_corrupted": is_corrupted  # Flag to track corrupted examples for analysis
                }
                
            except Exception as e:
                logger.warning(f"Error processing audio for item {idx}: {str(e)}")
                return None
                
        except Exception as e:
            logger.warning(f"Error processing item at index {idx}: {str(e)}")
            return None


# ===== CUSTOM COLLATE FUNCTION =====

def custom_collate_fn(batch):
    """
    Custom collate function that properly handles variable-length audio features.
    Also collates corruption flags for analysis.
    """
    # Filter out None values
    batch = [item for item in batch if item is not None]
    
    # If batch is empty after filtering, return dummy batch
    if len(batch) == 0:
        logger.warning("Empty batch after filtering")
        return {
            "input_ids": torch.zeros((1, 128), dtype=torch.long),
            "attention_mask": torch.zeros((1, 128), dtype=torch.long),
            "input_features": torch.zeros((1, 1, 160), dtype=torch.float),
            "attention_mask_audio": torch.zeros((1, 1), dtype=torch.long),
            "is_corrupted": torch.zeros(1, dtype=torch.float)
        }
    
    # Check if all required keys are present
    required_keys = ["input_ids", "attention_mask", "input_features"]
    valid_batch = []
    
    for item in batch:
        if all(key in item for key in required_keys):
            valid_batch.append(item)
        else:
            missing = [key for key in required_keys if key not in item]
            logger.warning(f"Item missing keys: {missing}. Available keys: {item.keys()}")
    
    if not valid_batch:
        logger.warning("No valid items in batch")
        return {
            "input_ids": torch.zeros((1, 128), dtype=torch.long),
            "attention_mask": torch.zeros((1, 128), dtype=torch.long),
            "input_features": torch.zeros((1, 1, 160), dtype=torch.float),
            "attention_mask_audio": torch.zeros((1, 1), dtype=torch.long),
            "is_corrupted": torch.zeros(1, dtype=torch.float)
        }
    
    batch = valid_batch
    
    # Separate items 
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    input_features = [item["input_features"] for item in batch]
    is_corrupted = [item.get("is_corrupted", 0.0) for item in batch]
    
    # Get shapes for debugging
    feature_shapes = [feat.shape for feat in input_features]
    logger.debug(f"Feature shapes in batch: {feature_shapes}")
    
    # Determine max length for audio features
    max_time_dim = max([feat.size(0) for feat in input_features])
    feature_dim = input_features[0].size(1)  # Should typically be 160
    
    # Create padded tensors with the right dimensions
    padded_features = torch.zeros(len(batch), max_time_dim, feature_dim)
    attention_mask_audio = torch.zeros(len(batch), max_time_dim)
    
    # Fill in the actual values and create masks
    for i, feat in enumerate(input_features):
        time_dim = feat.size(0)
        padded_features[i, :time_dim, :] = feat
        attention_mask_audio[i, :time_dim] = 1  # 1 indicates actual content
    
    # Pad and stack text inputs
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask_padded = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    
    # Convert corruption flags to tensor
    is_corrupted_tensor = torch.tensor(is_corrupted, dtype=torch.float)
    
    return {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask_padded,
        "input_features": padded_features,
        "attention_mask_audio": attention_mask_audio,
        "is_corrupted": is_corrupted_tensor
    }


# ===== TRAINING AND EVALUATION FUNCTIONS =====

def train_epoch(model, data_loader, optimizer, scheduler, loss_fn, device, 
               epoch, accumulation_steps=4, scaler=None):
    """
    Train for one epoch with gradient accumulation and mixed precision.
    Includes tracking of performance on corrupted vs clean samples.
    """
    model.train()
    total_loss = 0
    sample_count = 0
    
    # Extra tracking for analysis
    clean_similarities = []
    corrupt_similarities = []
    
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch} [Train]")
    optimizer.zero_grad()  # Zero gradients at the start
    
    for batch_idx, batch in enumerate(progress_bar):
        try:
            # Check if we have a dummy batch (due to filtering out None values)
            if batch["input_ids"].shape[0] == 1 and torch.all(batch["input_ids"] == 0):
                logger.warning(f"Skipping dummy batch {batch_idx}")
                continue
                
            # Move inputs to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass with mixed precision if enabled
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    text_embeddings, audio_embeddings = model(batch)
                    
                    # Get alignment scores if available for alignment-aware loss
                    alignment_scores = getattr(model, 'last_alignment_scores', None)
                    
                    # Calculate loss with alignment awareness if possible
                    if isinstance(loss_fn, AlignmentAwareInfoNCE) and alignment_scores is not None:
                        loss_t2a = loss_fn(text_embeddings, audio_embeddings, alignment_scores)
                        loss_a2t = loss_fn(audio_embeddings, text_embeddings, alignment_scores)
                    else:
                        loss_t2a = loss_fn(text_embeddings, audio_embeddings)
                        loss_a2t = loss_fn(audio_embeddings, text_embeddings)
                    
                    loss = (loss_t2a + loss_a2t) / 2.0
                    
                    # Scale the loss by accumulation steps for gradient accumulation
                    loss = loss / accumulation_steps
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                
                # Step optimizer and scaler based on accumulation
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(data_loader):
                    # Gradient clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    # Update weights and scheduler
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
            else:
                # Standard precision training
                text_embeddings, audio_embeddings = model(batch)
                
                # Get alignment scores if available
                alignment_scores = getattr(model, 'last_alignment_scores', None)
                
                # Calculate loss with alignment awareness if possible
                if isinstance(loss_fn, AlignmentAwareInfoNCE) and alignment_scores is not None:
                    loss_t2a = loss_fn(text_embeddings, audio_embeddings, alignment_scores)
                    loss_a2t = loss_fn(audio_embeddings, text_embeddings, alignment_scores)
                else:
                    loss_t2a = loss_fn(text_embeddings, audio_embeddings)
                    loss_a2t = loss_fn(audio_embeddings, text_embeddings)
                
                loss = (loss_t2a + loss_a2t) / 2.0
                # Scale the loss for gradient accumulation
                loss = loss / accumulation_steps
                
                # Backward pass
                loss.backward()
                
                # Step based on accumulation
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(data_loader):
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    # Update weights and scheduler
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
            
            # Track similarities between audio and text embeddings
            with torch.no_grad():
                similarities = torch.sum(text_embeddings * audio_embeddings, dim=1).cpu()
                
                # Separate clean and corrupted samples for analysis
                is_corrupted = batch["is_corrupted"].cpu().bool()
                if torch.any(is_corrupted):
                    corrupt_similarities.extend(similarities[is_corrupted].numpy())
                if torch.any(~is_corrupted):
                    clean_similarities.extend(similarities[~is_corrupted].numpy())
            
            # Update progress bar (use the non-scaled loss for reporting)
            batch_size = batch["input_ids"].shape[0]
            sample_count += batch_size
            # Multiply back by accumulation steps to get the actual loss value
            total_loss += (loss.item() * accumulation_steps) * batch_size
            avg_loss = total_loss / sample_count if sample_count > 0 else float('inf')
            
            # Calculate average similarities for clean and corrupted samples
            avg_clean_sim = np.mean(clean_similarities) if clean_similarities else 0
            avg_corrupt_sim = np.mean(corrupt_similarities) if corrupt_similarities else 0
            sim_diff = avg_clean_sim - avg_corrupt_sim
            
            # Update progress bar with more info
            progress_bar.set_postfix({
                "loss": f"{avg_loss:.4f}", 
                "clean_sim": f"{avg_clean_sim:.3f}",
                "corrupt_sim": f"{avg_corrupt_sim:.3f}",
                "diff": f"{sim_diff:.3f}"
            })
            
            # Explicitly clear some memory
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"Error in batch {batch_idx}: {str(e)}")
            # Continue with next batch
            continue
    
    # Return more detailed metrics
    train_metrics = {
        "loss": total_loss / sample_count if sample_count > 0 else float('inf'),
        "clean_similarity": np.mean(clean_similarities) if clean_similarities else 0,
        "corrupt_similarity": np.mean(corrupt_similarities) if corrupt_similarities else 0,
        "similarity_gap": np.mean(clean_similarities) - np.mean(corrupt_similarities) 
                         if clean_similarities and corrupt_similarities else 0
    }
    
    return train_metrics


def evaluate(model, data_loader, loss_fn, device, epoch=None, split="Validation", fp16=False):
    """
    Evaluate the model with proper error handling.
    Tracks performance on clean vs corrupted examples.
    """
    model.eval()
    total_loss = 0
    all_similarities = []
    clean_similarities = []
    corrupt_similarities = []
    sample_count = 0
    
    desc = f"Epoch {epoch} [{split}]" if epoch is not None else f"[{split}]"
    progress_bar = tqdm(data_loader, desc=desc)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Check if we have a dummy batch (due to filtering out None values)
                if batch["input_ids"].shape[0] == 1 and torch.all(batch["input_ids"] == 0):
                    logger.warning(f"Skipping dummy batch during evaluation")
                    continue
                    
                # Move inputs to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Forward pass with mixed precision if enabled
                if fp16:
                    with torch.amp.autocast('cuda'):
                        text_embeddings, audio_embeddings = model(batch)
                        
                        # Get alignment scores if available
                        alignment_scores = getattr(model, 'last_alignment_scores', None)
                        
                        # Calculate loss with alignment awareness if possible
                        if isinstance(loss_fn, AlignmentAwareInfoNCE) and alignment_scores is not None:
                            loss_t2a = loss_fn(text_embeddings, audio_embeddings, alignment_scores)
                            loss_a2t = loss_fn(audio_embeddings, text_embeddings, alignment_scores)
                        else:
                            loss_t2a = loss_fn(text_embeddings, audio_embeddings)
                            loss_a2t = loss_fn(audio_embeddings, text_embeddings)
                        
                        loss = (loss_t2a + loss_a2t) / 2.0
                else:
                    # Forward pass
                    text_embeddings, audio_embeddings = model(batch)
                    
                    # Get alignment scores if available
                    alignment_scores = getattr(model, 'last_alignment_scores', None)
                    
                    # Calculate loss with alignment awareness if possible
                    if isinstance(loss_fn, AlignmentAwareInfoNCE) and alignment_scores is not None:
                        loss_t2a = loss_fn(text_embeddings, audio_embeddings, alignment_scores)
                        loss_a2t = loss_fn(audio_embeddings, text_embeddings, alignment_scores)
                    else:
                        loss_t2a = loss_fn(text_embeddings, audio_embeddings)
                        loss_a2t = loss_fn(audio_embeddings, text_embeddings)
                    
                    loss = (loss_t2a + loss_a2t) / 2.0
                
                # Calculate similarities between matching pairs
                similarities = torch.sum(text_embeddings * audio_embeddings, dim=1)
                all_similarities.extend(similarities.cpu().numpy())
                
                # Separate clean and corrupted samples for analysis
                is_corrupted = batch["is_corrupted"].cpu().bool()
                if torch.any(is_corrupted):
                    corrupt_similarities.extend(similarities[is_corrupted].cpu().numpy())
                if torch.any(~is_corrupted):
                    clean_similarities.extend(similarities[~is_corrupted].cpu().numpy())
                
                # Update metrics
                batch_size = batch["input_ids"].shape[0]
                sample_count += batch_size
                total_loss += loss.item() * batch_size
                
                # Calculate averages for progress bar
                avg_loss = total_loss / sample_count if sample_count > 0 else float('inf')
                avg_clean_sim = np.mean(clean_similarities) if clean_similarities else 0
                avg_corrupt_sim = np.mean(corrupt_similarities) if corrupt_similarities else 0
                sim_diff = avg_clean_sim - avg_corrupt_sim
                
                # Update progress bar with more info
                progress_bar.set_postfix({
                    "loss": f"{avg_loss:.4f}", 
                    "clean_sim": f"{avg_clean_sim:.3f}",
                    "corrupt_sim": f"{avg_corrupt_sim:.3f}",
                    "diff": f"{sim_diff:.3f}"
                })
                
                # Clear CUDA cache periodically
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"Error in evaluation batch {batch_idx}: {str(e)}")
                # Continue with next batch
                continue
    
    # If no valid samples were processed
    if sample_count == 0:
        logger.warning(f"No valid samples were processed during {split} evaluation")
        return {
            "loss": float('inf'),
            "avg_similarity": 0.0,
            "median_similarity": 0.0,
            "std_similarity": 0.0,
            "clean_similarity": 0.0,
            "corrupt_similarity": 0.0,
            "similarity_gap": 0.0
        }, []
    
    # Calculate metrics
    avg_similarity = np.mean(all_similarities) if all_similarities else 0.0
    std_similarity = np.std(all_similarities) if all_similarities else 0.0
    median_similarity = np.median(all_similarities) if all_similarities else 0.0
    
    # Calculate clean vs corrupted metrics
    avg_clean = np.mean(clean_similarities) if clean_similarities else 0.0
    avg_corrupt = np.mean(corrupt_similarities) if corrupt_similarities else 0.0
    similarity_gap = avg_clean - avg_corrupt
    
    # Print summary
    logger.info(f"{split} metrics:")
    logger.info(f"  Loss: {total_loss / sample_count:.4f}")
    logger.info(f"  Average similarity: {avg_similarity:.4f}")
    logger.info(f"  Median similarity: {median_similarity:.4f}")
    logger.info(f"  Clean sample similarity: {avg_clean:.4f}")
    logger.info(f"  Corrupted sample similarity: {avg_corrupt:.4f}")
    logger.info(f"  Similarity gap (clean - corrupt): {similarity_gap:.4f}")
    
    # Return metrics
    metrics = {
        "loss": total_loss / sample_count,
        "avg_similarity": avg_similarity,
        "median_similarity": median_similarity,
        "std_similarity": std_similarity,
        "clean_similarity": avg_clean,
        "corrupt_similarity": avg_corrupt,
        "similarity_gap": similarity_gap
    }
    
    return metrics, all_similarities


# ===== VISUALIZATION FUNCTIONS =====

def visualize_alignment(audio_file, text, model, processor, output_path=None):
    """
    Visualize the alignment between audio and text.
    """
    import matplotlib.pyplot as plt
    import librosa
    
    # Process text and audio
    text_inputs = processor.process_text(text)
    audio_inputs = processor.process_audio(audio_file)
    
    # Create batch with one item
    batch = {
        "input_ids": text_inputs["input_ids"].unsqueeze(0).to(device),
        "attention_mask": text_inputs["attention_mask"].unsqueeze(0).to(device),
        "input_features": audio_inputs["input_features"].unsqueeze(0).to(device),
        "attention_mask_audio": audio_inputs.get("attention_mask_audio", 
                              torch.ones(1, audio_inputs["input_features"].shape[0], 
                                        device=device))
    }
    
    # Forward pass
    with torch.no_grad():
        text_embeddings, audio_embeddings = model(batch)
    
    # Get alignment matrix
    alignment_matrix = model.last_alignment_matrix[0].cpu().numpy()
    
    # Get token list (decode input_ids)
    tokens = processor.tokenizer.convert_ids_to_tokens(
        batch["input_ids"][0].cpu().numpy())
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(alignment_matrix, aspect='auto', cmap='viridis')
    
    # Label axes
    ax.set_xlabel('Audio Frames')
    ax.set_ylabel('Text Tokens')
    
    # Set ticks for text tokens
    ax.set_yticks(range(len(tokens)))
    ax.set_yticklabels(tokens)
    
    # Add colorbar
    plt.colorbar(im, ax=ax)
    
    # Add title
    plt.title('Word-Level Audio-Text Alignment')
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()
    
    return alignment_matrix, tokens


def plot_similarity_distributions(clean_similarities, corrupt_similarities, output_path=None):
    """
    Plot histograms of similarities for clean and corrupted samples.
    """
    plt.figure(figsize=(10, 6))
    
    plt.hist(clean_similarities, alpha=0.7, bins=30, label='Clean Samples', color='green')
    plt.hist(corrupt_similarities, alpha=0.7, bins=30, label='Corrupted Samples', color='red')
    
    plt.axvline(np.mean(clean_similarities), color='green', linestyle='dashed', 
               linewidth=2, label=f'Clean Mean: {np.mean(clean_similarities):.3f}')
    plt.axvline(np.mean(corrupt_similarities), color='red', linestyle='dashed', 
               linewidth=2, label=f'Corrupt Mean: {np.mean(corrupt_similarities):.3f}')
    
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.title('Distribution of Similarities for Clean vs Corrupted Samples')
    plt.legend()
    plt.grid(alpha=0.3)
    
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()


# ===== MAIN TRAINING FUNCTION =====

def train_and_evaluate_model(
    train_dataset,
    val_dataset,
    test_dataset,
    output_dir,
    text_model_name="sentence-transformers/all-roberta-large-v1",
    audio_model_name="facebook/w2v-bert-2.0",
    max_text_length=128,
    batch_size=16,
    num_epochs=10,
    learning_rate=5e-5,
    weight_decay=0.01,
    temperature=0.07,
    num_warmup_steps=0,
    projection_dim=768,
    use_cross_modal=True,
    use_attentive_pooling=True,
    use_word_alignment=True,  # New default: use word-level alignment
    save_every=1,
    accumulation_steps=4,  # Gradient accumulation steps
    fp16=True,  # Use mixed precision training
    freeze_encoders=False,  # Default to False as recommended
    use_bucketing=False,   # Whether to use length-based bucketing
    seed=42,
    max_audio_len=160000  # Maximum audio length in samples
):
    """
    Train the audio-text embedding model and evaluate on validation and test sets.
    Enhanced with better architecture, training optimizations, and word-level alignment.
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging to file
    file_handler = logging.FileHandler(os.path.join(output_dir, "training.log"))
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)
    
    # Log training parameters
    logger.info("Training with parameters:")
    logger.info(f"  Text model: {text_model_name}")
    logger.info(f"  Audio model: {audio_model_name}")
    logger.info(f"  Freeze encoders: {freeze_encoders}")
    logger.info(f"  Use cross-modal attention: {use_cross_modal}")
    logger.info(f"  Use attentive pooling: {use_attentive_pooling}")
    logger.info(f"  Use word-level alignment: {use_word_alignment}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Gradient accumulation steps: {accumulation_steps}")
    logger.info(f"  Effective batch size: {batch_size * accumulation_steps}")
    logger.info(f"  Mixed precision training: {fp16}")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  Temperature: {temperature}")
    logger.info(f"  Projection dimension: {projection_dim}")
    logger.info(f"  Training samples: {len(train_dataset)}")
    logger.info(f"  Validation samples: {len(val_dataset)}")
    logger.info(f"  Test samples: {len(test_dataset)}")
    logger.info(f"  Max audio length: {max_audio_len} samples ({max_audio_len/16000:.2f} seconds at 16kHz)")
    
    # Initialize tokenizer and feature extractor
    logger.info("Loading tokenizer and feature extractor...")
    tokenizer = AutoTokenizer.from_pretrained(text_model_name)
    feature_extractor = AutoFeatureExtractor.from_pretrained(audio_model_name)  
    
    # Check what the feature extractor returns
    dummy_audio = np.zeros(1000, dtype=np.float32)
    dummy_features = feature_extractor(
        dummy_audio, 
        sampling_rate=16000,
        return_tensors="pt"
    )
    logger.info(f"Feature extractor output keys: {list(dummy_features.keys())}")
    
    # Create dataset wrappers
    logger.info("Creating datasets...")
    train_dataset_wrapper = CommonVoiceDataset(
        train_dataset, tokenizer, feature_extractor, max_text_length,
        max_audio_length=max_audio_len,
        add_corrupted_examples=True,  # Enable adding corrupted examples
        corruption_probability=0.2     # 20% chance of corruption
    )
    
    val_dataset_wrapper = CommonVoiceDataset(
        val_dataset, tokenizer, feature_extractor, max_text_length,
        max_audio_length=max_audio_len,
        add_corrupted_examples=True,
        corruption_probability=0.5  # Higher corruption rate in validation for better testing
    )
    
    test_dataset_wrapper = CommonVoiceDataset(
        test_dataset, tokenizer, feature_extractor, max_text_length,
        max_audio_length=max_audio_len,
        add_corrupted_examples=True,
        corruption_probability=0.5  # Higher corruption rate in test for better evaluation
    )
    
    # Create data loaders with custom collate function
    logger.info("Creating data loaders...")
    
    train_loader = DataLoader(
        train_dataset_wrapper,
        batch_size=batch_size,
        shuffle=True,
        num_workers=12,
        pin_memory=True,
        drop_last=True,
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset_wrapper,
        batch_size=batch_size,
        shuffle=False,
        num_workers=12,
        pin_memory=True,
        drop_last=False,
        collate_fn=custom_collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset_wrapper,
        batch_size=batch_size,
        shuffle=False,
        num_workers=12,
        pin_memory=True,
        drop_last=False,
        collate_fn=custom_collate_fn
    )
    
    # Debug: Check the first batch to verify dimensions
    try:
        logger.info("Checking a sample batch...")
        sample_batch = next(iter(train_loader))
        for k, v in sample_batch.items():
            if isinstance(v, torch.Tensor):
                logger.info(f"  {k}: {v.shape}")
            else:
                logger.info(f"  {k}: {type(v)}")
    except Exception as e:
        logger.warning(f"Error checking sample batch: {str(e)}")
    
    # Initialize enhanced model
    logger.info("Initializing model...")
    model = EnhancedAudioTextModel(
        text_model_name=text_model_name,
        audio_model_name=audio_model_name,
        projection_dim=projection_dim,
        use_cross_modal=use_cross_modal,
        use_attentive_pooling=use_attentive_pooling,
        use_word_alignment=use_word_alignment,
        freeze_encoders=freeze_encoders
    )
    
    model = model.to(device)
    
    # Initialize mixed precision training if requested
    scaler = torch.amp.GradScaler('cuda') if fp16 else None
    
    # Initialize optimizer - only include parameters that require gradients
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Initialize alignment-aware InfoNCE loss with temperature parameter
    loss_fn = AlignmentAwareInfoNCE(temperature=temperature, alignment_weight=0.5)
    
    # Initialize learning rate scheduler
    total_steps = len(train_loader) * num_epochs // accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps
    )
    
    # Initialize lists to store metrics
    train_metrics_history = []
    val_metrics_history = []
    clean_similarities_history = []
    corrupt_similarities_history = []
    best_val_loss = float("inf")
    best_similarity_gap = 0.0
    
    # Training loop
    logger.info(f"Starting training for {num_epochs} epochs")
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        
        try:
            # Train
            train_metrics = train_epoch(
                model, 
                train_loader, 
                optimizer, 
                scheduler, 
                loss_fn, 
                device, 
                epoch, 
                accumulation_steps=accumulation_steps, 
                scaler=scaler
            )
            train_metrics_history.append(train_metrics)
            
            # Validate
            val_metrics, val_similarities = evaluate(
                model, 
                val_loader, 
                loss_fn, 
                device, 
                epoch, 
                "Validation", 
                fp16=fp16
            )
            val_metrics_history.append(val_metrics)
            
            # Extract and store similarities for analysis
            clean_similarities_history.append(val_metrics["clean_similarity"])
            corrupt_similarities_history.append(val_metrics["corrupt_similarity"])
            
            epoch_time = time.time() - start_time
            
            logger.info(
                f"Epoch {epoch}/{num_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Clean Sim: {val_metrics['clean_similarity']:.4f}, "
                f"Corrupt Sim: {val_metrics['corrupt_similarity']:.4f}, "
                f"Gap: {val_metrics['similarity_gap']:.4f}, "
                f"Time: {epoch_time:.2f}s"
            )
            
            # Save model checkpoint if validation loss improves
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                logger.info(f"New best validation loss: {best_val_loss:.4f}")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_metrics": train_metrics,
                        "val_metrics": val_metrics,
                        "temperature": temperature,
                        "projection_dim": projection_dim,
                        "use_cross_modal": use_cross_modal,
                        "use_attentive_pooling": use_attentive_pooling,
                        "use_word_alignment": use_word_alignment,
                    },
                    os.path.join(output_dir, "best_model_loss.pt")
                )
            
            # Also save if similarity gap improves (better distinction between clean and corrupted)
            if val_metrics["similarity_gap"] > best_similarity_gap:
                best_similarity_gap = val_metrics["similarity_gap"]
                logger.info(f"New best similarity gap: {best_similarity_gap:.4f}")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_metrics": train_metrics,
                        "val_metrics": val_metrics,
                        "temperature": temperature,
                        "projection_dim": projection_dim,
                        "use_cross_modal": use_cross_modal,
                        "use_attentive_pooling": use_attentive_pooling,
                        "use_word_alignment": use_word_alignment,
                    },
                    os.path.join(output_dir, "best_model_gap.pt")
                )
            
            # Save model checkpoint periodically
            if epoch % save_every == 0:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_metrics": train_metrics,
                        "val_metrics": val_metrics,
                        "temperature": temperature,
                        "projection_dim": projection_dim,
                        "use_cross_modal": use_cross_modal,
                        "use_attentive_pooling": use_attentive_pooling,
                        "use_word_alignment": use_word_alignment,
                    },
                    os.path.join(output_dir, f"checkpoint_epoch_{epoch}.pt")
                )
            
            # Plot and save similarity distributions periodically
            if epoch % 5 == 0 or epoch == num_epochs:
                # Generate validation dataset with clean and corrupted examples
                clean_sim = []
                corrupt_sim = []
                
                # Get detailed similarities from validation set
                model.eval()
                with torch.no_grad():
                    for batch in val_loader:
                        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                                for k, v in batch.items()}
                        
                        text_embeddings, audio_embeddings = model(batch)
                        similarities = torch.sum(text_embeddings * audio_embeddings, dim=1).cpu()
                        
                        is_corrupted = batch["is_corrupted"].cpu().bool()
                        if torch.any(is_corrupted):
                            corrupt_sim.extend(similarities[is_corrupted].numpy())
                        if torch.any(~is_corrupted):
                            clean_sim.extend(similarities[~is_corrupted].numpy())
                
                # Plot distributions
                plot_similarity_distributions(
                    clean_sim, 
                    corrupt_sim, 
                    output_path=os.path.join(output_dir, f"similarity_dist_epoch_{epoch}.png")
                )
                
                # Create and save visualization of clean vs corrupted performance over epochs
                plt.figure(figsize=(12, 6))
                epochs = list(range(1, epoch + 1))
                plt.plot(epochs, clean_similarities_history, 'g-', label='Clean Samples')
                plt.plot(epochs, corrupt_similarities_history, 'r-', label='Corrupted Samples')
                plt.fill_between(epochs, clean_similarities_history, corrupt_similarities_history, 
                                color='lightgreen', alpha=0.3, label='Similarity Gap')
                plt.xlabel('Epoch')
                plt.ylabel('Average Similarity')
                plt.title('Clean vs Corrupted Sample Performance Over Training')
                plt.legend()
                plt.grid(alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "clean_corrupt_progress.png"))
                plt.close()
            
        except Exception as e:
            logger.error(f"Error in epoch {epoch}: {str(e)}")
            continue
    
    logger.info("Training completed!")
    
    # Save final model
    torch.save(
        {
            "epoch": num_epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_metrics": train_metrics if 'train_metrics' in locals() else None,
            "val_metrics": val_metrics if 'val_metrics' in locals() else None,
            "temperature": temperature,
            "projection_dim": projection_dim,
            "use_cross_modal": use_cross_modal,
            "use_attentive_pooling": use_attentive_pooling,
            "use_word_alignment": use_word_alignment,
        },
        os.path.join(output_dir, "final_model.pt")
    )
    
    # Evaluate on test set using both best models
    logger.info("Evaluating best models on test set...")
    
    # Results dictionary to store both model evaluations
    test_results = {}
    
    # Test the model with best loss
    best_loss_model_path = os.path.join(output_dir, "best_model_loss.pt")
    if os.path.exists(best_loss_model_path):
        checkpoint = torch.load(best_loss_model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded best loss model from epoch {checkpoint['epoch']}")
        
        test_metrics, test_similarities = evaluate(model, test_loader, loss_fn, device, split="Test (Best Loss)", fp16=fp16)
        test_results["best_loss_model"] = test_metrics
        
        # Plot test set distributions for the best loss model
        clean_sim = []
        corrupt_sim = []
        
        # Get detailed similarities from test set
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                text_embeddings, audio_embeddings = model(batch)
                similarities = torch.sum(text_embeddings * audio_embeddings, dim=1).cpu()
                
                is_corrupted = batch["is_corrupted"].cpu().bool()
                if torch.any(is_corrupted):
                    corrupt_sim.extend(similarities[is_corrupted].numpy())
                if torch.any(~is_corrupted):
                    clean_sim.extend(similarities[~is_corrupted].numpy())
        
        # Plot and save distributions
        plot_similarity_distributions(
            clean_sim, 
            corrupt_sim, 
            output_path=os.path.join(output_dir, "test_similarity_dist_best_loss.png")
        )
    else:
        logger.warning("Best loss model not found")
    
    # Test the model with best similarity gap
    best_gap_model_path = os.path.join(output_dir, "best_model_gap.pt")
    if os.path.exists(best_gap_model_path):
        checkpoint = torch.load(best_gap_model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded best gap model from epoch {checkpoint['epoch']}")
        
        test_metrics, test_similarities = evaluate(model, test_loader, loss_fn, device, split="Test (Best Gap)", fp16=fp16)
        test_results["best_gap_model"] = test_metrics
        
        # Plot test set distributions for the best gap model
        clean_sim = []
        corrupt_sim = []
        
        # Get detailed similarities from test set
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                text_embeddings, audio_embeddings = model(batch)
                similarities = torch.sum(text_embeddings * audio_embeddings, dim=1).cpu()
                
                is_corrupted = batch["is_corrupted"].cpu().bool()
                if torch.any(is_corrupted):
                    corrupt_sim.extend(similarities[is_corrupted].numpy())
                if torch.any(~is_corrupted):
                    clean_sim.extend(similarities[~is_corrupted].numpy())
        
        # Plot and save distributions
        plot_similarity_distributions(
            clean_sim, 
            corrupt_sim, 
            output_path=os.path.join(output_dir, "test_similarity_dist_best_gap.png")
        )
    else:
        logger.warning("Best gap model not found")
    
    # Save test metrics
    with open(os.path.join(output_dir, "test_metrics.json"), "w") as f:
        json.dump(test_results, f, indent=2)
    
    logger.info("Evaluation completed!")
    
    # Print summary of test results
    for model_name, metrics in test_results.items():
        logger.info(f"Test results for {model_name}:")
        logger.info(f"  Loss: {metrics['loss']:.4f}")
        logger.info(f"  Clean Sample Similarity: {metrics['clean_similarity']:.4f}")
        logger.info(f"  Corrupted Sample Similarity: {metrics['corrupt_similarity']:.4f}")
        logger.info(f"  Similarity Gap: {metrics['similarity_gap']:.4f}")
    
    # Return both best models and test metrics
    return {
        "model": model,
        "test_metrics": test_results
    }


# ===== MAIN FUNCTION =====

def main():
    parser = argparse.ArgumentParser(description="Train audio-text embedding model on Common Voice")
    
    # Data parameters
    parser.add_argument("--data_dir", type=str, default="./common_voice_data",
                        help="Directory to store Common Voice data")
    parser.add_argument("--output_dir", type=str, default="./audio_text_model",
                        help="Directory to save model checkpoints and results")
    
    # Model parameters
    parser.add_argument("--text_model", type=str, default="sentence-transformers/all-roberta-large-v1",
                        help="Text encoder model name")
    parser.add_argument("--audio_model", type=str, default="facebook/w2v-bert-2.0",
                        help="Audio encoder model name")
    parser.add_argument("--projection_dim", type=int, default=1024,
                        help="Dimension of the shared embedding space")
    parser.add_argument("--no_cross_modal", action="store_true",
                        help="Disable cross-modal attention")
    parser.add_argument("--no_attentive_pooling", action="store_true",
                        help="Disable attentive pooling")
    parser.add_argument("--no_word_alignment", action="store_true",
                        help="Disable word-level alignment")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Training batch size")
    parser.add_argument("--num_epochs", type=int, default=15,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--temperature", type=float, default=0.07,
                        help="Temperature parameter for InfoNCE loss")
    parser.add_argument("--max_text_length", type=int, default=256,
                        help="Maximum token length for text")
    parser.add_argument("--warmup_steps", type=int, default=1000,
                        help="Number of warmup steps")
    parser.add_argument("--save_every", type=int, default=1,
                        help="Save model checkpoint every N epochs")
    parser.add_argument("--acc_steps", type=int, default=4,
                        help="Number of gradient accumulation steps")
    parser.add_argument("--no_fp16", action="store_true",
                        help="Disable mixed precision training")
    parser.add_argument("--no_freeze", action="store_true",
                        help="Don't freeze encoders (train entire model)")
    parser.add_argument("--bucket", action="store_true",
                        help="Use length-based bucketing for efficiency")
    parser.add_argument("--max_audio_len", type=int, default=480000,
                        help="Maximum audio length in samples (default 30s at 16kHz)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Enable debug logging if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        # Also output to console
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(console_handler)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load Common Voice dataset
    logger.info("Loading Common Voice dataset...")
    
    try:
        dataset = load_dataset("mozilla-foundation/common_voice_17_0", "pt")
        
        # Cast to Audio column with 16kHz sampling rate
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
        
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        logger.error("Make sure you have downloaded the Common Voice dataset correctly")
        exit(1)
    
    # Train and evaluate model
    logger.info("Starting model training and evaluation")
    results = train_and_evaluate_model(
        train_dataset=dataset["train"],
        val_dataset=dataset["validation"],
        test_dataset=dataset["test"],
        output_dir=args.output_dir,
        text_model_name=args.text_model,
        audio_model_name=args.audio_model,
        max_text_length=args.max_text_length,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        num_warmup_steps=args.warmup_steps,
        projection_dim=args.projection_dim,
        use_cross_modal=not args.no_cross_modal,
        use_attentive_pooling=not args.no_attentive_pooling,
        use_word_alignment=not args.no_word_alignment,
        save_every=args.save_every,
        accumulation_steps=args.acc_steps,
        fp16=not args.no_fp16,
        freeze_encoders=not args.no_freeze,
        use_bucketing=args.bucket,
        seed=args.seed,
        max_audio_len=args.max_audio_len
    )
    
    logger.info("All tasks completed!")
    

if __name__ == "__main__":
    main()