#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced Audio-Text Embedding Model Trainer with Word-Level Alignment
This implementation addresses issues with high similarity for incorrect transcripts
by adding word-level alignment mechanisms.
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
import matplotlib.pyplot as plt
from dotenv import load_dotenv
load_dotenv()   

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

# Configure environment settings
print(f"Using GPU: {torch.cuda.get_device_name(0)}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

def print_memory_summary():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated()/1024**3
        reserved = torch.cuda.memory_reserved()/1024**3
        max_allocated = torch.cuda.max_memory_allocated()/1024**3
        total = torch.cuda.get_device_properties(0).total_memory/1024**3
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Max: {max_allocated:.2f}GB, Total: {total:.2f}GB")

print(f"Using GPU: {torch.cuda.get_device_name(0)}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print_memory_summary()

# Verify CUDA allocation config was applied
print(f"CUDA Alloc Config: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'Not set')}")

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
    def __init__(self, text_hidden_dim, audio_hidden_dim, alignment_dim, num_heads=4, dropout=0.1):
        super().__init__()
        
        self.text_hidden_dim = text_hidden_dim
        self.audio_hidden_dim = audio_hidden_dim
        self.alignment_dim = alignment_dim
        self.num_heads = num_heads
        
        # Projection layers to create query (text) and key/value (audio) representations
        self.text_projection = nn.Linear(text_hidden_dim, alignment_dim)
        self.audio_projection = nn.Linear(audio_hidden_dim, alignment_dim)
        
        # Multi-head attention for alignment
        self.alignment_attention = nn.MultiheadAttention(
            embed_dim=alignment_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection and layer norm
        self.output_projection = nn.Linear(alignment_dim, alignment_dim)
        self.layer_norm = nn.LayerNorm(alignment_dim)
        
        # Alignment confidence scorer (predicts how well each word aligns with audio)
        self.alignment_confidence = nn.Sequential(
            nn.Linear(alignment_dim, alignment_dim // 2),
            nn.ReLU(),
            nn.Linear(alignment_dim // 2, 1)
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
    - Partial encoder unfreezing support
    """
    def __init__(
        self,
        text_model_name="sentence-transformers/all-roberta-large-v1",
        audio_model_name="facebook/w2v-bert-2.0",
        projection_dim=768,
        text_embedding_dim=768,
        audio_embedding_dim=1024,
        dropout=0.1,
        use_cross_modal=True,
        use_attentive_pooling=True,
        use_word_alignment=False,  # New parameter
        freeze_encoders="partial",    # Changed to string: "full", "partial", "none"
        text_layers_to_unfreeze=5,    # New parameter for partial unfreezing
        audio_layers_to_unfreeze=5,   # New parameter for partial unfreezing
    ):
        super().__init__()
        
        # Load pre-trained models
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.audio_encoder = AutoModel.from_pretrained(audio_model_name)
        self.text_hidden_dim = self.text_encoder.config.hidden_size
        self.audio_hidden_dim = self.audio_encoder.config.hidden_size
        logger.info(f"Text encoder hidden dim: {self.text_hidden_dim}")
        logger.info(f"Audio encoder hidden dim: {self.audio_hidden_dim}")
        # Save configuration
        self.projection_dim = projection_dim
        self.use_cross_modal = use_cross_modal
        self.use_attentive_pooling = use_attentive_pooling
        self.use_word_alignment = use_word_alignment
        
        # Enhanced partial freezing logic
        if freeze_encoders == "full":
            logger.info("Freezing text and audio encoders completely")
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            for param in self.audio_encoder.parameters():
                param.requires_grad = False
                
        elif freeze_encoders == "partial":
            logger.info(f"Partial freezing: unfreezing last {text_layers_to_unfreeze} text layers and {audio_layers_to_unfreeze} audio layers")
            
            # Freeze text encoder except last N layers
            if hasattr(self.text_encoder, 'encoder') and hasattr(self.text_encoder.encoder, 'layer'):
                # RoBERTa/BERT-style architecture
                total_text_layers = len(self.text_encoder.encoder.layer)
                for i, layer in enumerate(self.text_encoder.encoder.layer):
                    if i < total_text_layers - text_layers_to_unfreeze:
                        for param in layer.parameters():
                            param.requires_grad = False
                    else:
                        logger.info(f"Unfreezing text encoder layer {i}")
                
                # Also unfreeze the pooler if it exists
                if hasattr(self.text_encoder, 'pooler') and self.text_encoder.pooler is not None:
                    for param in self.text_encoder.pooler.parameters():
                        param.requires_grad = True
            else:
                # Fallback: freeze embeddings, unfreeze rest
                logger.info("Text encoder structure not recognized, using fallback partial freezing")
                if hasattr(self.text_encoder, 'embeddings'):
                    for param in self.text_encoder.embeddings.parameters():
                        param.requires_grad = False
            
            # Freeze audio encoder except last N layers  
            if hasattr(self.audio_encoder, 'encoder') and hasattr(self.audio_encoder.encoder, 'layers'):
                # Wav2Vec2-style architecture
                total_audio_layers = len(self.audio_encoder.encoder.layers)
                for i, layer in enumerate(self.audio_encoder.encoder.layers):
                    if i < total_audio_layers - audio_layers_to_unfreeze:
                        for param in layer.parameters():
                            param.requires_grad = False
                    else:
                        logger.info(f"Unfreezing audio encoder layer {i}")
                        
                # Unfreeze feature projection for better adaptation
                if hasattr(self.audio_encoder, 'feature_projection'):
                    for param in self.audio_encoder.feature_projection.parameters():
                        param.requires_grad = True
                        
            elif hasattr(self.audio_encoder, 'feature_projection'):
                # Alternative: unfreeze feature projection and some transformer layers
                logger.info("Audio encoder using alternative partial freezing strategy")
                for param in self.audio_encoder.feature_projection.parameters():
                    param.requires_grad = True
                if hasattr(self.audio_encoder, 'encoder'):
                    # Unfreeze last few transformer layers
                    if hasattr(self.audio_encoder.encoder, 'layers'):
                        total_layers = len(self.audio_encoder.encoder.layers)
                        for i in range(max(0, total_layers - audio_layers_to_unfreeze), total_layers):
                            for param in self.audio_encoder.encoder.layers[i].parameters():
                                param.requires_grad = True
            else:
                # Fallback for unknown architecture
                logger.info("Audio encoder structure not recognized, using fallback partial freezing")
                # Just unfreeze the last part of parameters
                all_params = list(self.audio_encoder.parameters())
                unfreeze_count = len(all_params) // 3  # Unfreeze last third
                for param in all_params[-unfreeze_count:]:
                    param.requires_grad = True
                        
        elif freeze_encoders == "none" or freeze_encoders is False:
            logger.info("All encoder parameters unfrozen")
            # Keep all parameters trainable (default behavior)
            
        else:
            # Backward compatibility: if freeze_encoders is True, freeze completely
            logger.info("Freezing text and audio encoders completely (legacy mode)")
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
            self.text_seq_to_projection = nn.Linear(
                self.text_hidden_dim,  # Dynamic: actual text encoder hidden size
                projection_dim         # Target projection dimension
            )
            self.audio_seq_to_projection = nn.Linear(
                self.audio_hidden_dim, # Dynamic: actual audio encoder hidden size
                projection_dim         # Target projection dimension
            )

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
                text_hidden_dim=self.text_hidden_dim,      # 768 for RoBERTa
                audio_hidden_dim=self.audio_hidden_dim,    # 1024 for w2v-bert
                alignment_dim=projection_dim,              # 768 (your shared space)
                dropout=dropout
            )
                    
            # Alignment scores are used directly in the loss function
            # No need for additional weighting layers
        
        # Log number of trainable parameters
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"Model initialized with {trainable_params:,} trainable parameters out of {total_params:,} total")


    @staticmethod
    def compute_pos_neg_embeddings(model, batch):
        """
        Runs both clean (pos) and corrupt (neg) transcripts
        through your full encode → cross-modal → alignment pipeline.
        Returns normalized [B,D] embeddings for pos, neg, and audio.
        Also leaves model.last_alignment_scores set for the pos side.
        """
        # 1) base encode_text + encode_audio
        # text_encoder returns (proj, last_hidden_state)
        txt_pos_proj, txt_pos_hidden = model.encode_text(
            batch["input_ids_pos"], batch["attention_mask_pos"]
        )
        txt_neg_proj, txt_neg_hidden = model.encode_text(
            batch["input_ids_neg"], batch["attention_mask_neg"]
        )
        aud_proj, aud_hidden = model.encode_audio(
            batch["input_values"], batch["attention_mask_audio"]
        )

        # 2) cross-modal fusion (if enabled)
        if model.use_cross_modal:
            # text→audio on pos
            txt_pos_fused, aud_fused = model.apply_cross_modal_attention(
                txt_pos_proj,
                txt_pos_hidden,
                batch["attention_mask_pos"],
                aud_proj,
                aud_hidden,
                batch["attention_mask_audio"],
            )

            # text→audio on neg
            txt_neg_fused, _ = model.apply_cross_modal_attention(
                txt_neg_proj,
                txt_neg_hidden,
                batch["attention_mask_neg"],
                aud_proj,
                aud_hidden,
                batch["attention_mask_audio"],
            )

        else:
            txt_pos_fused = txt_pos_proj
            txt_neg_fused = txt_neg_proj
            aud_fused     = aud_proj

        # 3) word-level alignment (if enabled)
        if model.use_word_alignment:
            aligned_pos, align_scores, _ = model.word_level_alignment(
                text_hidden_states=txt_pos_hidden,
                audio_hidden_states=aud_hidden,
                text_attention_mask=batch["attention_mask_pos"],
                audio_attention_mask=batch["attention_mask_audio"],
            )
            # store for loss fn - this is where alignment should influence training
            model.last_alignment_scores = align_scores

        # 4) normalize all
        txt_pos_norm = F.normalize(txt_pos_fused, p=2, dim=1)
        txt_neg_norm = F.normalize(txt_neg_fused, p=2, dim=1)
        aud_norm     = F.normalize(aud_fused,     p=2, dim=1)

        return txt_pos_norm, txt_neg_norm, aud_norm

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
    
    def encode_audio(self, input_values, attention_mask=None):
        """
        Encode audio inputs with enhanced processing.
        """
        # Adapt input name based on the model's expected input
        try:
            # First try with input_values (older models)
            outputs = self.audio_encoder(
                input_values=input_values,
                attention_mask=attention_mask
            )
        except TypeError:
            try:
                # Then try with input_features (newer models)
                outputs = self.audio_encoder(
                    input_features=input_values,
                    attention_mask=attention_mask
                )
            except TypeError as e:
                # Fallback to a generic approach
                logger.warning(f"Using fallback approach for audio encoder: {e}")
                outputs = self.audio_encoder(input_values)
        
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
    
        audio_proj_seq = self.audio_seq_to_projection(audio_hidden)  
        text_proj_seq = self.text_seq_to_projection(text_hidden)  

        try:
            text_attended = self.text_to_audio_attention(
                text_projected.unsqueeze(1),
                audio_proj_seq,
                audio_mask
            ).squeeze(1)
        except Exception as e:
            print(f"Error in text_to_audio_attention: {e}")
            raise
        
        try:
            audio_attended = self.audio_to_text_attention(
                audio_projected.unsqueeze(1),
                text_proj_seq,
                text_mask
            ).squeeze(1)
        except Exception as e:
            print(f"Error in audio_to_text_attention: {e}")
            raise
    
        
        # Fuse - now both are projection_dim size
        text_fused = self.text_fusion(torch.cat([text_projected, text_attended], dim=1))
        audio_fused = self.audio_fusion(torch.cat([audio_projected, audio_attended], dim=1))
        

        
        return text_fused, audio_fused
    
    def forward(self, batch):
        """
        Forward pass with enhanced processing and word-level alignment.
        Always uses compute_pos_neg_embeddings for consistency.
        """
        # Check if batch has the expected format
        if "input_ids_pos" not in batch or "input_ids_neg" not in batch:
            raise ValueError(
                "Batch must contain 'input_ids_pos' and 'input_ids_neg'. "
                "Got keys: {}".format(list(batch.keys()))
            )
        
        # Always use the same processing path for both training and evaluation
        return EnhancedAudioTextModel.compute_pos_neg_embeddings(self, batch)


# ===== ALIGNMENT-AWARE LOSS FUNCTION =====

class AlignmentAwareInfoNCE(torch.nn.Module):
    """
    2-way InfoNCE implemented as CE over [s_pos, s_neg],
    with optional word-alignment weighting and a corrupt-penalty.
    """
    def __init__(self, temperature=0.1, alignment_weight=0.3, corrupt_gamma=0.35):
        super().__init__()
        self.temperature = temperature
        self.alignment_weight = alignment_weight
        self.corrupt_gamma = corrupt_gamma

    def forward(self, s_pos: torch.Tensor, s_neg: torch.Tensor, alignment_scores: torch.Tensor=None):
        """
        Args:
          s_pos:  [B] cosine(audio, text_pos)
          s_neg:  [B] cosine(audio, text_neg)
          alignment_scores: [B, text_len] per-token alignment (optional)
        Returns:
          scalar loss
        """
        # 1) build logits and targets
        logits = torch.stack([s_pos, s_neg], dim=1) / self.temperature  # [B,2]
        targets = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)  # all zeros → pos is correct

        # 2) per-sample CE
        per_sample = F.cross_entropy(logits, targets, reduction="none")  # [B]

        # 3) alignment weighting (optional)
        if alignment_scores is not None:
            # mean over tokens → [B]
            mean_align = alignment_scores.mean(dim=1)
            factor = 1.0 - torch.sigmoid(mean_align) * self.alignment_weight
            per_sample = per_sample * factor

        loss = per_sample.mean()

        # 4) corrupt-penalty (optional)
        if self.corrupt_gamma > 0:
            loss = loss + self.corrupt_gamma * F.relu(s_neg).mean()
            
        return loss


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
        item = self.dataset[idx]
        speech_array = item["audio"]["array"]
        # 1) get clean + corrupted text
        clean_text = item["sentence"]
        corrupt_text = self.create_corrupted_transcript(clean_text)

        # 2) tokenize both
        pos_enc = self.tokenizer(
            clean_text,
            max_length=self.max_text_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        neg_enc = self.tokenizer(
            corrupt_text,
            max_length=self.max_text_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # 3) extract audio features
        audio_features = self.feature_extractor(
            speech_array,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
        )
        # use whichever key exists
        if "input_features" in audio_features:
            waveform = audio_features["input_features"].squeeze(0)
        else:
            waveform = audio_features["input_values"].squeeze(0)
        audio_mask = audio_features.get("attention_mask", None)
   
        return {
            "input_ids_pos":      pos_enc["input_ids"].squeeze(0),
            "attention_mask_pos": pos_enc["attention_mask"].squeeze(0),
            "input_ids_neg":      neg_enc["input_ids"].squeeze(0),
            "attention_mask_neg": neg_enc["attention_mask"].squeeze(0),
            "input_values":       waveform,
            "attention_mask_audio": audio_mask.squeeze(0) if audio_mask is not None else None,
        }


# ===== CUSTOM COLLATE FUNCTION =====

def custom_collate_fn(batch):
    # drop any items that returned None
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    # 1) pad clean (pos) text
    pos_ids   = [b["input_ids_pos"]      for b in batch]
    pos_mask  = [b["attention_mask_pos"] for b in batch]
    input_ids_pos      = pad_sequence(pos_ids,  batch_first=True, padding_value=0)
    attention_mask_pos = pad_sequence(pos_mask, batch_first=True, padding_value=0)

    # 2) pad corrupted (neg) text
    neg_ids   = [b["input_ids_neg"]      for b in batch]
    neg_mask  = [b["attention_mask_neg"] for b in batch]
    input_ids_neg      = pad_sequence(neg_ids,  batch_first=True, padding_value=0)
    attention_mask_neg = pad_sequence(neg_mask, batch_first=True, padding_value=0)

    # 3) pad audio (time × feat_dim)
    audios = [b["input_values"] for b in batch]
    max_t  = max(a.size(0) for a in audios)
    feat   = audios[0].size(1)
    B      = len(audios)
    padded_audio = torch.zeros((B, max_t, feat), dtype=audios[0].dtype)
    audio_mask   = torch.zeros((B, max_t), dtype=torch.long)
    for i, a in enumerate(audios):
        t = a.size(0)
        padded_audio[i, :t, :] = a
        audio_mask[i, :t]      = 1

    # 4) Create is_corrupted flags - zeros for clean samples in evaluation
    is_corrupted = torch.zeros(B, dtype=torch.long)

    return {
        "input_ids_pos":       input_ids_pos,
        "attention_mask_pos":  attention_mask_pos,
        "input_ids_neg":       input_ids_neg,
        "attention_mask_neg":  attention_mask_neg,
        "input_values":        padded_audio,
        "attention_mask_audio": audio_mask,
        "is_corrupted":        is_corrupted,  # Add flag for evaluation
    }


def to_human_readable(cosine: torch.Tensor,
                      temperature: float = 0.1,
                      scale: str = "prob") -> torch.Tensor:
    """
    Convert raw cosine similarities (-1 … 1) to something intuitive.

    scale = "0to1" :  linear map  (cos + 1) / 2      ∈ [0,1]
    scale = "prob" :  sigmoid(cos / τ)               ∈ [0,1]
                      (needs the same τ as the loss)
    """
    if scale == "0to1":
        return (cosine + 1.) * 0.5
    elif scale == "prob":
        return torch.sigmoid(cosine / temperature)
    else:
        raise ValueError(f"Unknown scale '{scale}'. Use '0to1' or 'prob'.")

# ===== GRADIENT ACCUMULATION VALIDATION =====

def validate_gradient_accumulation(model, data_loader, accumulation_steps=4, device="cuda", test_batches=10):
    """
    Validate that gradient accumulation is working correctly by comparing
    accumulated gradients vs single large batch gradients.
    
    This function helps debug gradient accumulation issues.
    """
    logger.info(f"Validating gradient accumulation with {accumulation_steps} steps...")
    
    model.train()
    
    # Get a few batches for testing
    test_data = []
    for i, batch in enumerate(data_loader):
        if i >= test_batches:
            break
        test_data.append(batch)
    
    if len(test_data) < accumulation_steps:
        logger.warning(f"Not enough test batches ({len(test_data)}) for accumulation_steps ({accumulation_steps})")
        return
    
    # Method 1: Accumulate gradients step by step
    model.zero_grad()
    accumulated_loss = 0
    
    for i in range(accumulation_steps):
        batch = test_data[i]
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Forward pass
        txt_pos_norm, txt_neg_norm, aud_norm = EnhancedAudioTextModel.compute_pos_neg_embeddings(model, batch)
        s_pos = (aud_norm * txt_pos_norm).sum(dim=1)
        s_neg = (aud_norm * txt_neg_norm).sum(dim=1)
        
        # Dummy loss for testing (simple contrastive loss)
        loss = -torch.log(torch.sigmoid(s_pos - s_neg)).mean()
        scaled_loss = loss / accumulation_steps
        accumulated_loss += loss.item()
        
        # Backward pass (accumulate gradients)
        scaled_loss.backward()
    
    # Get accumulated gradients
    accumulated_grads = []
    for param in model.parameters():
        if param.grad is not None:
            accumulated_grads.append(param.grad.clone())
        else:
            accumulated_grads.append(None)
    
    # Method 2: Single large batch (concatenate batches)
    model.zero_grad()
    
    # This is a simplified test - in practice, you'd need to properly concatenate
    # the audio and text data, which is complex due to padding
    logger.info(f"Accumulated loss over {accumulation_steps} steps: {accumulated_loss:.6f}")
    logger.info(f"Average loss per step: {accumulated_loss/accumulation_steps:.6f}")
    
    # Check if gradients exist and are reasonable
    grad_norms = []
    for i, grad in enumerate(accumulated_grads):
        if grad is not None:
            grad_norm = grad.norm().item()
            grad_norms.append(grad_norm)
    
    logger.info(f"Gradient norms (first 5): {grad_norms[:5]}")
    logger.info(f"Max gradient norm: {max(grad_norms) if grad_norms else 0:.6f}")
    logger.info(f"Mean gradient norm: {np.mean(grad_norms) if grad_norms else 0:.6f}")
    
    if max(grad_norms) > 100:
        logger.warning("Very large gradients detected - consider lowering learning rate")
    elif max(grad_norms) < 1e-8:
        logger.warning("Very small gradients detected - consider increasing learning rate")
    else:
        logger.info("✓ Gradient magnitudes look reasonable")
    
    model.zero_grad()  # Clean up
    logger.info("Gradient accumulation validation completed")


# ===== TRAINING AND EVALUATION FUNCTIONS =====

def train_epoch(
    model,
    data_loader,
    optimizer,
    scheduler,
    loss_fn: AlignmentAwareInfoNCE,
    device,
    epoch: int,
    accumulation_steps: int = 4,
    scaler = None,
):
    """
    Train one epoch with the full model architecture including:
    - Cross-modal attention  
    - Word-level alignment
    - InfoNCE loss with alignment weighting
    - Fixed gradient accumulation implementation
    """
    model.train()
    total_loss = 0.0
    sample_count = 0
    clean_sims, corrupt_sims = [], []
    
    # Track actual optimizer steps for scheduler
    optimizer_steps = 0

    pbar = tqdm(data_loader, desc=f"Epoch {epoch} [Train]")
    
    # Initialize gradients
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(pbar):
        # 1) move to device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device, non_blocking=True)

        # 2) Determine if this is the last accumulation step
        is_accumulation_step = (batch_idx + 1) % accumulation_steps == 0
        is_last_batch = (batch_idx + 1) == len(data_loader)
        should_step = is_accumulation_step or is_last_batch

        def _forward():
            # Use the COMPLETE architecture via compute_pos_neg_embeddings
            # This includes: encoding → cross-modal attention → word alignment
            txt_pos_norm, txt_neg_norm, aud_norm = EnhancedAudioTextModel.compute_pos_neg_embeddings(model, batch)
            # Compute cosine similarities
            s_pos = (aud_norm * txt_pos_norm).sum(dim=1)  # [B]
            s_neg = (aud_norm * txt_neg_norm).sum(dim=1)  # [B]
            
            # Get alignment scores which were set by compute_pos_neg_embeddings
            alignment_scores = getattr(model, "last_alignment_scores", None)
            
            # Calculate loss
            loss = loss_fn(s_pos, s_neg, alignment_scores=alignment_scores)
            return loss, s_pos, s_neg

        # 3) Forward pass and backward (with proper gradient accumulation)
        if scaler is not None:
            with torch.amp.autocast(device_type="cuda"):
                loss, s_pos, s_neg = _forward()
                # Scale loss for gradient accumulation
                scaled_loss = loss / accumulation_steps
            scaler.scale(scaled_loss).backward()
        else:
            loss, s_pos, s_neg = _forward()
            # Scale loss for gradient accumulation  
            scaled_loss = loss / accumulation_steps
            scaled_loss.backward()

        # 4) Optimizer step (only when accumulation is complete)
        if should_step:
            if scaler is not None:
                # Unscale gradients for clipping
                scaler.unscale_(optimizer)
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                # Update weights
                scaler.step(optimizer)
                scaler.update()
            else:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                # Update weights
                optimizer.step()
            
            # Update learning rate scheduler
            scheduler.step()
            optimizer_steps += 1
            
            # Reset gradients
            optimizer.zero_grad()

        # 5) Track similarities on CPU (use original loss for reporting)

        s_pos_hr= to_human_readable(s_pos, temperature=0.1, scale="prob")
        s_neg_hr= to_human_readable(s_neg, temperature=0.1, scale="prob")
        clean_sims.extend(s_pos_hr.detach().cpu().tolist())
        corrupt_sims.extend(s_neg_hr.detach().cpu().tolist())
        
        # 6) Accumulate loss for reporting (use original unscaled loss)
        batch_size = s_pos.size(0)
        total_loss += loss.item() * batch_size  # Use original loss, not scaled
        sample_count += batch_size
        
        # Convert similarities to human-readable format for progress bar
        hr_pos = to_human_readable(s_pos, temperature=0.1, scale="prob")
        hr_neg = to_human_readable(s_neg, temperature=0.1, scale="prob")

        # 7) Update progress bar
        avg_loss = total_loss / sample_count
        avg_clean = hr_pos.mean().item()
        avg_corrupt = hr_neg.mean().item()
        
        pbar.set_postfix({
            "loss": f"{avg_loss:.4f}",
            "clean_sim": f"{avg_clean:.3f}",
            "corr_sim": f"{avg_corrupt:.3f}",
            "gap": f"{(avg_clean-avg_corrupt):.3f}",
            "opt_steps": optimizer_steps
        })

        # 8 Clear memory periodically
        if batch_idx % 35 == 0:
            torch.cuda.empty_cache()
            if batch_idx % 100 == 0:
                print_memory_summary()

    logger.info(f"Epoch {epoch}: Total optimizer steps: {optimizer_steps}")
    
    return {
        "loss": total_loss / sample_count,
        "clean_similarity": np.mean(clean_sims),
        "corrupt_similarity": np.mean(corrupt_sims),  
        "similarity_gap": np.mean(clean_sims) - np.mean(corrupt_sims),
        "optimizer_steps": optimizer_steps,
    }


def evaluate(model,
             data_loader,
             loss_fn,
             device,
             epoch: int = None,
             split: str = "Validation",
             fp16: bool = False):
    """
    Evaluate the model using the full architecture pipeline,
    maintaining consistency with the training process.
    """
    model.eval()

    total_loss = 0.0
    all_similarities = []
    clean_similarities = []
    corrupt_similarities = []
    sample_count = 0

    desc = f"Epoch {epoch} [{split}]" if epoch is not None else f"[{split}]"
    progress_bar = tqdm(data_loader, desc=desc)

    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Skip empty batches
                if batch is None:
                    logger.warning("Skipping None batch during evaluation")
                    continue

                # Move to device
                batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}

                # Run forward pass with both positive and negative examples
                autocast_ctx = torch.amp.autocast(device_type='cuda') if fp16 else torch.cuda.amp.autocast(enabled=False)
                with autocast_ctx:
                    # Use compute_pos_neg_embeddings to get embeddings - same as in training
                    txt_pos_emb, txt_neg_emb, aud_emb = EnhancedAudioTextModel.compute_pos_neg_embeddings(model, batch)
                    
                    # Calculate similarities
                    s_pos = (aud_emb * txt_pos_emb).sum(dim=1)  # [B]
                    s_neg = (aud_emb * txt_neg_emb).sum(dim=1)  # [B]
                    
                    # Get alignment scores
                    alignment_scores = getattr(model, "last_alignment_scores", None)
                    
                    # Calculate loss correctly
                    loss = loss_fn(s_pos, s_neg, alignment_scores=alignment_scores)

                s_pos_hr = to_human_readable(s_pos, temperature=0.1, scale='prob')
                s_neg_hr = to_human_readable(s_neg, temperature=0.1, scale='prob')
                # Collect similarities
                all_similarities.extend(s_pos_hr.cpu().numpy())  # Use positive similarities for general metrics
                clean_similarities.extend(s_pos_hr.cpu().numpy())
                corrupt_similarities.extend(s_neg_hr.cpu().numpy())

                # Update counters
                batch_size = aud_emb.size(0)
                sample_count += batch_size
                total_loss += loss.item() * batch_size

                # Update progress bar
                avg_loss = total_loss / sample_count
                avg_clean = np.mean(clean_similarities) if clean_similarities else 0.0
                avg_corrupt = np.mean(corrupt_similarities) if corrupt_similarities else 0.0
                clean_hr = to_human_readable(s_pos, temperature=0.1, scale='prob')
                corr_hr  = to_human_readable(s_neg, temperature=0.1, scale='prob')

                progress_bar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "clean_sim": f"{clean_hr.mean().item():.3f}",
                    "corrupt_sim": f"{corr_hr.mean().item():.3f}",
                    "gap": f"{(clean_hr.mean()-corr_hr.mean()).item():.3f}"
                })

                # Clean up GPU memory
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()

            except Exception as e:
                logger.error(f"Error in evaluation batch {batch_idx}: {e}")
                logger.error(f"Batch keys: {list(batch.keys() if batch else [])}")
                import traceback
                logger.error(traceback.format_exc())
                continue

    # Calculate final metrics
    if sample_count == 0:
        logger.warning(f"No valid samples were processed during {split} evaluation")
        return {k: 0.0 for k in (
            "loss", "avg_similarity", "median_similarity", "std_similarity",
            "clean_similarity", "corrupt_similarity", "similarity_gap"
        )}, []

    avg_similarity = np.mean(all_similarities)
    std_similarity = np.std(all_similarities)
    median_similarity = np.median(all_similarities)
    avg_clean = np.mean(clean_similarities) if clean_similarities else 0.0
    avg_corrupt = np.mean(corrupt_similarities) if corrupt_similarities else 0.0
    similarity_gap = avg_clean - avg_corrupt

    logger.info(f"{split} metrics:")
    logger.info(f"  Loss: {total_loss / sample_count:.4f}")
    logger.info(f"  Average similarity: {avg_similarity:.4f}")
    logger.info(f"  Median similarity: {median_similarity:.4f}")
    logger.info(f"  Clean sample similarity: {avg_clean:.4f}")
    logger.info(f"  Corrupted sample similarity: {avg_corrupt:.4f}")
    logger.info(f"  Similarity gap (clean - corrupt): {similarity_gap:.4f}")

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
    temperature=0.1,
    num_warmup_steps=0,
    projection_dim=768,
    use_cross_modal=True,
    use_attentive_pooling=True,
    use_word_alignment=False,  # New default: use word-level alignment
    save_every=1,
    accumulation_steps=4,  # Gradient accumulation steps
    fp16=True,  # Use mixed precision training
    freeze_encoders="partial",  # New: "full", "partial", "none"
    text_layers_to_unfreeze=5,   # New: for partial unfreezing
    audio_layers_to_unfreeze=5,  # New: for partial unfreezing
    use_bucketing=False,   # Whether to use length-based bucketing
    seed=42,
    max_audio_len=160000  # Maximum audio length in samples
):
    """
    Train the audio-text embedding model and evaluate on validation and test sets.
    Enhanced with better architecture, training optimizations, word-level alignment,
    and partial encoder unfreezing with discriminative learning rates.
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
    logger.info(f"  Text layers to unfreeze: {text_layers_to_unfreeze}")
    logger.info(f"  Audio layers to unfreeze: {audio_layers_to_unfreeze}")
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
    
    # Initialize enhanced model with partial unfreezing support
    logger.info("Initializing model...")
    model = EnhancedAudioTextModel(
        text_model_name=text_model_name,
        audio_model_name=audio_model_name,
        projection_dim=projection_dim,
        use_cross_modal=use_cross_modal,
        use_attentive_pooling=use_attentive_pooling,
        use_word_alignment=use_word_alignment,
        freeze_encoders=freeze_encoders,
        text_layers_to_unfreeze=text_layers_to_unfreeze,
        audio_layers_to_unfreeze=audio_layers_to_unfreeze
    )
    
    model = model.to(device)
    
    # Initialize mixed precision training if requested
    scaler = torch.amp.GradScaler('cuda') if fp16 else None
    
    # Enhanced optimizer initialization with discriminative learning rates
    if freeze_encoders == "partial":
        # Use discriminative learning rates for partial unfreezing
        encoder_lr = learning_rate /50  
        
        # Separate parameters into encoder and non-encoder groups
        encoder_params = []
        non_encoder_params = []
        
        # Collect unfrozen encoder parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'text_encoder' in name or 'audio_encoder' in name:
                    encoder_params.append(param)
                else:
                    non_encoder_params.append(param)
        
        # Create parameter groups with different learning rates
        param_groups = [
            {'params': encoder_params, 'lr': encoder_lr, 'weight_decay': weight_decay},
            {'params': non_encoder_params, 'lr': learning_rate, 'weight_decay': weight_decay}
        ]
        
        optimizer = AdamW(param_groups)
        logger.info(f"Using discriminative learning rates: encoder_lr={encoder_lr}, main_lr={learning_rate}")
        logger.info(f"Encoder parameters: {len(encoder_params)}, Non-encoder parameters: {len(non_encoder_params)}")
        
    else:
        # Original single learning rate approach
        optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=learning_rate,
            weight_decay=weight_decay
        )
        logger.info(f"Using single learning rate: {learning_rate}")
    
    # Initialize alignment-aware InfoNCE loss with temperature parameter
    loss_fn = AlignmentAwareInfoNCE(temperature=temperature, alignment_weight=0.5)
    
    # Initialize learning rate scheduler
    # Calculate total optimizer steps more precisely 
    steps_per_epoch = len(train_loader) // accumulation_steps + (1 if len(train_loader) % accumulation_steps > 0 else 0)
    total_steps = steps_per_epoch * num_epochs
    
    logger.info(f"Scheduler setup:")
    logger.info(f"  Batches per epoch: {len(train_loader)}")
    logger.info(f"  Accumulation steps: {accumulation_steps}")
    logger.info(f"  Optimizer steps per epoch: {steps_per_epoch}")
    logger.info(f"  Total optimizer steps: {total_steps}")
    logger.info(f"  Warmup steps: {num_warmup_steps}")
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps
    )
    
    # Validate gradient accumulation setup (debug mode)
    if accumulation_steps > 1:
        logger.info("Validating gradient accumulation setup...")
        try:
            validate_gradient_accumulation(
                model, train_loader, 
                accumulation_steps=accumulation_steps, 
                device=device, 
                test_batches=min(accumulation_steps + 2, 10)
            )
        except Exception as e:
            logger.warning(f"Gradient accumulation validation failed: {e}")
            logger.warning("Continuing with training anyway...")
    
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
                        "freeze_encoders": freeze_encoders,
                        "text_layers_to_unfreeze": text_layers_to_unfreeze,
                        "audio_layers_to_unfreeze": audio_layers_to_unfreeze,
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
                        "freeze_encoders": freeze_encoders,
                        "text_layers_to_unfreeze": text_layers_to_unfreeze,
                        "audio_layers_to_unfreeze": audio_layers_to_unfreeze,
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
                        "freeze_encoders": freeze_encoders,
                        "text_layers_to_unfreeze": text_layers_to_unfreeze,
                        "audio_layers_to_unfreeze": audio_layers_to_unfreeze,
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
                        
                        txt_pos_emb, txt_neg_emb, aud_emb = model(batch)  # Returns 3 values
                        clean_sim.extend((aud_emb * txt_pos_emb).sum(dim=1).cpu().numpy())
                        corrupt_sim.extend((aud_emb * txt_neg_emb).sum(dim=1).cpu().numpy())
                
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
            "freeze_encoders": freeze_encoders,
            "text_layers_to_unfreeze": text_layers_to_unfreeze,
            "audio_layers_to_unfreeze": audio_layers_to_unfreeze,
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
                
                txt_pos_emb, txt_neg_emb, aud_emb = model(batch)  # Returns 3 values
                clean_sim.extend((aud_emb * txt_pos_emb).sum(dim=1).cpu().numpy())
                corrupt_sim.extend((aud_emb * txt_neg_emb).sum(dim=1).cpu().numpy())
                
              
        
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
                
                txt_pos_emb, txt_neg_emb, aud_emb = model(batch)  # Returns 3 values
                clean_sim.extend((aud_emb * txt_pos_emb).sum(dim=1).cpu().numpy())
                corrupt_sim.extend((aud_emb * txt_neg_emb).sum(dim=1).cpu().numpy())
        
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
    parser.add_argument("--projection_dim", type=int, default=768,
                        help="Dimension of the shared embedding space")
    parser.add_argument("--no_cross_modal", action="store_true",
                        help="Disable cross-modal attention")
    parser.add_argument("--no_attentive_pooling", action="store_true",
                        help="Disable attentive pooling")
    parser.add_argument("--no_word_alignment", action="store_true",
                        help="Disable word-level alignment")
    
    # Enhanced freezing parameters
    parser.add_argument("--freeze_encoders", type=str, default="partial", 
                        choices=["full", "partial", "none"],
                        help="Encoder freezing strategy: 'full' (freeze all), 'partial' (unfreeze last layers), 'none' (unfreeze all)")
    parser.add_argument("--text_layers_to_unfreeze", type=int, default=3,
                        help="Number of text encoder layers to unfreeze (for partial freezing)")
    parser.add_argument("--audio_layers_to_unfreeze", type=int, default=3,
                        help="Number of audio encoder layers to unfreeze (for partial freezing)")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Training batch size")
    parser.add_argument("--num_epochs", type=int, default=15,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--temperature", type=float, default=0.1,
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
        dataset = load_dataset("mozilla-foundation/common_voice_17_0", "pt", token=True)
        
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
        use_word_alignment=False,
        save_every=args.save_every,
        accumulation_steps=args.acc_steps,
        fp16=not args.no_fp16,
        freeze_encoders=args.freeze_encoders,
        text_layers_to_unfreeze=args.text_layers_to_unfreeze,
        audio_layers_to_unfreeze=args.audio_layers_to_unfreeze,
        use_bucketing=args.bucket,
        seed=args.seed,
        max_audio_len=args.max_audio_len
    )
    
    logger.info("All tasks completed!")
    

if __name__ == "__main__":
    main()