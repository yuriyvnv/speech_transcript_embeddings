"""
Model architecture definitions for the EnhancedAudioTextModel.
This module contains all the model components needed for inference.
"""

import torch
from torch import nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class EnhancedProjection(nn.Module):
    """Enhanced projection layer with multiple linear transformations and non-linearities."""
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
    """Cross-modal attention mechanism to capture relationships between modalities."""
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
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
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
    """Attentive pooling for sequence data."""
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, hidden_states, attention_mask=None):
        # Calculate attention scores
        attention_scores = self.attention(hidden_states).squeeze(-1)
        
        # Apply mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e9)
        
        # Apply softmax to get weights
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Apply attention weights
        pooled_output = torch.bmm(
            attention_weights.unsqueeze(1),
            hidden_states
        ).squeeze(1)
        
        return pooled_output


class EnhancedAudioTextModel(nn.Module):
    """Audio-Text multimodal embedding model."""
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
        freeze_encoders=True,
    ):
        super().__init__()
        
        # Load pre-trained models
        from transformers import AutoModel
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.audio_encoder = AutoModel.from_pretrained(audio_model_name)
        
        # Save configuration
        self.projection_dim = projection_dim
        self.use_cross_modal = use_cross_modal
        self.use_attentive_pooling = use_attentive_pooling
        
        # Freeze encoders if requested
        if freeze_encoders:
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
    
    def encode_text(self, input_ids, attention_mask=None):
        """Encode text inputs with enhanced processing."""
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
        """Encode audio inputs with enhanced processing."""
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
        """Apply cross-modal attention between text and audio."""
        if not self.use_cross_modal:
            return text_projected, audio_projected
        
        # Calculate cross-modal attentions
        text_attended = self.text_to_audio_attention(
            text_projected.unsqueeze(1),
            audio_hidden,
            audio_mask
        ).squeeze(1)
        
        audio_attended = self.audio_to_text_attention(
            audio_projected.unsqueeze(1),
            text_hidden,
            text_mask
        ).squeeze(1)
        
        # Fuse original and attended features
        text_fused = self.text_fusion(torch.cat([text_projected, text_attended], dim=1))
        audio_fused = self.audio_fusion(torch.cat([audio_projected, audio_attended], dim=1))
        
        return text_fused, audio_fused
    
    def forward(self, batch):
        """Forward pass with enhanced processing."""
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
        
        # Normalize embeddings
        text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
        audio_embeddings = F.normalize(audio_embeddings, p=2, dim=1)
        
        return text_embeddings, audio_embeddings