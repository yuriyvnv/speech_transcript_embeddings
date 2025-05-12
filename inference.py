#!/usr/bin/env python
"""
Simple, properly fixed script to run audio-text similarity inference.
Auto-detects model configuration from checkpoint and uses the model's forward pass
for consistency with training.
"""

import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import logging

# Import our modular components
from model import EnhancedAudioTextModel  # Import the model class
from processor import AudioTextProcessor  # Import the processor class

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def run_inference():
    """Run inference with our audio-text model."""
    # Configuration
    checkpoint_path = "/home/yperezhohin/transcript_speech_embedding/audio_text_model_optimized/best_model_gap.pt"  # Change if you want a different checkpoint
    audio_path = "/home/yperezhohin/transcript_speech_embedding/teste1.mp3"
    text = "nao"
    
    # Print header
    print("\n" + "=" * 60)
    print("Audio-Text Similarity Inference")
    print("=" * 60)
    print(f"Audio file: {audio_path}")
    print(f"Text: {text}")
    print(f"Model checkpoint: {checkpoint_path}")
    print("-" * 60)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load checkpoint first to get configuration
    print(f"Loading checkpoint to extract configuration...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model configuration from checkpoint
    epoch = checkpoint.get("epoch", "unknown")
    val_loss = checkpoint.get("val_loss", "unknown") 
    temperature = checkpoint.get("temperature", 0.07)
    projection_dim = checkpoint.get("projection_dim", 1024)
    
    # Check the state_dict for clues about model configuration
    state_dict = checkpoint["model_state_dict"]
    use_cross_modal = any("text_to_audio_attention" in key for key in state_dict.keys())
    use_attentive_pooling = any("text_pooling" in key for key in state_dict.keys())
    
    print(f"Detected configuration from checkpoint:")
    print(f" - Epoch: {epoch}")
    print(f" - Validation loss: {val_loss}")
    print(f" - Temperature: {temperature}")
    print(f" - Projection dimension: {projection_dim}")
    print(f" - Using cross-modal attention: {use_cross_modal}")
    print(f" - Using attentive pooling: {use_attentive_pooling}")
    
    # Initialize model with the correct configuration
    print(f"Initializing model with detected configuration...")
    model = EnhancedAudioTextModel(
        text_model_name="sentence-transformers/all-roberta-large-v1",
        audio_model_name="facebook/w2v-bert-2.0",
        projection_dim=projection_dim,
        text_embedding_dim=1024,
        audio_embedding_dim=1024,
        use_cross_modal=use_cross_modal,
        use_attentive_pooling=use_attentive_pooling,
        freeze_encoders=True
    )
    
    # Load state dict
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model = model.to(device)
    
    # Print model summary
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {trainable_params:,} trainable parameters out of {total_params:,} total")
    
    # Initialize processor
    print("Initializing processor...")
    processor = AudioTextProcessor(
        text_model_name="sentence-transformers/all-roberta-large-v1",
        audio_model_name="facebook/w2v-bert-2.0",
        device=device
    )
    
    # Process text
    print("Processing text...")
    text_input = processor.process_text(text)
    
    # Process audio
    print("Loading and processing audio...")
    audio_input = processor.process_audio_file(audio_path)
    
    # Create batch for model.forward()
    batch = {
        "input_ids": text_input["input_ids"],
        "attention_mask": text_input["attention_mask"],
        "input_features": audio_input["input_features"],
        "attention_mask_audio": audio_input["attention_mask_audio"]
    }
    
    # Generate embeddings using model's forward method (this is how it was used in training)
    print("Generating embeddings via model.forward()...")
    with torch.no_grad():
        text_embeddings, audio_embeddings = model(batch)
        # Model already normalizes embeddings in forward pass
        similarity = torch.sum(text_embeddings * audio_embeddings, dim=1).cpu().numpy()[0]
    
    # For comparison, also get embeddings using processor methods
    print("For comparison, also generating embeddings via processor...")
    with torch.no_grad():
        text_embedding_alt = processor.get_text_embedding(model, text_input)
        audio_embedding_alt = processor.get_audio_embedding(model, audio_input)
        similarity_alt = processor.compute_similarity(text_embedding_alt, audio_embedding_alt)[0]
    
    # Print results
    print("\n" + "=" * 60)
    print(f"Similarity score (via model.forward): {similarity:.4f}")
    print(f"Similarity score (via processor): {similarity_alt:.4f}")
    print("=" * 60)
    
    # Create visualization
    plt.figure(figsize=(8, 4))
    
    # Plot both similarities for comparison
    plt.bar([0, 1], [similarity, similarity_alt], 
            color=['#3498db', '#e74c3c'], width=0.4)
    plt.xticks([0, 1], ['Via model.forward()', 'Via processor'])
    plt.title('Text-Audio Similarity')
    plt.ylabel('Cosine Similarity')
    plt.ylim(-1, 1)  # Show full range of possible similarities
    
    # Add the values as text on the bars
    plt.text(0, similarity/2 if similarity > 0 else similarity/2, 
             f'{similarity:.4f}', ha='center', va='center', 
             fontweight='bold', color='white', fontsize=14)
    plt.text(1, similarity_alt/2 if similarity_alt > 0 else similarity_alt/2, 
             f'{similarity_alt:.4f}', ha='center', va='center', 
             fontweight='bold', color='white', fontsize=14)
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save visualization
    output_path = "similarity_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")
    
    return similarity

if __name__ == "__main__":
    run_inference()