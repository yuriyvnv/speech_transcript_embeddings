#!/usr/bin/env python
"""
Script to calculate text-audio similarities for Common Voice samples.
Properly matches the model configuration from training.
"""

import torch
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging

# Import our modular components
from model import EnhancedAudioTextModel
from processor import AudioTextProcessor

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def run_cv_inference(num_samples=10):
    """Run inference on Common Voice samples using our audio-text model."""
    # Configuration
    checkpoint_path = "/home/yperezhohin/transcript_speech_embedding/audio_text_model/best_model.pt"
    results_dir = "cv_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Print header
    print("\n" + "=" * 60)
    print(f"Common Voice Test Split Similarity (First {num_samples} Samples)")
    print("=" * 60)
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
        text_embedding_dim=1024,  # These should match your training config
        audio_embedding_dim=1024,
        use_cross_modal=use_cross_modal,
        use_attentive_pooling=use_attentive_pooling,
        freeze_encoders=True  # Assuming this was used during training
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
    
    # Let's try using the model's own forward method directly
    def compute_similarity_using_model(model, batch):
        """Compute similarity using the model's forward pass for consistency."""
        with torch.no_grad():
            text_embeddings, audio_embeddings = model.forward(batch)
            # Model already normalizes embeddings in forward pass
            similarity = torch.sum(text_embeddings * audio_embeddings, dim=1).cpu().numpy()
        return similarity, text_embeddings, audio_embeddings
    
    # Load Common Voice dataset
    try:
        from datasets import load_dataset, Audio
        
        print(f"Loading Common Voice test split (first {num_samples} samples)...")
        
        # Load and preprocess the dataset
        dataset = load_dataset("mozilla-foundation/common_voice_17_0", "pt", split=f"test[:{num_samples}]")
        
        # Cast to Audio column with 16kHz sampling rate
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
        
        print(f"Successfully loaded {len(dataset)} samples from Common Voice test split")
        
        # Store results
        results = []
        
        # Process each sample
        for idx, sample in enumerate(tqdm(dataset, desc="Processing samples")):
            print(f"\nSample {idx+1}/{len(dataset)}")
            
            # Get text and audio
            text = sample["sentence"]
            print(f"Text: {text}")
            
            audio_array = sample["audio"]["array"]
            sampling_rate = sample["audio"]["sampling_rate"]
            
            # Process text and audio using our processor
            text_input = processor.process_text(text)
            audio_input = processor.process_audio_array(audio_array, sampling_rate)
            
            # Prepare batch for model.forward()
            batch = {
                "input_ids": text_input["input_ids"],
                "attention_mask": text_input["attention_mask"],
                "input_features": audio_input["input_features"],
                "attention_mask_audio": audio_input["attention_mask_audio"]
            }
            
            # Option 1: Use model's forward method directly (recommended)
            similarity, _, _ = compute_similarity_using_model(model, batch)
            similarity = similarity[0]  # Extract scalar from array
            
            # Option 2: Alternative approach using processor methods
            text_embedding = processor.get_text_embedding(model, text_input)
            audio_embedding = processor.get_audio_embedding(model, audio_input)
            similarity_alt = processor.compute_similarity(text_embedding, audio_embedding)[0]
            
            # Print both results to check if they match
            print(f"Similarity score (via model.forward): {similarity:.4f}")
            print(f"Similarity score (via processor): {similarity_alt:.4f}")
            
            # Store result using the model.forward approach for consistency with training
            results.append({
                "sample_id": sample.get("id", str(idx)),
                "text": text,
                "similarity": similarity
            })
            
            # Create visualization
            plt.figure(figsize=(8, 3))
            plt.bar(['Similarity'], [similarity], color='#3498db', width=0.4)
            plt.title(f'Sample {idx+1}: Text-Audio Similarity')
            plt.ylabel('Cosine Similarity')
            plt.ylim(-1, 1)  # Show full range of possible similarities
            plt.text(0, similarity/2 if similarity > 0 else similarity/2, 
                    f'{similarity:.4f}', ha='center', va='center', 
                    fontweight='bold', color='white', fontsize=14)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            # Save visualization
            viz_path = os.path.join(results_dir, f"sample_{idx+1}_similarity.png")
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        # Save results to CSV
        results_df = pd.DataFrame(results)
        csv_path = os.path.join(results_dir, "cv_similarities.csv")
        results_df.to_csv(csv_path, index=False)
        
        # Print summary
        print("\n" + "=" * 60)
        print("Results Summary")
        print("=" * 60)
        print(f"Processed {len(results)} samples")
        print(f"Average similarity: {results_df['similarity'].mean():.4f}")
        print(f"Min similarity: {results_df['similarity'].min():.4f}")
        print(f"Max similarity: {results_df['similarity'].max():.4f}")
        
        print("\nTop 3 samples by similarity:")
        top_3 = results_df.sort_values(by='similarity', ascending=False).head(3)
        for i, row in top_3.iterrows():
            print(f"  {i+1}. {row['similarity']:.4f} - \"{row['text'][:50]}{'...' if len(row['text']) > 50 else ''}\"")
        
        print(f"\nResults saved to: {csv_path}")
        print(f"Visualizations saved to: {results_dir}/")
        
        # Create combined visualization
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(results)), [r['similarity'] for r in results], color='#3498db')
        plt.xlabel('Sample Number')
        plt.ylabel('Similarity Score')
        plt.title('Similarity Scores for Common Voice Samples')
        plt.xticks(range(len(results)), [f'{i+1}' for i in range(len(results))])
        plt.ylim(-1, 1)  # Show full range
        
        # Add values on top of bars
        for i, sim in enumerate([r['similarity'] for r in results]):
            pos = sim + 0.05 if sim > 0 else sim - 0.1
            plt.text(i, pos, f'{sim:.2f}', ha='center')
            
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save combined visualization
        combined_viz_path = os.path.join(results_dir, "all_similarities.png")
        plt.savefig(combined_viz_path, dpi=300, bbox_inches='tight')
        print(f"Combined visualization saved to: {combined_viz_path}")
        
        return results
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nUnable to load Common Voice dataset. Please make sure you have:")
        print("1. Installed the datasets library: pip install datasets")
        print("2. Logged in to Hugging Face: huggingface-cli login")
        print("3. Accepted the dataset terms at: https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0")
        return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Common Voice inference")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of samples to process")
    
    args = parser.parse_args()
    
    run_cv_inference(args.num_samples)