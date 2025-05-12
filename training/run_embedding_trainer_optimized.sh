#!/bin/bash

# Simplified Optimized Audio-Text Embedding Training Runner Script
# This script runs the simplified optimized audio-text embedding trainer

# Exit on error
set -e

# Default values - updated with optimized settings
DATA_DIR="./common_voice_data"
OUTPUT_DIR="./audio_text_model_1024"
TEXT_MODEL="sentence-transformers/all-roberta-large-v1"
AUDIO_MODEL="facebook/w2v-bert-2.0"
PROJECTION_DIM=1024
BATCH_SIZE=16
NUM_EPOCHS=10
LEARNING_RATE=3e-5  # Optimized learning rate
WEIGHT_DECAY=0.01
TEMPERATURE=0.07
MAX_TEXT_LENGTH=256
WARMUP_STEPS=1000
SAVE_EVERY=1
ACC_STEPS=4
MAX_AUDIO_LEN=480000  # Increased to 12 seconds at 16kHz
SEED=42
DEBUG=false

# Default flags - enabling optimizations by default
BUCKET_FLAG="--bucket"  # Enable bucketing by default
FP16_FLAG=""  # Enable mixed precision by default
FREEZE_FLAG=""  # Freeze encoders by default
DEBUG_FLAG=""
AUDIO_AUG_FLAG=""  # Enable audio augmentation by default
COSINE_SCHEDULER_FLAG=""  # Enable cosine scheduler by default

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --data_dir)
      DATA_DIR="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --text_model)
      TEXT_MODEL="$2"
      shift 2
      ;;
    --audio_model)
      AUDIO_MODEL="$2"
      shift 2
      ;;
    --projection_dim)
      PROJECTION_DIM="$2"
      shift 2
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --num_epochs)
      NUM_EPOCHS="$2"
      shift 2
      ;;
    --learning_rate|--lr)
      LEARNING_RATE="$2"
      shift 2
      ;;
    --weight_decay|--wd)
      WEIGHT_DECAY="$2"
      shift 2
      ;;
    --temperature|--temp)
      TEMPERATURE="$2"
      shift 2
      ;;
    --max_text_length)
      MAX_TEXT_LENGTH="$2"
      shift 2
      ;;
    --warmup_steps)
      WARMUP_STEPS="$2"
      shift 2
      ;;
    --save_every)
      SAVE_EVERY="$2"
      shift 2
      ;;
    --acc_steps)
      ACC_STEPS="$2"
      shift 2
      ;;
    --max_audio_len)
      MAX_AUDIO_LEN="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --no_bucket)
      BUCKET_FLAG=""  # Disable bucketing
      shift
      ;;
    --no_fp16)
      FP16_FLAG="--no_fp16"  # Disable mixed precision
      shift
      ;;
    --no_freeze)
      FREEZE_FLAG="--no_freeze"  # Don't freeze encoders
      shift
      ;;
    --no_audio_aug)
      AUDIO_AUG_FLAG="--no_audio_aug"  # Disable audio augmentations
      shift
      ;;
    --no_cosine_scheduler)
      COSINE_SCHEDULER_FLAG="--no_cosine_scheduler"  # Use linear scheduler
      shift
      ;;
    --debug)
      DEBUG_FLAG="--debug"
      shift
      ;;
    --help)
      echo "Simplified Optimized Audio-Text Embedding Training Runner Script"
      echo ""
      echo "Usage: $0 [options]"
      echo ""
      echo "Options:"
      echo "  --data_dir DIR          Directory to store Common Voice data (default: $DATA_DIR)"
      echo "  --output_dir DIR        Directory to save model checkpoints (default: $OUTPUT_DIR)"
      echo "  --text_model NAME       Text encoder model name (default: $TEXT_MODEL)"
      echo "  --audio_model NAME      Audio encoder model name (default: $AUDIO_MODEL)"
      echo "  --projection_dim DIM    Dimension of shared embedding space (default: $PROJECTION_DIM)"
      echo "  --batch_size SIZE       Training batch size (default: $BATCH_SIZE)"
      echo "  --num_epochs N          Number of training epochs (default: $NUM_EPOCHS)"
      echo "  --learning_rate RATE    Learning rate (default: $LEARNING_RATE)"
      echo "  --weight_decay RATE     Weight decay (default: $WEIGHT_DECAY)"
      echo "  --temperature TEMP      Temperature for InfoNCE loss (default: $TEMPERATURE)"
      echo "  --max_text_length LEN   Maximum token length for text (default: $MAX_TEXT_LENGTH)"
      echo "  --warmup_steps STEPS    Number of warmup steps (default: $WARMUP_STEPS)"
      echo "  --save_every N          Save model checkpoint every N epochs (default: $SAVE_EVERY)"
      echo "  --acc_steps STEPS       Gradient accumulation steps (default: $ACC_STEPS)"
      echo "  --max_audio_len LEN     Maximum audio length in samples (default: $MAX_AUDIO_LEN, ~12s at 16kHz)"
      echo "  --seed SEED             Random seed (default: $SEED)"
      echo "  --no_bucket             Disable length-based bucketing (enabled by default)"
      echo "  --no_fp16               Disable mixed precision training (enabled by default)"
      echo "  --no_freeze             Don't freeze encoders (train entire model)"
      echo "  --no_audio_aug          Disable audio augmentations (enabled by default)"
      echo "  --no_cosine_scheduler   Use linear scheduler instead of cosine (cosine enabled by default)"
      echo "  --debug                 Enable debug logging"
      echo "  --help                  Show this help message and exit"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Run '$0 --help' for usage information"
      exit 1
      ;;
  esac
done

# Build command with appropriate flags
CMD="python trainer_optimized.py"
CMD+=" --data_dir $DATA_DIR"
CMD+=" --output_dir $OUTPUT_DIR"
CMD+=" --text_model \"$TEXT_MODEL\""
CMD+=" --audio_model \"$AUDIO_MODEL\""
CMD+=" --projection_dim $PROJECTION_DIM"
CMD+=" --batch_size $BATCH_SIZE"
CMD+=" --num_epochs $NUM_EPOCHS"
CMD+=" --learning_rate $LEARNING_RATE"
CMD+=" --weight_decay $WEIGHT_DECAY"
CMD+=" --temperature $TEMPERATURE"
CMD+=" --max_text_length $MAX_TEXT_LENGTH"
CMD+=" --warmup_steps $WARMUP_STEPS"
CMD+=" --save_every $SAVE_EVERY"
CMD+=" --acc_steps $ACC_STEPS"
CMD+=" --max_audio_len $MAX_AUDIO_LEN"
CMD+=" --seed $SEED"

# Add boolean flags
if [ -n "$BUCKET_FLAG" ]; then
  CMD+=" $BUCKET_FLAG"
fi

if [ -n "$FP16_FLAG" ]; then
  CMD+=" $FP16_FLAG"
fi

if [ -n "$FREEZE_FLAG" ]; then
  CMD+=" $FREEZE_FLAG"
fi

if [ -n "$AUDIO_AUG_FLAG" ]; then
  CMD+=" $AUDIO_AUG_FLAG"
fi

if [ -n "$COSINE_SCHEDULER_FLAG" ]; then
  CMD+=" $COSINE_SCHEDULER_FLAG"
fi

if [ -n "$DEBUG_FLAG" ]; then
  CMD+=" $DEBUG_FLAG"
fi

# Check if output directory exists and create if needed
if [ ! -d "$OUTPUT_DIR" ]; then
  echo "Creating output directory: $OUTPUT_DIR"
  mkdir -p "$OUTPUT_DIR"
fi

# Display the command being run
echo "Running command:"
echo "$CMD"
echo "------------------------------------------------------------"

# Run the command with Python
eval "$CMD"

echo "------------------------------------------------------------"
echo "Training completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "Best model saved to: $OUTPUT_DIR/best_model.pt"
echo "Test metrics saved to: $OUTPUT_DIR/test_metrics.json"