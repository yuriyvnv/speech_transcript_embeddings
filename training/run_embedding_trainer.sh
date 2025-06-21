#!/bin/bash

# Audio-Text Embedding Training Runner Script
# This script runs the audio-text embedding trainer with the specified arguments

# Exit on error
set -e

# Default values
DATA_DIR="/home/yperezhohin/speech_transcript_embeddings/common_voice_data"
OUTPUT_DIR="/home/yperezhohin/speech_transcript_embeddings/audio_text_model_optimized_wo_alignment"
TEXT_MODEL="sentence-transformers/all-roberta-large-v1"
AUDIO_MODEL="facebook/w2v-bert-2.0"
PROJECTION_DIM=1024
BATCH_SIZE=32 # 64 with accumulation steps
NUM_EPOCHS=20
LEARNING_RATE=2e-5
WEIGHT_DECAY=0.01
TEMPERATURE=0.1
MAX_TEXT_LENGTH=256
WARMUP_STEPS=2000
SAVE_EVERY=1
ACC_STEPS=6
MAX_AUDIO_LEN=480000
SEED=42
DEBUG=false
FP16_FLAG="--no_fp16"
FREEZE_FLAG=""
DEBUG_FLAG=""

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
    --bucket)
      BUCKET_FLAG="--bucket"
      shift
      ;;
    --no_fp16)
      FP16_FLAG="--no_fp16"
      shift
      ;;
    --no_freeze)
      FREEZE_FLAG="--no_freeze"
      shift
      ;;
    --debug)
      DEBUG_FLAG="--debug"
      shift
      ;;
    --help)
      echo "Audio-Text Embedding Training Runner Script"
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
      echo "  --max_audio_len LEN     Maximum audio length in samples (default: $MAX_AUDIO_LEN)"
      echo "  --seed SEED             Random seed (default: $SEED)"
      echo "  --bucket                Enable length-based bucketing for efficiency"
      echo "  --no_fp16               Disable mixed precision training"
      echo "  --no_freeze             Don't freeze encoders (train entire model)"
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

if [ -n "$DEBUG_FLAG" ]; then
  CMD+=" $DEBUG_FLAG"
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Display the command being run
echo "Running command:"
echo "$CMD"
echo "------------------------------------------------------------"

# Run the command with Python
eval "$CMD"

echo "------------------------------------------------------------"
echo "Training completed!"
echo "Results saved to: $OUTPUT_DIR"