#!/bin/bash

# Audio-Text Embedding Training Runner Script
# This script runs the audio-text embedding trainer with the specified arguments

# Exit on error
set -e

# Default values
DATA_DIR="/home/yperezhohin/speech_transcript_embeddings/common_voice_data"
OUTPUT_DIR="/home/yperezhohin/speech_transcript_embeddings/audio_text_model_optimized_unfreeze_3_layers"
TEXT_MODEL="sentence-transformers/all-roberta-large-v1"
AUDIO_MODEL="facebook/w2v-bert-2.0"
PROJECTION_DIM=1024
BATCH_SIZE=8 # 64 with accumulation steps
NUM_EPOCHS=10
LEARNING_RATE=1e-5
WEIGHT_DECAY=0.01
TEMPERATURE=0.1
MAX_TEXT_LENGTH=256
WARMUP_STEPS=500
SAVE_EVERY=1
ACC_STEPS=16
MAX_AUDIO_LEN=480000
SEED=42
DEBUG=false
FP16_FLAG="--no_fp16"
FREEZE_STRATEGY="partial"  # New: default to partial freezing
TEXT_LAYERS_TO_UNFREEZE=3  # New: default unfreezing 3 text layers
AUDIO_LAYERS_TO_UNFREEZE=3 # New: default unfreezing 3 audio layers
DEBUG_FLAG=""
BUCKET_FLAG=""
VALIDATE_GRADIENTS_FLAG=""

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
    --freeze_encoders)
      FREEZE_STRATEGY="$2"
      shift 2
      ;;
    --text_layers_to_unfreeze)
      TEXT_LAYERS_TO_UNFREEZE="$2"
      shift 2
      ;;
    --audio_layers_to_unfreeze)
      AUDIO_LAYERS_TO_UNFREEZE="$2"
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
    --debug)
      DEBUG_FLAG="--debug"
      shift
      ;;
    --validate_gradients)
      VALIDATE_GRADIENTS_FLAG="--validate_gradients"
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
      echo ""
      echo "Enhanced Freezing Options:"
      echo "  --freeze_encoders STRATEGY        Encoder freezing strategy (default: $FREEZE_STRATEGY)"
      echo "                                    Options: 'full' (freeze all), 'partial' (unfreeze last layers), 'none' (unfreeze all)"
      echo "  --text_layers_to_unfreeze N       Number of text encoder layers to unfreeze for partial freezing (default: $TEXT_LAYERS_TO_UNFREEZE)"
      echo "  --audio_layers_to_unfreeze N      Number of audio encoder layers to unfreeze for partial freezing (default: $AUDIO_LAYERS_TO_UNFREEZE)"
      echo ""
      echo "Training Optimizations:"
      echo "  --bucket                Enable length-based bucketing for efficiency"
      echo "  --no_fp16               Disable mixed precision training"
      echo "  --debug                 Enable debug logging"
      echo "  --validate_gradients    Run gradient accumulation validation only (no training)"
      echo ""
      echo "Examples:"
      echo "  # Default partial unfreezing (recommended)"
      echo "  $0"
      echo ""
      echo "  # Partial unfreezing with custom layer counts"
      echo "  $0 --freeze_encoders partial --text_layers_to_unfreeze 5 --audio_layers_to_unfreeze 4"
      echo ""
      echo "  # Full freezing (projection layers only)"
      echo "  $0 --freeze_encoders full"
      echo ""
      echo "  # No freezing (full fine-tuning)"
      echo "  $0 --freeze_encoders none"
      echo ""
      echo "  # High learning rate with partial unfreezing"
      echo "  $0 --learning_rate 1e-4 --freeze_encoders partial --text_layers_to_unfreeze 2"
      echo ""
      echo "  # Validate gradient accumulation setup"
      echo "  $0 --validate_gradients --acc_steps 8"
      echo ""
      echo "Notes:"
      echo "  - Partial unfreezing uses discriminative learning rates (10x lower for encoders)"
      echo "  - Text and audio layer counts are independent"
      echo "  - Use 'full' freezing for fastest training, 'partial' for best performance"
      echo "  - Gradient accumulation is properly implemented with correct scheduler timing"
      echo "  - Use --validate_gradients to test gradient accumulation setup before training"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Run '$0 --help' for usage information"
      exit 1
      ;;
  esac
done

# Validate freeze_encoders parameter
case $FREEZE_STRATEGY in
  full|partial|none)
    # Valid option
    ;;
  *)
    echo "Error: Invalid freeze_encoders strategy: $FREEZE_STRATEGY"
    echo "Valid options: full, partial, none"
    exit 1
    ;;
esac

# Validate layer counts for partial freezing
if [[ $FREEZE_STRATEGY == "partial" ]]; then
  if [[ $TEXT_LAYERS_TO_UNFREEZE -lt 1 || $TEXT_LAYERS_TO_UNFREEZE -gt 24 ]]; then
    echo "Warning: text_layers_to_unfreeze should be between 1 and 24, got: $TEXT_LAYERS_TO_UNFREEZE"
  fi
  if [[ $AUDIO_LAYERS_TO_UNFREEZE -lt 1 || $AUDIO_LAYERS_TO_UNFREEZE -gt 24 ]]; then
    echo "Warning: audio_layers_to_unfreeze should be between 1 and 24, got: $AUDIO_LAYERS_TO_UNFREEZE"
  fi
fi

# Display configuration summary
echo "================================================================"
echo "Audio-Text Embedding Training Configuration"
echo "================================================================"
echo "Data & Model:"
echo "  Data directory:         $DATA_DIR"
echo "  Output directory:       $OUTPUT_DIR"
echo "  Text model:             $TEXT_MODEL"
echo "  Audio model:            $AUDIO_MODEL"
echo "  Projection dimension:   $PROJECTION_DIM"
echo ""
echo "Training Parameters:"
echo "  Batch size:             $BATCH_SIZE"
echo "  Accumulation steps:     $ACC_STEPS"
echo "  Effective batch size:   $((BATCH_SIZE * ACC_STEPS))"
echo "  Number of epochs:       $NUM_EPOCHS"
echo "  Learning rate:          $LEARNING_RATE"
echo "  Weight decay:           $WEIGHT_DECAY"
echo "  Temperature:            $TEMPERATURE"
echo "  Warmup steps:           $WARMUP_STEPS"
echo "  Gradient accumulation:  ✓ Fixed implementation with proper scheduler"
echo ""
echo "Enhanced Freezing Strategy:"
echo "  Freeze encoders:        $FREEZE_STRATEGY"
if [[ $FREEZE_STRATEGY == "partial" ]]; then
echo "  Text layers to unfreeze: $TEXT_LAYERS_TO_UNFREEZE"
echo "  Audio layers to unfreeze: $AUDIO_LAYERS_TO_UNFREEZE"
echo "  → Discriminative LR:    ${LEARNING_RATE} (main) / $( echo "$LEARNING_RATE * 0.1" | bc -l ) (encoders)"
fi
echo ""
echo "Other Settings:"
echo "  Max text length:        $MAX_TEXT_LENGTH"
echo "  Max audio length:       $MAX_AUDIO_LEN samples ($( echo "scale=1; $MAX_AUDIO_LEN / 16000" | bc )s at 16kHz)"
echo "  Random seed:            $SEED"
echo "  Mixed precision:        $([ "$FP16_FLAG" = "--no_fp16" ] && echo "Disabled" || echo "Enabled")"
echo "  Length bucketing:       $([ -n "$BUCKET_FLAG" ] && echo "Enabled" || echo "Disabled")"
echo "  Debug logging:          $([ -n "$DEBUG_FLAG" ] && echo "Enabled" || echo "Disabled")"
echo "================================================================"
echo ""

# Build command with appropriate flags
CMD="python trainer_unfreeze.py"
CMD+=" --data_dir \"$DATA_DIR\""
CMD+=" --output_dir \"$OUTPUT_DIR\""
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

# Add enhanced freezing parameters
CMD+=" --freeze_encoders $FREEZE_STRATEGY"
CMD+=" --text_layers_to_unfreeze $TEXT_LAYERS_TO_UNFREEZE"
CMD+=" --audio_layers_to_unfreeze $AUDIO_LAYERS_TO_UNFREEZE"

# Add boolean flags
if [ -n "$BUCKET_FLAG" ]; then
  CMD+=" $BUCKET_FLAG"
fi

if [ -n "$FP16_FLAG" ]; then
  CMD+=" $FP16_FLAG"
fi

if [ -n "$DEBUG_FLAG" ]; then
  CMD+=" $DEBUG_FLAG"
fi

if [ -n "$VALIDATE_GRADIENTS_FLAG" ]; then
  CMD+=" $VALIDATE_GRADIENTS_FLAG"
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Display the command being run
echo "Running command:"
echo "$CMD"
echo "================================================================"
echo ""

# Record start time
START_TIME=$(date)
echo "Training started at: $START_TIME"
echo ""

# Run the command with Python
eval "$CMD"

# Record completion
END_TIME=$(date)
echo ""
echo "================================================================"
echo "Training completed!"
echo "Started:  $START_TIME"
echo "Finished: $END_TIME"
echo "Results saved to: $OUTPUT_DIR"
echo ""

# Display key files created
echo "Key files created:"
if [ -f "$OUTPUT_DIR/best_model_loss.pt" ]; then
  echo "  ✓ $OUTPUT_DIR/best_model_loss.pt"
fi
if [ -f "$OUTPUT_DIR/best_model_gap.pt" ]; then
  echo "  ✓ $OUTPUT_DIR/best_model_gap.pt"
fi
if [ -f "$OUTPUT_DIR/final_model.pt" ]; then
  echo "  ✓ $OUTPUT_DIR/final_model.pt"
fi
if [ -f "$OUTPUT_DIR/training.log" ]; then
  echo "  ✓ $OUTPUT_DIR/training.log"
fi
if [ -f "$OUTPUT_DIR/test_metrics.json" ]; then
  echo "  ✓ $OUTPUT_DIR/test_metrics.json"
fi
echo "================================================================"