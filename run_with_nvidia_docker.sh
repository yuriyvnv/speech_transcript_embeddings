#!/bin/bash
# Fixed script that handles missing nvidia-docker

# Build
echo "Building image..."
docker build -f Dockerfile.cuda -t speech:latest .

# Stop old container if exists
docker stop speech-train 2>/dev/null || true
docker rm speech-train 2>/dev/null || true

# Try different GPU methods
echo "Starting container..."

# Method 1: Try docker with --gpus flag (newer Docker versions)
if docker run -d \
    --gpus all \
    --name speech-train \
    -v $(pwd)/training:/app/training \
    -v $(pwd)/checkpoints:/app/checkpoints \
    -v $(pwd)/logs:/app/logs \
    -v $(pwd)/common_voice_data:/app/common_voice_data \
    -v $(pwd)/model.py:/app/model.py \
    -v $(pwd)/processor.py:/app/processor.py \
    -v $(pwd)/inference.py:/app/inference.py \
    speech:latest 2>/dev/null; then
    echo "Started with --gpus all"
# Method 2: Try with --runtime=nvidia
elif docker run -d \
    --runtime=nvidia \
    --name speech-train \
    -v $(pwd)/training:/app/training \
    -v $(pwd)/checkpoints:/app/checkpoints \
    -v $(pwd)/logs:/app/logs \
    -v $(pwd)/common_voice_data:/app/common_voice_data \
    -v $(pwd)/model.py:/app/model.py \
    -v $(pwd)/processor.py:/app/processor.py \
    -v $(pwd)/inference.py:/app/inference.py \
    speech:latest 2>/dev/null; then
    echo "Started with --runtime=nvidia"
# Method 3: Run without GPU
else
    echo "WARNING: Could not start with GPU support. Running without GPU..."
    docker run -d \
        --name speech-train \
        -v $(pwd)/training:/app/training \
        -v $(pwd)/checkpoints:/app/checkpoints \
        -v $(pwd)/logs:/app/logs \
        -v $(pwd)/common_voice_data:/app/common_voice_data \
        -v $(pwd)/model.py:/app/model.py \
        -v $(pwd)/processor.py:/app/processor.py \
        -v $(pwd)/inference.py:/app/inference.py \
        speech:latest
fi

echo ""
echo "Done! Check if container is running: docker ps"
echo "Check logs with: docker logs -f speech-train"
echo "Test GPU access: docker exec speech-train nvidia-smi"