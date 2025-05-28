#!/bin/bash
# Script to run StreamDiffusion on Mac with optimal parameters
# Note: The t_index_list values have been adjusted in img2img.py to work with SD-Turbo on macOS

# Change to the application directory
cd "$(dirname "$0")"

# Kill any existing Python processes using port 7860
echo "Cleaning up any existing StreamDiffusion processes..."
lsof -ti:7860 | xargs kill -9 2>/dev/null

# Clear Python cache files
echo "Cleaning Python cache to ensure fresh imports..."
find __pycache__ -name "*.pyc" -delete 2>/dev/null
find ../__pycache__ -name "*.pyc" -delete 2>/dev/null
find ../../utils/__pycache__ -name "*.pyc" -delete 2>/dev/null

# Verify Python environment is correctly set up
echo "Verifying Python environment..."
PYTHON_VERSION=$(python --version)
echo "Using Python: $PYTHON_VERSION"
PIP_VERSION=$(pip --version)
echo "Using pip: $PIP_VERSION"

# Check if required libraries are installed
echo "Checking required libraries..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}, CUDA available: {torch.cuda.is_available()}, MPS available: {torch.backends.mps.is_available() if hasattr(torch.backends, \"mps\") else False}')"

# Set robust environment variables
export PYTHONPATH="$PYTHONPATH:$(cd ../..; pwd)"
echo "PYTHONPATH set to: $PYTHONPATH"

# Create logs directory if it doesn't exist
mkdir -p logs

# Ensure acceleration is set to none for macOS
sed -i '' 's/acceleration: tensorrt/acceleration: none/' main.py

# Run the application with Mac-friendly settings
echo "Starting StreamDiffusion with Mac-optimized settings..."
python main.py --taesd --acceleration none --debug 2>&1 | tee logs/streamdiffusion_$(date +%Y%m%d_%H%M%S).log
