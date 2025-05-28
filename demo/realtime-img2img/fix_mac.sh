#!/bin/bash
# Script for fixing common StreamDiffusion Mac issues

# Change to the application directory
cd "$(dirname "$0")"
ROOT_DIR=$(cd ../..; pwd)
echo "StreamDiffusion root directory: $ROOT_DIR"

echo "===== StreamDiffusion Diagnostic & Fix Tool for Mac ====="
echo "This script will diagnose and fix common issues with StreamDiffusion on macOS."

# Check Python environment
echo -e "\n1. Checking Python environment..."
PYTHON_VERSION=$(python --version)
echo "Python version: $PYTHON_VERSION"

# Check libraries
echo -e "\n2. Checking required libraries..."
MISSING_LIBRARIES=0

check_library() {
  local lib=$1
  if python -c "import $lib" 2>/dev/null; then
    echo "✅ $lib is installed"
    return 0
  else
    echo "❌ $lib is missing"
    MISSING_LIBRARIES=$((MISSING_LIBRARIES+1))
    return 1
  fi
}

check_library torch
check_library diffusers
check_library streamdiffusion
check_library huggingface_hub

# Check PyTorch MPS support
echo -e "\n3. Checking PyTorch MPS support..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}, CUDA available: {torch.cuda.is_available()}, MPS available: {torch.backends.mps.is_available() if hasattr(torch.backends, \"mps\") else False}')"

# Reinstall StreamDiffusion if needed
if ! check_library streamdiffusion; then
  echo -e "\n4. Reinstalling StreamDiffusion..."
  echo "Installing StreamDiffusion in development mode..."
  pip install -e "$ROOT_DIR"
  echo "Installation completed."
fi

# Fix permissions
echo -e "\n5. Fixing permissions..."
chmod +x run_mac.sh

# Clean cache
echo -e "\n6. Cleaning Python cache..."
find "$ROOT_DIR" -name "__pycache__" -type d -exec rm -rf {} +  2>/dev/null || true
find "$ROOT_DIR" -name "*.pyc" -delete

echo -e "\n7. Checking StreamDiffusion installation path..."
STREAMDIFFUSION_PATH=$(python -c "import streamdiffusion; print(streamdiffusion.__file__)" 2>/dev/null)
if [ -n "$STREAMDIFFUSION_PATH" ]; then
  echo "StreamDiffusion is installed at: $STREAMDIFFUSION_PATH"
else
  echo "Warning: Cannot locate StreamDiffusion module path."
fi

# Print wrap-up
echo -e "\n===== Diagnosis Complete ====="
if [ $MISSING_LIBRARIES -gt 0 ]; then
  echo "⚠️ $MISSING_LIBRARIES libraries are missing or problematic."
  echo "Please try running the application now with: ./run_mac.sh"
else
  echo "✅ All checks passed. Try running the application with: ./run_mac.sh"
fi

echo -e "\nIf you still encounter issues, try completely reinstalling the package:"
echo "pip uninstall -y streamdiffusion && pip install -e \"$ROOT_DIR\""
