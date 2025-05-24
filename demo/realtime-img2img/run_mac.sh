#!/bin/bash
# Script to run StreamDiffusion on Mac with optimal parameters
# Note: The t_index_list values have been adjusted in img2img.py to work with SD-Turbo on macOS

# Change to the application directory
cd "$(dirname "$0")"

# Kill any existing Python processes using port 7860
echo "Cleaning up any existing StreamDiffusion processes..."
lsof -ti:7860 | xargs kill -9 2>/dev/null

# Clear Python cache files
echo "Clearing Python cache files..."
find __pycache__ -name "*.pyc" -delete 2>/dev/null

# Set environment variables
export PYTHONPATH=$PYTHONPATH:/Users/patrickhartono/Documents/TD-Experiment/SD/sd-git/StreamDiffusion

# Run the application with Mac-friendly settings
echo "Starting StreamDiffusion with Mac-optimized settings..."
python main.py --taesd --acceleration none --debug
