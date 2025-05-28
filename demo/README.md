# StreamDiffusion Demos

This directory contains various demos showcasing StreamDiffusion capabilities.

## Available Demos

- [realtime-img2img](./realtime-img2img/): Real-time image-to-image generation with webcam
  - [macOS Instructions](./realtime-img2img/MACOS_README.md): Special instructions for macOS users

## Requirements

Each demo may have specific requirements. See the individual demo folders for details.

## Quick Start

For the most popular demo (realtime-img2img):

### Windows/Linux:
```bash
cd realtime-img2img
pip install -r requirements.txt
python main.py --acceleration tensorrt
```

### macOS:
```bash
cd realtime-img2img
pip install -r requirements.txt
./run_mac.sh
# Or: python main.py --taesd --acceleration none --debug
```

Then open your browser at http://localhost:7860
