# StreamDiffusion Image-to-Image Example

[English](./README.md) | [日本語](./README-ja.md)

<p align="center">
  <img src="../../assets/img2img1.gif" width=80%>
</p>

<p align="center">
  <img src="../../assets/img2img2.gif" width=80%>
</p>


This example, based on this [MPJEG server](https://github.com/radames/Real-Time-Latent-Consistency-Model/), runs image-to-image with a live webcam feed or screen capture on a web browser.

## Usage
You need Node.js 18+ and Python 3.10 to run this example.
Please make sure you've installed all dependencies according to the [installation instructions](../../README.md#installation).

## Cross-Platform Installation

### Step 1: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Build the frontend

```bash
cd frontend
npm i
npm run build
cd ..
```

### Step 3: Run the application

#### For Windows/Linux with NVIDIA GPU:
```bash
python main.py --acceleration tensorrt
```

#### For macOS (Apple Silicon or Intel):
```bash
./run_mac.sh
```
Or manually:
```bash
python main.py --taesd --acceleration none --debug
```

#### For quick start (any platform):
```bash
chmod +x start.sh
./start.sh
```

Then open `http://localhost:7860` in your browser.
(*If `http://localhost:7860` does not work well, try `http://0.0.0.0:7860`)

## macOS-Specific Instructions

StreamDiffusion has been updated to work properly on macOS with the following enhancements:

1. **Dependency Compatibility:**
   - Updated `huggingface_hub` to version 0.32.1
   - Updated `diffusers` to version 0.33.1
   - Updated `accelerate` to version 1.7.0

2. **Prompt Update Fix:**
   - Fixed the `update_prompt` method in the `StreamDiffusionWrapper` class to ensure prompt changes take effect
   - The application now correctly processes new prompts during image generation

3. **macOS Optimization:**
   - Modified `t_index_list` values for better compatibility with SD-Turbo on macOS
   - Disabled TensorRT acceleration and using appropriate alternatives for Apple hardware
   - Ensured proper handling of the MPS (Metal Performance Shaders) backend

### Troubleshooting macOS Issues

If you see the Joker image regardless of prompt changes:
1. Delete the `utils/wrapper.py.bak` file if it exists: `rm utils/wrapper.py.bak`
2. Ensure you're using the main StreamDiffusion wrapper by checking imports
3. Use the `img2img` method for image processing rather than the `__call__` method

## Docker Support

```bash
docker build -t img2img .
docker run -ti -e ENGINE_DIR=/data -e HF_HOME=/data -v ~/.cache/huggingface:/data  -p 7860:7860 --gpus all img2img
```

Where `ENGINE_DIR` and `HF_HOME` set a local cache directory, making it faster to restart the docker container.
