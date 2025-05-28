# Installation Guide for StreamDiffusion

This guide provides detailed instructions for installing StreamDiffusion on different platforms.

## System Requirements

- **Windows/Linux with NVIDIA GPU**: For full performance with TensorRT acceleration
- **macOS with Apple Silicon**: Supported with some limitations (no TensorRT)
- **Python**: Version 3.9+ recommended (3.10 tested and working on all platforms)

## Installation Steps

### Step 1: Clone the repository

```bash
git clone https://github.com/yourusername/StreamDiffusion.git
cd StreamDiffusion
```

### Step 2: Set up Python environment

```bash
# Create a new conda environment (recommended)
conda create -n streamdiffusion python=3.10
conda activate streamdiffusion

# Or use a venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install the package

#### Method A: Install as editable package (recommended for development)

```bash
# Install the base package with required dependencies
pip install -e .

# Optional: Install with specific extras based on your platform
# For Windows/Linux with NVIDIA GPU:
pip install -e ".[tensorrt,xformers]"  

# For macOS:
pip install -e ".[macos]"
```

#### Method B: Install specific demo dependencies

```bash
# Navigate to the specific demo directory
cd demo/realtime-img2img

# Install dependencies for that demo
pip install -r requirements.txt
```

## Platform-Specific Notes

### Windows/Linux (NVIDIA GPU)

- Make sure you have the latest NVIDIA drivers installed
- CUDA 11.8+ is recommended
- TensorRT acceleration requires a compatible NVIDIA GPU
- For best performance, install xformers: `pip install xformers`

### macOS (Apple Silicon)

- PyTorch with MPS (Metal Performance Shaders) acceleration is required
- Install PyTorch 2.0+ with MPS support: `pip install --pre torch torchvision`
- Use the `--acceleration none` parameter with demos
- Use the `--taesd` parameter to enable TinyVAE for better compatibility
- For demos, use the provided `run_mac.sh` script when available

### Common Issues

1. **Dependency Conflicts**: If you encounter dependency conflicts:
   ```bash
   pip install --upgrade pip
   pip install -e . --no-dependencies
   pip install diffusers==0.33.1 huggingface_hub==0.32.1 accelerate==1.7.0
   ```

2. **ImportError with update_prompt**: Ensure you're using the correct StreamDiffusionWrapper implementation:
   ```bash
   # Check import path in your demo's img2img.py or similar file
   # Make sure it's importing from main project instead of demo placeholder
   ```

3. **IndexError with t_index_list**: On macOS, ensure t_index values are below 30 (e.g., [15, 25])

## Running the Demo

After installation, you can run the demo with:

```bash
# For Windows/Linux with NVIDIA GPU:
python main.py --acceleration tensorrt

# For macOS:
python main.py --taesd --acceleration none --debug
# Or use the provided script:
./run_mac.sh
```

Then open your browser at http://localhost:7860
