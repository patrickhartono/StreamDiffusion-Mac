# macOS Installation Guide for StreamDiffusion

This guide provides detailed instructions for installing StreamDiffusion on macOS with Apple Silicon (M1/M2/M3) or Intel processors.

## System Requirements

- **macOS**: macOS 12.0 (Monterey) or newer recommended
- **Apple Silicon** (M1/M2/M3/M4) or **Intel-based Mac**
- **Python**: Version 3.10 recommended
- **Memory**: 24GB minimum, 64GB recommended for better performance

## Installation Steps

### Step 1: Clone the repository

```bash
git clone https://github.com/yourusername/StreamDiffusion-Mac.git
cd StreamDiffusion-Mac
```

### Step 2: Set up Python environment

```bash
# Option 1: Create a new conda environment (recommended)
conda create -n streamdiffusion python=3.10
conda activate streamdiffusion

# Option 2: Or use a venv
python -m venv .venv
source .venv/bin/activate
```

### Step 3: Install PyTorch with MPS support

```bash
# Install PyTorch with Metal Performance Shaders (MPS) support
pip install --pre torch torchvision
```

### Step 4: Install StreamDiffusion

```bash
# Install the package with macOS-specific dependencies
pip install -e ".[macos]"

# If you encounter dependency conflicts, you can try:
pip install --upgrade pip
pip install -e . --no-dependencies
pip install diffusers==0.33.1 huggingface_hub==0.32.1 accelerate==1.7.0
```

### Step 5: Verify installation

```bash
# Check if PyTorch has MPS support
python -c "import torch; print(f'PyTorch version: {torch.__version__}, MPS available: {torch.backends.mps.is_available() if hasattr(torch.backends, \"mps\") else False}')"
```

## Running the Demos

### Real-time Image-to-Image Demo

```bash
# Navigate to the demo directory
cd demo/realtime-img2img

# Make the run script executable
chmod +x run_mac.sh

# Run the demo
./run_mac.sh
```

Then open your browser at http://localhost:7860

## Advanced Configuration

### Using TinyVAE for Better Performance

The TinyVAE (TAESD) is strongly recommended for macOS as it significantly improves performance:

```python
from diffusers import AutoencoderTiny

# Add this to your code after creating the StreamDiffusion object
stream.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(device=pipe.device, dtype=pipe.dtype)
```

When running demos, always use the `--taesd` flag:

```bash
python main.py --taesd --acceleration none
```

### Optimizing t_index_list Values for macOS

On macOS, it's recommended to use lower t_index values than the defaults:

```python
# Good values for macOS
stream = StreamDiffusion(
    pipe,
    t_index_list=[15, 25],  # Lower values work better on macOS
    torch_dtype=torch.float16,
)
```

## Troubleshooting

### Common Issues

1. **IndexError with t_index_list**: 
   - On macOS, ensure t_index values are below 30 (e.g., [15, 25])
   - Example error: `IndexError: index out of bounds`
   - Solution: Modify t_index_list in your code or in the demo's img2img.py file

2. **MPS out of memory error**:
   - Error: `RuntimeError: MPS backend out of memory`
   - Solutions:
     - Reduce batch size or image dimensions
     - Close other GPU-intensive applications
     - Try using SD-Turbo instead of larger models
     - Restart your Mac to clear GPU memory

3. **ImportError with update_prompt**:
   - Error: `AttributeError: 'StreamDiffusionWrapper' object has no attribute 'update_prompt'`
   - This has been fixed in this fork by adding the `update_prompt` method 
   - Ensure you're using the correct StreamDiffusionWrapper implementation

4. **Slow performance**:
   - Use the TinyVAE (`--taesd` flag)
   - Lower the resolution in the UI settings
   - Try using SD-Turbo model which is faster than other models
   - Ensure your Mac is plugged in and not in battery-saving mode

5. **Frontend issues**:
   - If the web interface doesn't load, ensure the frontend is built:
   ```bash
   cd demo/realtime-img2img/frontend
   npm i
   npm run build
   ```

### Debug Mode

To see detailed error messages, add the `--debug` flag:

```bash
python main.py --taesd --acceleration none --debug
```

For more detailed troubleshooting, see the [macOS-specific demo README](./demo/realtime-img2img/MACOS_README.md).
# Or use the provided script:
./run_mac.sh
```

Then open your browser at http://localhost:7860
