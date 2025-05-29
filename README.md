# StreamDiffusion for macOS

⚠️ **Important Notice:**  
This repository is a **macOS-only fork** of the original [StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion) project.  
It has been specifically modified to work on **macOS**, supporting both Apple Silicon (M1/M2/M3) and Intel-based Macs.  
  
🔧 **Current limitation:** Only the **img2img** pipeline has been tested and verified to work on macOS. The txt2img demo is not yet fully supported.  
  
✅ **Quick Start for macOS Users:**  
- See [macOS installation guide](./INSTALL.md)  
- See [macOS-specific demo instructions](./demo/realtime-img2img/MACOS_README.md)

<p align="center">
  <img src="./assets/demo_07.gif" width=90%>
</p>

# StreamDiffusion for macOS: Real-Time Interactive Generation

This is a macOS-specific fork of StreamDiffusion, an innovative diffusion pipeline designed for real-time interactive generation. The original project was developed by the StreamDiffusion team, and this version has been specifically adapted to work on macOS.

[![arXiv](https://img.shields.io/badge/arXiv-2312.12491-b31b1b.svg)](https://arxiv.org/abs/2312.12491)
[![macOS Support](https://img.shields.io/badge/platform-macOS-lightgrey)](https://github.com/cumulo-autumn/StreamDiffusion)

## macOS-Specific Features

1. **Native macOS Support**
   - Works on Apple Silicon (M1/M2/M3/M4) and Intel-based Macs
   - Uses Metal Performance Shaders (MPS) for GPU acceleration

2. **Optimized for macOS Performance**
   - Modified pipeline with fallback timing mechanism instead of CUDA Events
   - Adjusted t_index_list values for compatibility with macOS

3. **Simple Installation**
   - Streamlined dependencies for macOS
   - Easy setup with Python and PyTorch

4. **Real-Time Image Generation**
   - Interactive img2img pipeline with webcam feed or screen capture
   - Support for prompt-based image manipulation

## System Requirements

- **macOS**: macOS 12.0 (Monterey) or newer recommended
- **Hardware**: Apple Silicon (M1/M2/M3/M4) or Intel-based Mac
- **Python**: Version 3.10 recommended
- **Node.js & npm**: Required for building the demo frontend (install from [nodejs.org](https://nodejs.org/))
- **Memory**: 16GB RAM minimum recommended for better performance

## macOS Performance

While not as fast as NVIDIA GPUs with TensorRT, this macOS port still provides interactive speeds.

> Note: Performance may vary based on your specific Mac hardware, model settings, and image resolution.

## Installation

### Quick Start

For detailed installation instructions, see our [macOS Installation Guide](./INSTALL.md).

### Step 1: Clone this repository

```bash
git clone https://github.com/patrickhartono/StreamDiffusion-Mac.git
cd StreamDiffusion-Mac
```

### Step 2: Set up Python environment

```bash
# Create a new conda environment
conda create -n streamdiffusion python=3.10
conda activate streamdiffusion

# Or use a venv
python -m venv .venv
source .venv/bin/activate
```

### Step 3: Install PyTorch for macOS

```bash
# Install PyTorch with MPS support
pip install --pre torch torchvision
```

### Step 4: Install StreamDiffusion

```bash
# Install the package with macOS-specific dependencies
pip install -e ".[macos]"
```

## Quick Start for macOS

Here's a complete minimal example to get you started with StreamDiffusion on macOS:

```bash
# 1. Clone the repository
git clone https://github.com/patrickhartono/StreamDiffusion-Mac.git
cd StreamDiffusion-Mac

# 2. Set up Python environment (choose one)
# Option A: With conda
conda create -n streamdiffusion python=3.10
conda activate streamdiffusion

# Option B: With venv
python -m venv .venv
source .venv/bin/activate

# 3. Install PyTorch for macOS
pip install --pre torch torchvision

# 4. Install StreamDiffusion
pip install -e ".[macos]"

# 5. Run the demo
cd demo/realtime-img2img
pip install -r requirements.txt
cd frontend && npm i && npm run build && cd ..
chmod +x run_mac.sh
./run_mac.sh
```

Then open your browser at http://localhost:7860

## Running the Demo

To run the realtime img2img demo:

```bash
# Navigate to the demo directory
cd demo/realtime-img2img

# Install additional dependencies required for the demo
pip install -r requirements.txt

# Build the frontend (required the first time, needs Node.js/npm installed)
cd frontend && npm i && npm run build && cd ..

# Make the run script executable and run it
chmod +x run_mac.sh
./run_mac.sh
```

Then open your browser at http://localhost:7860

### Demo Features

- Real-time image generation from webcam or screen capture
- Adjustable settings for generation quality and speed
- Prompt-based image manipulation
- Support for custom models

For detailed instructions, see the [macOS-specific demo README](./demo/realtime-img2img/MACOS_README.md).

Here's a simple example of how to use StreamDiffusion for image-to-image generation on macOS:

### Image-to-Image Example

```python
import torch
from diffusers import AutoencoderTiny, StableDiffusionPipeline
from diffusers.utils import load_image

from streamdiffusion import StreamDiffusion
from streamdiffusion.image_utils import postprocess_image

# Load model with appropriate settings for macOS
pipe = StableDiffusionPipeline.from_pretrained("stabilityai/sd-turbo").to(
    device=torch.device("mps" if torch.backends.mps.is_available() else "cpu"),
    dtype=torch.float16,
)

# Use t_index values that work well on macOS
stream = StreamDiffusion(
    pipe,
    t_index_list=[15, 25],  # Lower values for macOS compatibility
    torch_dtype=torch.float16,
)

# If the loaded model is not LCM, merge LCM
stream.load_lcm_lora()
stream.fuse_lora()
# Use Tiny VAE for better performance on macOS
stream.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(device=pipe.device, dtype=pipe.dtype)

prompt = "1girl with dog hair, thick frame glasses"
# Prepare the stream
stream.prepare(prompt)

# Prepare image (included in the repository)
init_image = load_image("assets/img2img_example.png").resize((512, 512))

# Warmup >= len(t_index_list) x frame_buffer_size
for _ in range(2):
    stream(init_image)

# Run the stream
for i in range(10):
    x_output = stream(init_image)
    output_image = postprocess_image(x_output, output_type="pil")[0]
    output_image.save(f"output_{i}.png")
    output_image.show()
    input("Press Enter to generate next image...")
```

For more detailed examples, please refer to the [`examples`](./examples) directory.

## Real-Time Img2Img Demo

The real-time img2img demo with webcam feed or screen capture is the main focus of this macOS port. This demo runs in a web browser and allows you to apply Stable Diffusion models to your webcam or screen in real-time.

<p align="center">
  <img src="./assets/img2img1.gif" width=100%>
</p>

### Running the Demo

```bash
cd demo/realtime-img2img
chmod +x run_mac.sh
./run_mac.sh
```

Then open your browser at http://localhost:7860

### Demo Features

- Real-time image generation from webcam or screen capture
- Adjustable settings for generation quality and speed
- Prompt-based image manipulation
- Support for custom models

### Required Models

The demo automatically downloads the necessary models from Hugging Face, but if you want to use specific models:

1. **SD-Turbo** (recommended for macOS): Used by default, provides faster performance
   - Downloaded automatically from `stabilityai/sd-turbo`

2. **KohakuV2** (optional): Can be used for higher quality outputs but slower
   - Can be downloaded from [Hugging Face](https://huggingface.co/KBlueLeaf/kohaku-v2.1)
   
3. **LCM-LoRA** (optional): Automatically merged if the base model is not LCM
   - Downloaded automatically from `latent-consistency/lcm-lora-sdv1-5`

To use custom models, place them in the appropriate folders:
- Base models: `models/Model/`
- LoRA weights: `models/LoRA/`
- LCM LoRA weights: `models/LCM_LoRA/`

For detailed instructions, see the [macOS-specific demo README](./demo/realtime-img2img/MACOS_README.md).

## Troubleshooting macOS Issues

### Common Issues on macOS

1. **IndexError: index out of bounds**
   - This typically means the t_index_list values are too high for macOS.
   - Solution: Use lower values like `t_index_list=[15, 25]` instead of the default values.

2. **Slow performance**
   - Make sure you're using the TinyVAE (`--taesd` flag)
   - Lower the resolution in the UI settings
   - Try using SD-Turbo model instead of larger models

3. **"RuntimeError: MPS backend out of memory"**
   - Reduce batch size or image dimensions
   - Close other GPU-intensive applications
   - Restart your computer to clear GPU memory

4. **Prompt updates not taking effect**
   - This has been fixed in this fork by adding the `update_prompt` method to the StreamDiffusionWrapper class

5. **SD-Turbo Model Variant Warning**
   - You may see a warning message: `Some weights of the model checkpoint were not used...`
   - This is normal and can be safely ignored
   - The warning occurs because SD-Turbo has some structural differences from standard SD models

### Checking MPS Support

To verify that your Mac supports MPS acceleration, run:

```python
import torch
print(f"MPS available: {torch.backends.mps.is_available()}")
```

### Example of Successful Setup

When everything is working correctly, you should see output similar to this when starting the demo:

```
Verifying Python environment...
Using Python: Python 3.10.x
Using pip: pip 23.x.x from ...

Checking required libraries...
PyTorch version: 2.x.x, CUDA available: False, MPS available: True

PYTHONPATH set to: /Users/username/StreamDiffusion-Mac

Starting StreamDiffusion with Mac-optimized settings...
Running on local URL:  http://127.0.0.1:7860
```

The web interface should load and display controls for adjusting prompts, image settings, and generation parameters.

## Credits and Acknowledgments

This macOS port is based on the original [StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion) project.

The original StreamDiffusion was developed by Akio Kodaira, Chenfeng Xu, Toshiki Hazama, Takanori Yoshimoto, Kohei Ohno, Shogo Mitsuhori, Soichi Sugano, Hanying Cho, Zhijian Liu, and Kurt Keutzer.

For the full research paper, see [StreamDiffusion: A Pipeline-level Solution for Real-time Interactive Generation](https://arxiv.org/abs/2312.12491).

The macOS compatibility fixes and this fork are maintained by [@patrickhartono](https://github.com/patrickhartono).

## License

This project is licensed under the original StreamDiffusion license.

## Development Team

[Aki](https://twitter.com/cumulo_autumn),
[Ararat](https://twitter.com/AttaQjp),
[Chenfeng Xu](https://twitter.com/Chenfeng_X),
[ddPn08](https://twitter.com/ddPn08),
[kizamimi](https://twitter.com/ArtengMimi),
[ramune](https://twitter.com/__ramu0e__),
[teftef](https://twitter.com/hanyingcl),
[Tonimono](https://twitter.com/toni_nimono),
[Verb](https://twitter.com/IMG_5955),

(\*alphabetical order)
</br>

## Acknowledgements

The video and image demos in this GitHub repository were generated using [LCM-LoRA](https://huggingface.co/latent-consistency/lcm-lora-sdv1-5) + [KohakuV2](https://civitai.com/models/136268/kohaku-v2) and [SD-Turbo](https://arxiv.org/abs/2311.17042).

Special thanks to [LCM-LoRA authors](https://latent-consistency-models.github.io/) for providing the LCM-LoRA and Kohaku BlueLeaf ([@KBlueleaf](https://twitter.com/KBlueleaf)) for providing the KohakuV2 model and ,to [Stability AI](https://ja.stability.ai/) for [SD-Turbo](https://arxiv.org/abs/2311.17042).

KohakuV2 Models can be downloaded from [Civitai](https://civitai.com/models/136268/kohaku-v2) and [Hugging Face](https://huggingface.co/KBlueLeaf/kohaku-v2.1).

SD-Turbo is also available on [Hugging Face Space](https://huggingface.co/stabilityai/sd-turbo).

## Contributors

<a href="https://github.com/cumulo-autumn/StreamDiffusion/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=cumulo-autumn/StreamDiffusion" />
</a>
