# macOS Compatibility Fix for StreamDiffusion

This patch adds macOS compatibility to the StreamDiffusion image-to-image application.

## Changes Made

1. Modified `pipeline.py` to detect macOS and use a fallback timing mechanism instead of CUDA Events
2. Added macOS-specific configurations in `img2img.py`
3. Created a `run_mac.sh` script with optimized parameters for macOS
4. Fixed t_index_list values to be compatible with SD-Turbo model's timesteps on macOS

## How to Run on macOS

1. Use the included `run_mac.sh` script:
   ```bash
   ./run_mac.sh
   ```

2. Or run with these parameters:
   ```bash
   python main.py --taesd --acceleration none --debug
   ```

## Requirements for macOS

- Python 3.9+ recommended
- PyTorch with MPS support
- diffusers package
- streamdiffusion installed as a local package

## Troubleshooting

If you encounter any issues:
1. Ensure you're using the latest PyTorch with MPS support
2. Set `--acceleration none` to disable CUDA-specific optimizations
3. Use `--taesd` to enable TinyVAE for better macOS compatibility
4. Add `--debug` flag to see detailed error messages

### Common Errors

- **IndexError: index out of bounds**: If you see this error, it means the t_index_list values are too high for the model. The default values have been adjusted to work with SD-Turbo on macOS.

For more information, see the main StreamDiffusion documentation.
