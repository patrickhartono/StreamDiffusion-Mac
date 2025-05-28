# macOS Compatibility Fix for StreamDiffusion

This patch adds macOS compatibility to the StreamDiffusion image-to-image application and fixes prompt update issues.

## Changes Made

1. **Core Compatibility:**
   - Modified `pipeline.py` to detect macOS and use a fallback timing mechanism instead of CUDA Events
   - Added macOS-specific configurations in `img2img.py`
   - Created a `run_mac.sh` script with optimized parameters for macOS
   - Fixed t_index_list values to be compatible with SD-Turbo model's timesteps on macOS

2. **Dependency Updates:**
   - Updated `huggingface_hub` to version 0.32.1
   - Updated `diffusers` to version 0.33.1
   - Updated `accelerate` to version 1.7.0

3. **Critical Fixes:**
   - Added the missing `update_prompt` method to `StreamDiffusionWrapper` class
   - Updated image processing to use `img2img` method with explicit prompt parameters
   - Fixed import paths to ensure the correct wrapper implementation is used

## How to Run on macOS

1. Install dependencies with the updated versions:
   ```bash
   pip install -r requirements.txt
   ```

2. Build the frontend (if not already built):
   ```bash
   cd frontend && npm i && npm run build && cd ..
   ```

3. Use the included `run_mac.sh` script:
   ```bash
   chmod +x run_mac.sh
   ./run_mac.sh
   ```

4. Or run with these parameters:
   ```bash
   python main.py --taesd --acceleration none --debug
   ```

5. Open your browser at http://localhost:7860

## Requirements for macOS

- Python 3.9+ recommended
- PyTorch 2.0+ with MPS support
- Updated dependencies as specified in requirements.txt:
  - diffusers==0.33.1
  - huggingface_hub==0.32.1
  - accelerate==1.7.0
- streamdiffusion installed as a local package

## Troubleshooting

If you encounter any issues:
1. Ensure you're using the latest PyTorch with MPS support
2. Set `--acceleration none` to disable CUDA-specific optimizations
3. Use `--taesd` to enable TinyVAE for better macOS compatibility
4. Add `--debug` flag to see detailed error messages

### Common Errors

- **IndexError: index out of bounds**: If you see this error, it means the t_index_list values are too high for the model. The default values have been adjusted to work with SD-Turbo on macOS.

- **Prompt updates not taking effect**: If changing the prompt doesn't change the generated image:
  1. Check if the correct `StreamDiffusionWrapper` is being used (from the main project, not a placeholder)
  2. Ensure `img2img.py` is correctly importing the wrapper from the main project path
  3. Use `self.stream.img2img(image=input_image, prompt=self.last_prompt)` instead of `self.stream(image=input_image)`

- **AttributeError: 'StreamDiffusionWrapper' object has no attribute 'update_prompt'**: The `update_prompt` method has been added to the main wrapper class. If you encounter this error, ensure you're using the latest version.

### File Structure Changes

To avoid confusion with duplicate implementations, the placeholder wrapper.py file has been renamed to wrapper.py.bak. The application now correctly imports the real StreamDiffusionWrapper from the main project.

For more information, see the main StreamDiffusion documentation.
