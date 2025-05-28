import sys
import os

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
    )
)

from utils.wrapper import StreamDiffusionWrapper

import torch

from config import Args
from pydantic import BaseModel, Field
from PIL import Image, ImageDraw
import math
import numpy as np

base_model = "stabilityai/sd-turbo"
taesd_model = "madebyollin/taesd"

default_prompt = "Portrait of The Joker halloween costume, face painting, with , glare pose, detailed, intricate, full of colour, cinematic lighting, trending on artstation, 8k, hyperrealistic, focused, extreme details, unreal engine 5 cinematic, masterpiece"
default_negative_prompt = "black and white, blurry, low resolution, pixelated,  pixel art, low quality, low fidelity"

page_content = """<h1 class="text-3xl font-bold">StreamDiffusion</h1>
<h3 class="text-xl font-bold">Image-to-Image SD-Turbo</h3>
<p class="text-sm">
    This demo showcases
    <a
    href="https://github.com/cumulo-autumn/StreamDiffusion"
    target="_blank"
    class="text-blue-500 underline hover:no-underline">StreamDiffusion
</a>
Image to Image pipeline using
    <a
    href="https://huggingface.co/stabilityai/sd-turbo"
    target="_blank"
    class="text-blue-500 underline hover:no-underline">SD-Turbo</a
    > with a MJPEG stream server.
</p>
"""


class Pipeline:
    class Info(BaseModel):
        name: str = "StreamDiffusion img2img"
        input_mode: str = "image"
        page_content: str = page_content

    class InputParams(BaseModel):
        prompt: str = Field(
            default_prompt,
            title="Prompt",
            field="textarea",
            id="prompt",
        )
        # negative_prompt: str = Field(
        #     default_negative_prompt,
        #     title="Negative Prompt",
        #     field="textarea",
        #     id="negative_prompt",
        # )
        width: int = Field(
            512, min=2, max=15, title="Width", disabled=True, hide=True, id="width"
        )
        height: int = Field(
            512, min=2, max=15, title="Height", disabled=True, hide=True, id="height"
        )

    def __init__(self, args: Args, device: torch.device, torch_dtype: torch.dtype):
        params = self.InputParams()
        try:
            print(f"Initializing StreamDiffusionWrapper with device={device}, dtype={torch_dtype}, acceleration={args.acceleration}")
            
            # Check if streamdiffusion module is properly imported
            import streamdiffusion
            print(f"StreamDiffusion module version: {streamdiffusion.__version__ if hasattr(streamdiffusion, '__version__') else 'Unknown'}")
            
            # Enhanced configurations for Mac
            use_tiny_vae = args.taesd
            is_mac = device.type == 'mps'
            
            # Force CPU on Mac if MPS is having issues
            if is_mac and torch.backends.mps.is_available():
                print("Detected Mac with Apple Silicon, applying optimized configurations")
                # Force tiny VAE on Mac for better compatibility
                use_tiny_vae = True
                # Set model parameters for better Mac compatibility
                # SD-Turbo on Mac has only 30 timesteps, so indices must be < 30
                t_index_list = [15, 25]  # Lower indices for Mac compatibility
                frame_buffer_size = 1
                warmup = 5  # Reduced warmup for faster startup
                print("Using MPS device for acceleration")
            else:
                t_index_list = [35, 45]
                frame_buffer_size = 1
                warmup = 10
                if is_mac:
                    print("MPS not available, falling back to CPU")
                
            print(f"Model parameters - tiny_vae: {use_tiny_vae}, t_index: {t_index_list}, warmup: {warmup}")
            
            # Ensure TAESD model is always used on Mac for better performance
            vae_model = taesd_model if (use_tiny_vae or is_mac) else None
            
            # Ensure that models directory exists
            models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", "Model")
            os.makedirs(models_dir, exist_ok=True)
            
            print(f"Base model: {base_model}")
            print(f"VAE model: {vae_model}")
            print(f"Models directory: {models_dir}")
            
            # First ensure we can import all necessary modules
            try:
                from streamdiffusion import StreamDiffusion
                from streamdiffusion.image_utils import postprocess_image
                print("Successfully imported StreamDiffusion modules")
            except ImportError as e:
                print(f"Error importing StreamDiffusion: {e}")
                print("Try reinstalling: pip install -e /Users/patrickhartono/Documents/TD-Experiment/SD/sd-git/StreamDiffusion")
                raise
            
            # Try initializing with huggingface caching
            try:
                import huggingface_hub
                print(f"HuggingFace Hub cache location: {huggingface_hub.constants.HF_HOME}")
            except:
                print("Could not determine HuggingFace cache location")
            
            # Initialize with proper error handling
            self.stream = StreamDiffusionWrapper(
                model_id_or_path=base_model,
                use_tiny_vae=use_tiny_vae,
                device=device,
                dtype=torch_dtype,
                t_index_list=t_index_list,
                frame_buffer_size=frame_buffer_size,
                width=params.width,
                height=params.height,
                use_lcm_lora=False,
                output_type="pil",
                warmup=warmup,
                vae_id=vae_model,
                acceleration="none" if is_mac else args.acceleration,
                mode="img2img",
                use_denoising_batch=True,
                cfg_type="none",
                use_safety_checker=args.safety_checker,
                engine_dir=args.engine_dir,
            )
            print("StreamDiffusionWrapper initialized successfully")
        except Exception as e:
            print(f"Error initializing StreamDiffusionWrapper: {e}")
            import traceback
            traceback.print_exc()
            raise

        self.last_prompt = default_prompt
        
        # Enhanced preparation for Mac
        if device.type == 'mps':
            print("Using optimized parameters for Mac")
            num_inference_steps = 30  # Fewer steps for better performance on Mac
            guidance_scale = 1.0     # Lower guidance scale for more stable results
        else:
            num_inference_steps = 50
            guidance_scale = 1.2
            
        print(f"Preparing model with steps={num_inference_steps}, guidance={guidance_scale}")
        try:
            self.stream.prepare(
                prompt=default_prompt,
                negative_prompt=default_negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )
            print("Model preparation successful")
        except Exception as e:
            print(f"Warning: Error during model preparation: {e}")
            import traceback
            traceback.print_exc()

    def predict(self, params: "Pipeline.InputParams") -> Image.Image:
        # Debug: cek apakah input image ada
        if not hasattr(params, "image") or params.image is None:
            print("No input image provided to predict()")
            return None

        try:
            print(f"Input image dimensions: {params.image.size}")
            
            # Ensure image is in RGB mode for processing
            if params.image.mode != 'RGB':
                params.image = params.image.convert('RGB')
            
            # Resize image if needed to match model's expected dimensions
            target_width, target_height = params.width, params.height
            if params.image.size != (target_width, target_height):
                params.image = params.image.resize((target_width, target_height), Image.LANCZOS)
                print(f"Resized image to {params.image.size}")
            
            # Update the prompt if it has changed
            if hasattr(params, "prompt") and params.prompt != self.last_prompt:
                print(f"Updating prompt from '{self.last_prompt}' to '{params.prompt}'")
                self.stream.update_prompt(params.prompt)
                self.last_prompt = params.prompt
            
            # Try to directly process through StreamDiffusion API
            try:
                # Force use of PIL image directly to avoid tensor conversion issues
                print(f"Processing with current prompt: '{self.last_prompt}'")
                # Make a copy of the input image to prevent any modification issues
                input_image = params.image.copy() 
                output_image = self.stream(image=input_image)
                print(f"Output image processed successfully, type: {type(output_image)}")
                
                if output_image is not None:
                    return output_image
            except Exception as e:
                print(f"Direct processing failed: {e}, trying tensor conversion...")
            
            # Fallback to tensor conversion if direct approach fails
            import numpy as np
            # Convert PIL to numpy array, then to tensor with proper normalization
            image_np = np.array(params.image)
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
            
            # Check if we need to fix tensor dimensions
            if len(image_tensor.shape) == 3:  # Add batch dimension if needed
                image_tensor = image_tensor.unsqueeze(0)
                
            print(f"Tensor shape: {image_tensor.shape}, attempting processing...")
            
            # Force specified prompt and processing
            self.stream.update_prompt(params.prompt)
            output_image = self.stream(image=image_tensor)
            
            if output_image is None:
                print("StreamDiffusionWrapper returned None")
                # Create a placeholder image to indicate processing issue
                from PIL import ImageDraw, ImageFont
                placeholder = Image.new('RGB', (params.width, params.height), color=(30, 30, 50))
                draw = ImageDraw.Draw(placeholder)
                draw.text((10, 10), "StreamDiffusion Error", fill=(255, 0, 0))
                draw.text((10, 30), "Check console for details", fill=(255, 255, 0))
                return placeholder
                
            print(f"Successfully generated output with shape: {output_image.size if hasattr(output_image, 'size') else 'unknown'}")
            return output_image
            
        except Exception as e:
            print(f"Critical error in StreamDiffusionWrapper: {e}")
            import traceback
            traceback.print_exc()
            # Create an error image instead of returning the input image
            from PIL import ImageDraw
            error_img = Image.new('RGB', (params.width, params.height), color=(50, 0, 0))
            draw = ImageDraw.Draw(error_img)
            draw.text((10, 10), f"Error: {str(e)[:50]}", fill=(255, 255, 255))
            return error_img