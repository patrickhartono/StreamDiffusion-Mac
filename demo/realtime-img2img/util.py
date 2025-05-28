from importlib import import_module
from types import ModuleType
from typing import Dict, Any
from pydantic import BaseModel as PydanticBaseModel, Field
from PIL import Image
import io


def get_pipeline_class(pipeline_name: str) -> ModuleType:
    try:
        module = import_module(f"pipelines.{pipeline_name}")
    except ModuleNotFoundError:
        raise ValueError(f"Pipeline {pipeline_name} module not found")

    pipeline_class = getattr(module, "Pipeline", None)

    if pipeline_class is None:
        raise ValueError(f"'Pipeline' class not found in module '{pipeline_name}'.")

    return pipeline_class


def bytes_to_pil(image_bytes: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(image_bytes))
    return image


def pil_to_frame(image: Image.Image) -> bytes:
    try:
        # Ensure image is in RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Use higher quality JPEG for better visual output
        frame_data = io.BytesIO()
        image.save(frame_data, format="JPEG", quality=95)
        frame_data = frame_data.getvalue()
        
        # Add debug info every 100 frames
        global _frame_counter
        if not '_frame_counter' in globals():
            _frame_counter = 0
        _frame_counter += 1
        if _frame_counter % 100 == 0:
            print(f"Encoded frame {_frame_counter}, size: {len(frame_data)} bytes")
        
        return (
            b"--frame\r\n"
            + b"Content-Type: image/jpeg\r\n"
            + f"Content-Length: {len(frame_data)}\r\n\r\n".encode()
            + frame_data
            + b"\r\n"
        )
    except Exception as e:
        print(f"Error in pil_to_frame: {e}")
        import traceback
        traceback.print_exc()
        
        # Return a transparent 1x1 pixel as emergency fallback
        empty_data = b'--frame\r\nContent-Type: image/png\r\nContent-Length: 68\r\n\r\niVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=\r\n'
        return empty_data


def is_firefox(user_agent: str) -> bool:
    return "Firefox" in user_agent
