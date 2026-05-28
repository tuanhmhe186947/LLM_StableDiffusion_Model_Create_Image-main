"""
Image generation module.

Uses Stable Diffusion v2.1 to create book cover images from
English-language text descriptions.
"""

from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline
from PIL.Image import Image as PILImage

from .config import ImageGenConfig


def load_pipeline(config: ImageGenConfig = ImageGenConfig()) -> StableDiffusionPipeline:
    """Load and optimise the Stable Diffusion pipeline.

    Args:
        config: Image generation configuration.

    Returns:
        A ``StableDiffusionPipeline`` ready on GPU.
    """
    dtype = getattr(torch, config.torch_dtype)

    pipe = StableDiffusionPipeline.from_pretrained(
        config.model_id,
        torch_dtype=dtype,
        use_safetensors=config.use_safetensors,
    )
    pipe = pipe.to("cuda")

    # Memory optimisations for Colab / limited VRAM
    pipe.enable_attention_slicing()
    pipe.enable_sequential_cpu_offload()

    return pipe


def generate_image(
    pipe: StableDiffusionPipeline,
    description_en: str,
    output_path: str = "generated_image.png",
    config: ImageGenConfig = ImageGenConfig(),
) -> PILImage:
    """Generate an image from a textual description.

    Args:
        pipe: A loaded Stable Diffusion pipeline.
        description_en: The English prompt for the image.
        output_path: File path where the image will be saved.
        config: Image generation hyper-parameters.

    Returns:
        The generated PIL ``Image``.
    """
    image: PILImage = pipe(
        description_en,
        num_inference_steps=config.num_inference_steps,
        guidance_scale=config.guidance_scale,
        height=config.height,
        width=config.width,
    ).images[0]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print(f"Image saved to: {output_path}")

    return image
