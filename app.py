from typing import Tuple
from datetime import datetime
import os.path

import requests
import random
import numpy as np
import gradio as gr
import torch
from PIL import Image
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device

from diffusers import FluxInpaintPipeline

MARKDOWN = """
# FLUX.1 Inpainting 🔥

Shoutout to [Black Forest Labs](https://huggingface.co/black-forest-labs) team for 
creating this amazing model, and a big thanks to [Gothos](https://github.com/Gothos) 
for taking it to the next level by enabling inpainting with the FLUX.
"""

MAX_SEED = np.iinfo(np.int32).max
IMAGE_SIZE = 768  # Reduced from 1024 to save memory
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Memory optimization settings
ENABLE_CPU_OFFLOAD = True
ENABLE_ATTENTION_SLICING = True
ENABLE_MODEL_CPU_OFFLOAD = True
ENABLE_SEQUENTIAL_LOADING = True  # Added sequential loading

# Set PyTorch memory allocator settings
torch.cuda.set_per_process_memory_fraction(0.85)  # Use only 85% of available VRAM
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def remove_background(image: Image.Image, threshold: int = 50) -> Image.Image:
    image = image.convert("RGBA")
    data = image.getdata()
    new_data = []
    for item in data:
        avg = sum(item[:3]) / 3
        if avg < threshold:
            new_data.append((0, 0, 0, 0))
        else:
            new_data.append(item)

    image.putdata(new_data)
    return image


EXAMPLES = [
    [
        {
            "background": Image.open(requests.get("https://media.roboflow.com/spaces/doge-2-image.png", stream=True).raw),
            "layers": [remove_background(Image.open(requests.get("https://media.roboflow.com/spaces/doge-2-mask-2.png", stream=True).raw))],
            "composite": Image.open(requests.get("https://media.roboflow.com/spaces/doge-2-composite-2.png", stream=True).raw),
        },
        "little lion",
        42,
        False,
        0.85,
        30
    ],
    [
        {
            "background": Image.open(requests.get("https://media.roboflow.com/spaces/doge-2-image.png", stream=True).raw),
            "layers": [remove_background(Image.open(requests.get("https://media.roboflow.com/spaces/doge-2-mask-3.png", stream=True).raw))],
            "composite": Image.open(requests.get("https://media.roboflow.com/spaces/doge-2-composite-3.png", stream=True).raw),
        },
        "tribal tattoos",
        42,
        False,
        0.85,
        30
    ]
]

# Clear CUDA cache before loading model
torch.cuda.empty_cache()

# Load model components sequentially
pipe = FluxInpaintPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.float16,
    use_safetensors=True,
    low_cpu_mem_usage=True,
)

# Move components to CPU initially
pipe.text_encoder.to("cpu")
pipe.vae.to("cpu")
pipe.unet.to("cpu")
torch.cuda.empty_cache()

# Enable memory optimizations
pipe.enable_attention_slicing(1)
pipe.enable_model_cpu_offload()
pipe.enable_sequential_cpu_offload()

# Enable xformers if available
if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
    pipe.enable_xformers_memory_efficient_attention()

# Use torch.compile for better memory efficiency
if hasattr(torch, 'compile') and torch.cuda.get_device_capability()[0] >= 7:
    pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
    pipe.vae = torch.compile(pipe.vae, mode="reduce-overhead", fullgraph=True)

# Force CUDA to clean up memory again
torch.cuda.empty_cache()

def resize_image_dimensions(
    original_resolution_wh: Tuple[int, int],
    maximum_dimension: int = IMAGE_SIZE
) -> Tuple[int, int]:
    width, height = original_resolution_wh

    if width > height:
        scaling_factor = maximum_dimension / width
    else:
        scaling_factor = maximum_dimension / height

    new_width = int(width * scaling_factor)
    new_height = int(height * scaling_factor)

    new_width = new_width - (new_width % 32)
    new_height = new_height - (new_height % 32)

    return new_width, new_height


def process(
    input_image_editor: dict,
    input_text: str,
    seed_slicer: int,
    randomize_seed_checkbox: bool,
    strength_slider: float,
    num_inference_steps_slider: int,
    progress=gr.Progress(track_tqdm=True)
):
    if not input_text:
        gr.Info("Please enter a text prompt.")
        return None, None

    image = input_image_editor['background']
    mask = input_image_editor['layers'][0]
    original_filename = input_image_editor.get('name', 'image')
    base_filename = os.path.splitext(original_filename)[0]
    date_str = datetime.now().strftime("%d%b%Y").lower()
    new_filename = f"{base_filename}-inpainted-{date_str}.png"

    if not image:
        gr.Info("Please upload an image.")
        return None, None

    if not mask:
        gr.Info("Please draw a mask on the image.")
        return None, None

    # Clear CUDA cache before processing
    torch.cuda.empty_cache()

    width, height = resize_image_dimensions(original_resolution_wh=image.size)
    resized_image = image.resize((width, height), Image.LANCZOS)
    resized_mask = mask.resize((width, height), Image.LANCZOS)

    if randomize_seed_checkbox:
        seed_slicer = random.randint(0, MAX_SEED)
    generator = torch.Generator(device=DEVICE).manual_seed(seed_slicer)
    
    try:
        # Move required components to GPU just before inference
        pipe.text_encoder.to(DEVICE)
        pipe.unet.to(DEVICE)
        
        with torch.inference_mode(), torch.autocast(DEVICE):
            result = pipe(
                prompt=input_text,
                image=resized_image,
                mask_image=resized_mask,
                width=width,
                height=height,
                strength=strength_slider,
                generator=generator,
                num_inference_steps=num_inference_steps_slider
            ).images[0]
    finally:
        # Move components back to CPU after inference
        pipe.text_encoder.to("cpu")
        pipe.unet.to("cpu")
        torch.cuda.empty_cache()
    
    print('INFERENCE DONE')
    return (result, {"name": new_filename}), resized_mask


with gr.Blocks() as demo:
    gr.Markdown(MARKDOWN)
    with gr.Row():
        with gr.Column():
            input_image_editor_component = gr.ImageEditor(
                label='Image',
                type='pil',
                sources=["upload", "webcam"],
                image_mode='RGB',
                layers=False,
                brush=gr.Brush(colors=["#FFFFFF"], color_mode="fixed"))

            with gr.Row():
                input_text_component = gr.Text(
                    label="Prompt",
                    show_label=False,
                    max_lines=1,
                    placeholder="Enter your prompt",
                    container=False,
                )
                submit_button_component = gr.Button(
                    value='Submit', variant='primary', scale=0)

            with gr.Accordion("Advanced Settings", open=False):
                seed_slicer_component = gr.Slider(
                    label="Seed",
                    minimum=0,
                    maximum=MAX_SEED,
                    step=1,
                    value=42,
                )

                randomize_seed_checkbox_component = gr.Checkbox(
                    label="Randomize seed", value=True)

                with gr.Row():
                    strength_slider_component = gr.Slider(
                        label="Strength",
                        info="Indicates extent to transform the reference `image`. "
                             "Must be between 0 and 1. `image` is used as a starting "
                             "point and more noise is added the higher the `strength`.",
                        minimum=0,
                        maximum=1,
                        step=0.01,
                        value=0.85,
                    )

                    num_inference_steps_slider_component = gr.Slider(
                        label="Number of inference steps",
                        info="The number of denoising steps. More denoising steps "
                             "usually lead to a higher quality image at the",
                        minimum=1,
                        maximum=50,
                        step=1,
                        value=20,
                    )
        with gr.Column():
            output_image_component = gr.Image(
                type='pil', image_mode='RGB', label='Generated image', format="png")
            with gr.Accordion("Debug", open=False):
                output_mask_component = gr.Image(
                    type='pil', image_mode='RGB', label='Input mask', format="png")
    with gr.Row():
        gr.Examples(
            fn=process,
            examples=EXAMPLES,
            inputs=[
                input_image_editor_component,
                input_text_component,
                seed_slicer_component,
                randomize_seed_checkbox_component,
                strength_slider_component,
                num_inference_steps_slider_component
            ],
            outputs=[
                output_image_component,
                output_mask_component
            ],
            run_on_click=True,
            cache_examples=True
        )

    submit_button_component.click(
        fn=process,
        inputs=[
            input_image_editor_component,
            input_text_component,
            seed_slicer_component,
            randomize_seed_checkbox_component,
            strength_slider_component,
            num_inference_steps_slider_component
        ],
        outputs=[
            output_image_component,
            output_mask_component
        ]
    )

demo.launch(debug=True, share=True, server_name='0.0.0.0', show_error=True)

