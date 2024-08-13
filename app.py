from typing import Tuple

import random
import numpy as np
import gradio as gr
import spaces
import torch
from PIL import Image
from diffusers import FluxInpaintPipeline

MARKDOWN = """
# FLUX.1 Inpainting 🔥

Shoutout to [Black Forest Labs](https://huggingface.co/black-forest-labs) team for 
creating this amazing model, and a big thanks to [Gothos](https://github.com/Gothos) 
for taking it to the next level by enabling inpainting with the FLUX.
"""

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 2048
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

pipe = FluxInpaintPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16).to(DEVICE)


def resize_image_dimensions(
    original_resolution_wh: Tuple[int, int],
    maximum_dimension: int = 2048
) -> Tuple[int, int]:
    width, height = original_resolution_wh

    if width <= maximum_dimension and height <= maximum_dimension:
        width = width - (width % 32)
        height = height - (height % 32)
        return width, height

    if width > height:
        scaling_factor = maximum_dimension / width
    else:
        scaling_factor = maximum_dimension / height

    new_width = int(width * scaling_factor)
    new_height = int(height * scaling_factor)

    new_width = new_width - (new_width % 32)
    new_height = new_height - (new_height % 32)

    return new_width, new_height


@spaces.GPU(duration=150)
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
        return None

    image = input_image_editor['background']
    mask = input_image_editor['layers'][0]

    if not image:
        gr.Info("Please upload an image.")
        return None

    if not mask:
        gr.Info("Please draw a mask on the image.")
        return None

    width, height = resize_image_dimensions(original_resolution_wh=image.size)
    resized_image = image.resize((width, height), Image.LANCZOS)
    resized_mask = mask.resize((width, height), Image.NEAREST)

    if randomize_seed_checkbox:
        seed_slicer = random.randint(0, MAX_SEED)
    generator = torch.Generator().manual_seed(seed_slicer)
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
    print('INFERENCE DONE')
    return result, resized_mask


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
                    label="Randomize seed", value=False)

                with gr.Row():
                    strength_slider_component = gr.Slider(
                        label="Strength",
                        minimum=0,
                        maximum=1,
                        step=0.01,
                        value=0.75,
                    )

                    num_inference_steps_slider_component = gr.Slider(
                        label="Number of inference steps",
                        minimum=1,
                        maximum=50,
                        step=1,
                        value=20,
                    )
        with gr.Column():
            output_image_component = gr.Image(
                type='pil', image_mode='RGB', label='Generated image')
            with gr.Accordion("Debug", open=False):
                output_mask_component = gr.Image(
                    type='pil', image_mode='RGB', label='Input mask')

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

demo.launch(debug=False, show_error=True)
