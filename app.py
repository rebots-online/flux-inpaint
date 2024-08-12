import torch
import gradio as gr
from diffusers import FluxInpaintPipeline

MARKDOWN = """
# FLUX.1 Inpainting 🔥

Shoutout to [Black Forest Labs](https://huggingface.co/black-forest-labs) team for 
creating this amazing model, and a big thanks to [Gothos](https://github.com/Gothos) 
for taking it to the next level by enabling inpainting with the FLUX.
"""

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

pipe = FluxInpaintPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16).to(DEVICE)


@spaces.GPU(duration=200)
def process(input_image_editor, input_text, progress=gr.Progress(track_tqdm=True)):
    if not input_text:
        gr.Info("Please enter a text prompt.")
        return None

    image = input_image_editor['background']
    mask_image = input_image_editor['layers'][0]

    if not image:
        gr.Info("Please upload an image.")
        return None

    if not mask_image:
        gr.Info("Please draw a mask on the image.")
        return None

    generator = torch.Generator().manual_seed(42)
    return pipe(
        prompt=input_text,
        image=image,
        mask_image=mask_image,
        width=1024,
        height=1024,
        strength=0.9
    ).images[0]


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
                brush=gr.Brush(colors=["#000000"], color_mode="fixed"))
            input_text_component = gr.Textbox(
                label='Text prompt',
                placeholder='Cartoon cactus',)
            submit_button_component = gr.Button(
                value='Submit', variant='primary')
        with gr.Column():
            output_image_component = gr.Image(
                type='pil', image_mode='RGB', label='Generated image')

    submit_button_component.click(
        fn=process,
        inputs=[
            input_image_editor_component,
            input_text_component
        ],
        outputs=[
            output_image_component
        ]
    )

demo.launch(debug=False, show_error=True)
