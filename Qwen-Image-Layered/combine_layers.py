import gradio as gr
from PIL import Image

def combine_images(uploaded_files):
    if not uploaded_files:
        return None

    images = []
    for f in uploaded_files:
        img = Image.open(f.name).convert("RGBA")
        images.append(img)

    min_height = min(img.height for img in images)
    resized_images = [img.resize((int(img.width * min_height / img.height), min_height), Image.LANCZOS) for img in images]

    combined = resized_images[0]
    for img in resized_images[1:]:
        combined = Image.alpha_composite(combined, img)

    return combined

demo = gr.Interface(
    fn=combine_images,
    inputs=gr.File(file_count="multiple", label="Upload images with transparency (PNG) in order."),
    outputs=gr.Image(type="pil", label="Combined Result"),
    title="Layer Blender",
    description="Upload multiple PNG images, and the system will blend them into a single image.",
)

if __name__ == "__main__":
    demo.launch()