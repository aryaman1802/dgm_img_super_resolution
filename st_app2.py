import streamlit as st
import torch
from PIL import Image
from diffusers import StableDiffusionUpscalePipeline
import gc

# Title and description
st.title("Image Super Resolution App")
st.write("Upload an image and enhance it with Stable Diffusion Upscaling.")

# Detect device and load the model accordingly
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    st.write("Device detected:", device)
    pipeline = StableDiffusionUpscalePipeline.from_pretrained(
        "stabilityai/stable-diffusion-x4-upscaler", variant="fp16", torch_dtype=torch.float16
    )
    pipeline = pipeline.to("cuda")
else:
    st.write("Device detected:", device)
    st.write("Using a CPU will take a lot of time (~ 1 hour) to enhance the image!")
    st.write("Consider using a GPU instead...")
    st.write("Anyways, the process has started (you can exit anytime)...")
    pipeline = StableDiffusionUpscalePipeline.from_pretrained(
        "stabilityai/stable-diffusion-x4-upscaler", variant="fp16"
    )
    pipeline = pipeline.to("cpu")

# File uploader for the image
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    low_res_img = Image.open(uploaded_file).convert("RGB")
    # Resize for consistency in display; adjust dimensions as needed
    low_res_img = low_res_img.resize((128, 128))
    st.image(low_res_img, caption="Low resolution image", use_column_width=True)
else:
    st.info("Please upload an image to begin.")

# If CPU, let the user enter a prompt for better results (optional)
prompt = ""
if device == "cpu":
    prompt = st.text_input("Enter a prompt describing the image (optional)", "a cat")

# Enhance button and progress bar
if st.button("Enhance"):
    if uploaded_file is None:
        st.warning("Please upload an image before enhancing!")
    else:
        # Initialize progress bar
        progress_bar = st.progress(0)
        
        # Determine total steps (if the scheduler has num_inference_steps, use it; otherwise assume 20)
        total_steps = getattr(pipeline.scheduler, "num_inference_steps", 20)
        
        # Callback to update progress bar
        def progress_callback(step, timestep, latents):
            progress = int((step + 1) / total_steps * 100)
            progress_bar.progress(progress)
        
        # Run the enhancement with a spinner
        with st.spinner("Enhancing image..."):
            result = pipeline(
                prompt=prompt,
                image=low_res_img,
                callback=progress_callback,
                callback_steps=1  # adjust callback frequency if needed
            )
            upscaled_image = result.images[0]
        
        st.success("Enhancement complete!")
        
        # Display the original and upscaled images side by side
        col1, col2 = st.columns(2)
        with col1:
            st.image(low_res_img, caption="Low resolution image", use_column_width=True)
        with col2:
            st.image(upscaled_image, caption="High resolution image", use_column_width=True)
