import os
import requests
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import numpy as np
from fpdf import FPDF
import io
from dotenv import load_dotenv
import google.generativeai as genai
import base64

load_dotenv()
huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=google_api_key)

def get_gemini_response(prompt):
    model = genai.GenerativeModel('gemini-1.5-flash')
    return model.generate_content(prompt).text

HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/Shakker-Labs/FLUX.1-dev-LoRA-Children-Simple-Sketch"
HUGGINGFACE_HEADERS = {"Authorization": f"Bearer {huggingface_api_key}"}

def query_photorealistic(prompt):
    API_URL = "https://api-inference.huggingface.co/models/ostris/photorealistic-slider-sdxl-lora"
    response = requests.post(API_URL, headers={"Authorization": f"Bearer {huggingface_api_key}"}, json={"inputs": prompt})
    return response.content

def query_watercolor(prompt):
    API_URL = "https://api-inference.huggingface.co/models/ostris/watercolor_style_lora_sdxl"
    response = requests.post(API_URL, headers={"Authorization": f"Bearer {huggingface_api_key}"}, json={"inputs": prompt})
    return response.content

def query_anime(prompt):
    API_URL = "https://api-inference.huggingface.co/models/alvdansen/softserve_anime"
    response = requests.post(API_URL, headers={"Authorization": f"Bearer {huggingface_api_key}"}, json={"inputs": prompt})
    return response.content

def query_vector(prompt):
    API_URL = "https://api-inference.huggingface.co/models/DoctorDiffusion/doctor-diffusion-s-controllable-vector-art-xl-lora"
    response = requests.post(API_URL, headers={"Authorization": f"Bearer {huggingface_api_key}"}, json={"inputs": prompt})
    return response.content

def query_flux(prompt):
    response = requests.post(HUGGINGFACE_API_URL, headers=HUGGINGFACE_HEADERS, json={"inputs": prompt})
    return response.content

def generate_image_from_text(prompt, style):
    if style == "Photo Realistic":
        image_bytes = query_photorealistic(prompt)
    elif style == "Watercolor":
        image_bytes = query_watercolor(prompt)
    elif style == "Anime":
        image_bytes = query_anime(prompt)
    elif style == "Vector":
        image_bytes = query_vector(prompt)
    elif style == "Simple Sketch":
        image_bytes = query_flux(prompt)
    else:
        st.error("Selected style not supported.")
        return None

    try:
        if image_bytes:
            return Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        st.error(f"Error creating image from bytes: {e}")

    return None

def generate_story(image_prompts, age_group, theme, story_length):
    full_story = ""
    
    for prompt in image_prompts:
        story_response = get_gemini_response(f"{prompt} Age group: {age_group}. Theme: {theme}. Story Length: {story_length}")
        if story_response:
            full_story += f"{story_response}\n\n"
        else:
            st.error("Failed to generate a story segment for prompt: " + prompt)
    
    return full_story.strip()

def sanitize_text(text):
    return text.replace('\u2013', '-').replace('\u2014', '--').replace('\u2018', "'").replace('\u2019', "'").replace('\u201c', '"').replace('\u201d', '"')

def create_pdf(story_scenes, generated_images, generated_story):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    for idx, scene in enumerate(story_scenes):
        pdf.add_page()
        pdf.set_font("Times", size=12)
        pdf.multi_cell(0, 10, txt=sanitize_text(f"Scene {idx + 1}: {scene['prompt']}"))

        img_path = f"generated_image_{idx + 1}.png"
        generated_images[idx].save(img_path)
        pdf.image(img_path, x=10, y=30, w=100)

    pdf.add_page()
    pdf.set_font("Times", size=12)
    pdf.multi_cell(0, 10, txt="Generated Story:")
    pdf.multi_cell(0, 10, txt=sanitize_text(generated_story))

    pdf_output_path = "storyboard.pdf"
    pdf.output(pdf_output_path.encode('latin-1', 'replace').decode('latin-1'))
    return pdf_output_path

def query_captioning_model(image):
    API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
    headers = {"Authorization": f"Bearer {huggingface_api_key}"}
    
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    data = buffered.getvalue()

    response = requests.post(API_URL, headers=headers, data=data)
    
    if response.status_code != 200:
        st.error(f"Error from captioning model: {response.status_code} - {response.text}")
        return "Error generating caption."

    response_json = response.json()
    if isinstance(response_json, list):
        return response_json[0] if response_json else "No caption generated."
    elif isinstance(response_json, dict):
        return response_json.get("generated_text", "No caption generated.")
    else:
        return "Unexpected response format."

def storyboard_generation():
    st.title("StoryCraft: Your Imagination Unleashed")
    st.markdown("Merging Creativity and AI for Endless Storytelling")
    
    if 'story_scenes' not in st.session_state:
        st.session_state['story_scenes'] = []
    if 'generated_images' not in st.session_state:
        st.session_state['generated_images'] = []

    age_group = st.selectbox("Select Age Group:", ["Child", "Teen", "Adult", "Senior"])
    theme = st.selectbox("Select Story Theme:", ["Adventure", "Love", "Betrayal", "Heroism", "Courage", "Sacrifice", "Science Fiction", "Romance", "Mystery", "Fantasy", "Thriller", "Historical", "Horror", "Drama", "Comedy", "Action Western", "Crime", "Science", "Fiction"])
    story_length = st.selectbox("Select Word Length:", ["Too Short (150-300 words)", "Short (300-600 words)", "Medium (600-900 words)", "Long (900-1200 words)"])

    user_input = st.text_input("Enter Scene Description:", key="scene_input")

    style = st.selectbox("Select Image Style:", ["Photo Realistic", "Watercolor", "Anime", "Vector", "Simple Sketch"])

    st.subheader("Draw a Scene")
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=2,
        stroke_color="black",
        background_color="white",
        width=400,
        height=400,
        drawing_mode="freedraw",
        key="scene_canvas",
    )

    drawing_image = None
    if canvas_result.image_data is not None:
        drawing_image = Image.fromarray((canvas_result.image_data[:, :, :3]).astype(np.uint8))

    if st.button("Analyze Drawing"):
        if drawing_image:
            try:
                drawing_text = query_captioning_model(drawing_image)
                st.subheader("Drawing Prediction:")
                st.write(drawing_text)
            except Exception as e:
                st.error(f"Error analyzing drawing: {e}")
        else:
            st.error("Please draw something before analyzing.")

    if st.button("Generate Storyboard"):
        if user_input or drawing_image:
            if drawing_image:
                drawing_text = query_captioning_model(drawing_image)
                combined_prompt = f"{user_input} : In this setting, {drawing_text}"
            else:
                combined_prompt = user_input
            
            image_prompts = [combined_prompt]
            generated_story = generate_story(image_prompts, age_group, theme, story_length)

            st.session_state['story_scenes'].append({"prompt": combined_prompt, "drawing": drawing_image})
            generated_images = []

            user_image = generate_image_from_text(user_input, style)
            generated_images.append(user_image)

            if drawing_image:
                drawing_prompt = f"{user_input}  with a visual representation: {drawing_text}"
                drawing_image = generate_image_from_text(drawing_prompt, style)
                generated_images.append(drawing_image)

            st.session_state['generated_images'] = generated_images

            for idx, image in enumerate(generated_images):
                if image:
                    st.image(image, caption=f"Generated Image {idx + 1}", use_column_width=True)
            
            full_story = generate_story(image_prompts, age_group, theme, story_length)
            st.subheader("Generated Story:")
            st.write(full_story)

            pdf_path = create_pdf(st.session_state['story_scenes'], st.session_state['generated_images'], full_story)
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()

            b64_pdf = base64.b64encode(pdf_bytes).decode()
            href = f'<a href="data:application/octet-stream;base64,{b64_pdf}" download="storyboard.pdf">Download Storyboard PDF</a>'
            st.markdown(href, unsafe_allow_html=True)
        else:
            st.error("Please enter a scene description or draw something.")

if __name__ == "__main__":
    storyboard_generation()
