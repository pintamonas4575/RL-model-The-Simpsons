import streamlit as st
from PIL import Image

st.set_page_config(page_title="Episode Gallery", page_icon="🦈", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""<style>.stApp {background-color: #000000;}.main .block-container {background-color: #000000;}</style>""", unsafe_allow_html=True)

images: list[tuple[Image.Image, int]] = st.session_state.get("gallery_images", [])

# ----------------------------------------MAIN BODY----------------------------------------
col1, col2, col3 = st.columns([1, 2, 1], border=True)
with col2:
    st.markdown("<h1 style='text-align: center;'>Episode Gallery</h1>", unsafe_allow_html=True)
with col3:
    button_html = """
        <style>
            .gallery-button {
                display: inline-block;
                background: linear-gradient(90deg, #ff9800 0%, #f44336 100%);
                color: black !important;
                font-size: 1.6em;
                font-weight: bold;
                border-radius: 2em;
                padding: 0.75em 2.5em;
                border: none;
                text-decoration: none !important;
                cursor: pointer;
                user-select: none;
                box-shadow: 0 0 16px 6px #ff980088;
            }
            .gallery-button:hover {
                box-shadow: 0 0 32px 12px #ff5722aa;
                transform: scale(1.05);
                transition: all 0.3s ease;
            }
        </style>
        <div style="display:flex; justify-content:center; margin-top:2em;">
            <a class="gallery-button" href="http://localhost:8501/mainV5_1_app" target="_self">← Back to Main</a>
        </div>
    """
    st.markdown(button_html, unsafe_allow_html=True)

if not images:
    st.info("No images in gallery. Train a model before")
else:
    num_images = len(images)
    cols_per_row = min(3, num_images)
    cols = st.columns(cols_per_row)
    for i, (img, episode_num) in enumerate(images):
        with cols[i % cols_per_row]:
            st.image(img, use_container_width=True, caption=f"Episode {episode_num}")
        if (i+1) % cols_per_row == 0 and i+1 < num_images:
            cols = st.columns(cols_per_row)
