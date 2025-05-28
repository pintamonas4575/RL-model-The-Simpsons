import streamlit as st
from PIL import Image

st.set_page_config(page_title="Episode Gallery", page_icon="🦈", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""<style>.stApp {background-color: #000000;}.main .block-container {background-color: #000000;}</style>""", unsafe_allow_html=True)

images: list[tuple[Image.Image, int]] = st.session_state.get("gallery_images", [])

# ----------------------------------------MAIN BODY----------------------------------------
col1, col2, col3 = st.columns([1, 2, 1], border=True)
with col2:
    rainbow_html = """
    <style>
    @keyframes rainbowText {
        0%   { color: #ff5f6d; }
        16%  { color: #ffc371; }
        32%  { color: #47cf73; }
        48%  { color: #1fa2ff; }
        64%  { color: #a259c9; }
        80%  { color: #ff6a00; }
        100% { color: #ff5f6d; }
    }
    .rainbow-text {
        font-size: 2.2rem;
        font-weight: 600;
        font-family: 'Robot', 'Arial', sans-serif;
        letter-spacing: 0.02em;
        word-spacing: 0.12em;
        text-align: center;
        background: linear-gradient(90deg,#ff5f6d,#ffc371,#47cf73,#1fa2ff,#a259c9,#ff6a00,#ff5f6d);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: rainbowBG 8s linear infinite;
        transition: all 0.3s;
        filter: drop-shadow(0 2px 8px rgba(0,0,0,0.08));
        margin-bottom: 0.5em;
    }
    @keyframes rainbowBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    </style>
    <div class="rainbow-text">
        Episode Art Gallery Collection
    </div>
    """
    st.markdown(rainbow_html, unsafe_allow_html=True)
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
