import streamlit as st
from PIL import Image

st.set_page_config(page_title="Episode Gallery", page_icon="🦈", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""<style>.stApp {background-color: #000000;}.main .block-container {background-color: #000000;}</style>""", unsafe_allow_html=True)

images: list[tuple[Image.Image, int]] = st.session_state.get("gallery_images", [])

# ----------------------------------------MAIN BODY----------------------------------------
gallery_title_cols = st.columns([1, 2, 1], border=False)
with gallery_title_cols[1]:
    rainbow_html = """
    <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@700&display=swap" rel="stylesheet">
    <style>
    .rainbow-text {
        font-size: 2.5rem;
        font-family: 'Montserrat', 'Robot', Arial, sans-serif;
        font-weight: 700;
        letter-spacing: 0.03em;
        word-spacing: 0.12em;
        text-align: center;
        background: linear-gradient(90deg,#00f2fe,#4facfe,#00f2fe,#43e97b,#38f9d7,#fa8bff,#00f2fe);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: rainbowBG 8s linear infinite;
        filter: drop-shadow(0 2px 12px rgba(0,0,0,0.10));
        margin-bottom: 0.5em;
        margin-top: 0.2em;
        transition: all 0.3s;
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
with gallery_title_cols[2]:
    button_html = """
        <style>
            .gallery-button-container {
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100px; /* Ajusta este valor según la altura de la columna */
                min-height: 100px;
                width: 100%;
            }
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
                transition: all 0.3s ease;
            }
            .gallery-button:hover {
                box-shadow: 0 0 32px 12px #ff5722aa;
                transform: scale(1.05);
            }
        </style>
        <div class="gallery-button-container">
            <a class="gallery-button" href="http://localhost:8501/mainV5_1_app" target="_self">← Back to Main Hall</a>
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
