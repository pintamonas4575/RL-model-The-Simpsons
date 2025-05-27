import streamlit as st
import time

st.set_page_config(page_title="Página Principal", page_icon="🏠", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    div.stButton > button {
        background: linear-gradient(90deg, #ff9800 0%, #f44336 100%);
        color: white;
        font-size: 1.6em !important;
        font-weight: normal !important;
        border-radius: 2em;
        padding: 0.75em 2.5em;
        border: none;
        box-shadow: 0 4px 14px rgba(63,81,181,0.3);
        transition: background 0.3s, box-shadow 0.2s, filter 0.2s;
        position: relative;
        z-index: 1;
        overflow: hidden;
    }
    div.stButton > button:hover {
        background: linear-gradient(90deg, #ff9800 0%, #f44336 100%);
        box-shadow:
            0 0 16px 6px #ff980088,
            0 0 32px 12px #ff5722aa,
            0 0 48px 24px #ffd60055;
        filter: brightness(1.08) saturate(1.3);
        animation: fire-glow 0.7s alternate infinite;
        cursor: pointer;
    }
    @keyframes fire-glow {
        0% { box-shadow: 0 0 16px 6px #ff980088, 0 0 32px 12px #ff5722aa, 0 0 48px 24px #ffd60055; }
        50% { box-shadow: 0 0 28px 14px #ff9800cc, 0 0 44px 20px #ff5722cc, 0 0 60px 32px #ffd60088; }
        100% { box-shadow: 0 0 16px 6px #ff980088, 0 0 32px 12px #ff5722aa, 0 0 48px 24px #ffd60055; }
    }
    </style>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns((3,2,3))
with col2:
    st.markdown('<div class="fade-in-button">', unsafe_allow_html=True)
    if st.button("Entrar en la app", use_container_width=True):
        st.switch_page("pages/mainV5_1_app.py")
    st.markdown('</div>', unsafe_allow_html=True)
# -------------------------------------------------------------------------------------------------

st.markdown("""
    <style>
    .fake-button {
        display: inline-block;
        background: linear-gradient(90deg, #ff9800 0%, #f44336 100%);
        color: black;
        font-size: 1.6em;
        font-family: 'Segoe UI', 'Roboto', Arial, sans-serif;
        font-weight: bold;
        border-radius: 2em;
        padding: 0.75em 2.5em;
        border: none;
        box-shadow: 0 4px 14px rgba(63,81,181,0.3);
        transition: background 0.3s, box-shadow 0.2s, filter 0.2s;
        position: relative;
        z-index: 1;
        overflow: hidden;
        text-align: center;
        cursor: pointer;
        margin: 2em auto;
        user-select: none;
    }
    .fake-button:hover {
        background: linear-gradient(90deg, #ff9800 0%, #f44336 100%);
        box-shadow:
            0 0 16px 6px #ff980088,
            0 0 32px 12px #ff5722aa,
            0 0 48px 24px #ffd60055;
        filter: brightness(1.08) saturate(1.3);
        animation: fire-glow 0.7s alternate infinite;
    }
    @keyframes fire-glow {
        0% { box-shadow: 0 0 16px 6px #ff980088, 0 0 32px 12px #ff5722aa, 0 0 48px 24px #ffd60055; }
        50% { box-shadow: 0 0 28px 14px #ff9800cc, 0 0 44px 20px #ff5722cc, 0 0 60px 32px #ffd60088; }
        100% { box-shadow: 0 0 16px 6px #ff980088, 0 0 32px 12px #ff5722aa, 0 0 48px 24px #ffd60055; }
    }
    </style>
    <div style="display: flex; justify-content: center;">
        <div class="fake-button">Botón Falso</div>
    </div>
""", unsafe_allow_html=True)



