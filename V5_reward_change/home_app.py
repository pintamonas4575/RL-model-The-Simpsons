import streamlit as st
import time

st.set_page_config(page_title="Página Principal", page_icon="🏠", layout="wide", initial_sidebar_state="collapsed")



st.markdown("""
    <style>
    .fake-button {
        display: inline-block;
        background: linear-gradient(90deg, #ff9800 0%, #f44336 100%);
        color: black !important;
        font-size: 1.6em;
        font-family: 'Roboto', sans-serif;
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
        text-decoration: none !important; /* Quita subrayado */
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
        <a class="fake-button" href="http://localhost:8501/mainV5_1_app" target="_self">GO TO GLORIOUS HALL</a>
    </div>
""", unsafe_allow_html=True)



