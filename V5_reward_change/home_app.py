import streamlit as st
import time

st.set_page_config(page_title="Página Principal", page_icon="🏠", layout="wide", initial_sidebar_state="collapsed")


st.markdown("""
    <style>
    @keyframes fade-in {
        0%   { opacity: 0; transform: translateY(32px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    @keyframes fire-glow {
        0%   { box-shadow: 0 0 16px 6px #ff980088, 0 0 32px 12px #ff5722aa, 0 0 48px 24px #ffd60055; }
        50%  { box-shadow: 0 0 28px 14px #ff9800cc, 0 0 44px 20px #ff5722cc, 0 0 60px 32px #ffd60088; }
        100% { box-shadow: 0 0 16px 6px #ff980088, 0 0 32px 12px #ff5722aa, 0 0 48px 24px #ffd60055; }
    }
    .fake-button {
        display: inline-block;
        background: linear-gradient(90deg, #ff9800 0%, #f44336 100%);
        color: black !important;
        font-size: 1.6em;
        font-weight: bold;
        border-radius: 2em;
        padding: 0.75em 2.5em;
        border: none;
        box-shadow: 0 4px 14px rgba(63,81,181,0.3);
        text-decoration: none !important;
        cursor: pointer;
        user-select: none;

        /* aquí indicamos LAS DOS animaciones */
        /* - fade-in: se ejecuta 1 vez al cargar y mantiene estado final (forwards) */
        /* - fire-glow: infinite pero PAUSADA hasta el hover */
        animation:
        fade-in   5s cubic-bezier(.23,1.14,.69,.98) forwards,
        fire-glow 0.7s alternate infinite paused;
        opacity: 0; /* valor inicial, fade-in se encarga de llevarlo a 1 */
    }
    /* Al hacer hover solo arrancamos la segunda animación (fire-glow) */
    .fake-button:hover {
        animation-play-state: paused, running;
    }
    </style>
    <div style="display:flex; justify-content:center; margin-top:2em;">
        <a class="fake-button" href="http://localhost:8501/mainV5_1_app" target="_self">GO TO GLORIOUS HALL</a>
    </div>
""", unsafe_allow_html=True)




