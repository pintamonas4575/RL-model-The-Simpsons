import streamlit as st
import time

st.set_page_config(page_title="Página Principal", page_icon="🏠", layout="wide", initial_sidebar_state="collapsed")


st.markdown("""
    <style>
        @keyframes fade-in {
            0%   { opacity: 0; transform: translateY(32px); }
            100% { opacity: 1; transform: translateY(0); }
        }
        @keyframes pulse-glow {
            0% {
                box-shadow: 0 0 16px 6px #ff980088;
                filter: hue-rotate(0deg);
            }
            50% {
                box-shadow: 0 0 36px 14px #ff5722aa;
                filter: hue-rotate(180deg);
            }
            100% {
                box-shadow: 0 0 16px 6px #ff980088;
                filter: hue-rotate(360deg);
            }
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
            text-decoration: none !important;
            cursor: pointer;
            user-select: none;
            opacity: 0;
            /* 1) fade-in: dura 5s y mantiene estado final (forwards) */
            /* 2) pulse-glow: dura 4s, ciclo infinito, empieza tras 5s (delay) */
            animation:
            fade-in    5s cubic-bezier(.23,1.14,.69,.98) forwards,
            pulse-glow 4s ease-in-out infinite           5s;
        }
        .fake-button:hover {
            /* Si quieres algo extra al hover, lo puedes añadir aquí */
        }
    </style>
    <div style="display:flex; justify-content:center; margin-top:2em;">
        <a class="fake-button"
           href="http://localhost:8501/mainV5_1_app"
           target="_self">
           GO TO GLORIOUS HALL
        </a>
    </div>
""", unsafe_allow_html=True)

