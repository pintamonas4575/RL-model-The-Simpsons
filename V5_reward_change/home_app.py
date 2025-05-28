import streamlit as st

st.set_page_config(page_title="Página Principal", page_icon="🏠", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""<style>.stApp {background-color: #000000;}.main .block-container {background-color: #000000;}</style>""", unsafe_allow_html=True)

# ************************************* SIDEBAR MENU *************************************
st.sidebar.markdown("""<div style='text-align:center;'><span style='font-size:24px; font-weight:bold; color:#ffb300; letter-spacing:1px;'>🌟 MENU 🌟</span></div>""", unsafe_allow_html=True)

side_bar_html = """
    <style>
        /* Fondo y bordes del sidebar */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #000000 60%, #f44611 100%);
            border-radius: 0 20px 20px 0;
            box-shadow: 2px 0 18px #0002;
        }
        /* Radio button */
        [data-testid="stSidebar"] .stRadio [role="radiogroup"] > div {
            background: rgba(255,255,255,0.10);
            border-radius: 12px;
            margin-bottom: 10px;
            transition: background 0.2s;
            box-shadow: 0 1px 6px #0001;
            padding: 10px 16px;
        }
        [data-testid="stSidebar"] .stRadio [role="radiogroup"] > div:hover {
            background: rgba(255,255,255,0.18);
        }
        /* Texto de opciones */
        [data-testid="stSidebar"] .stRadio [role="radio"] p {
            color: #fff !important;
            font-weight: 600;
            font-size: 18px;
            letter-spacing: 0.5px;
        }
        /* Captions */
        [data-testid="stSidebar"] .stRadio [data-testid="stCaption"] {
            color: #ffe07a !important;
            font-size: 13px !important;
            margin-top: 2px;
            margin-left: 6px;
            font-style: italic;
        }
        /* Círculo de selección */
        [data-testid="stSidebar"] .stRadio [role="radio"] span[aria-checked] {
            border: 2px solid #ffe07a !important;
            box-shadow: 0 0 8px #ffe07a44;
        }
        [data-testid="stSidebar"] .stRadio [role="radio"][aria-checked="true"] span[aria-checked] {
            background: #ffe07a !important;
            border: 2px solid #fff !important;
        }
    </style>
"""
st.markdown(side_bar_html, unsafe_allow_html=True)

st.sidebar.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
st.sidebar.page_link("home_app.py", icon="🏠", label="Home")
st.sidebar.page_link("pages/mainV5_1_app.py", icon="🖥️", label="Main Hall")
st.sidebar.page_link("pages/gallery_app.py", icon="🖼️", label="Episode Gallery")

# ************************************* MAIN BODY *************************************
animation_html = """
    <style>
        .bg {
            position: relative;
            width: 100%;
            height: 300px;
            overflow: hidden;
            background: #000000;
        }
        .circle {
            position: absolute;
            border-radius: 50%;
            opacity: 0.6;
        }
        .c1 { width: 120px; height: 120px; background: #ff9800; left: 10%;  top: 120%; animation: rise 3s ease-out forwards; }
        .c2 { width: 150px; height: 150px; background: #f44336; left: 50%;  top: 120%; animation: rise 3s ease-out 1s forwards; }
        .c3 { width:  80px; height:  80px; background: #ffd600; left: 80%;  top: 120%; animation: rise 3s ease-out 2s forwards; }
        @keyframes rise {
            0%   { transform: translateY(0); }
            80%  { transform: translateY(-140%); }
            100% { transform: translateY(-140%); }
        }

        /* 2) Botón: fade-in tras terminar la animación de fondo */
        @keyframes btn-fade {
            from { opacity: 0; }
            to   { opacity: 1; }
        }
        .btn-container {
            opacity: 0;
            animation: btn-fade 2s ease-in forwards;
            animation-delay: 4s;  /* Empieza 4s después */
            display: flex;
            justify-content: center;
            margin-top: 2em;
        }
    </style>
    <div class="bg">
        <div class="circle c1"></div>
        <div class="circle c2"></div>
        <div class="circle c3"></div>
    </div>
"""
st.markdown(animation_html, unsafe_allow_html=True)

button_html = """
    <style>
        @keyframes fade-in {
            0% {
                opacity: 0;
                transform: translateY(32px);
                box-shadow:
                    0 0 0 0 #ff980000,
                    0 0 0 0 #ff572200,
                    0 0 0 0 #ffd60000;
                filter: hue-rotate(0deg);
            }
            100% {
                opacity: 1;
                transform: translateY(0);
                box-shadow:
                    0 0 16px 6px #ff980088,
                    0 0 32px 12px #ff5722aa,
                    0 0 48px 24px #ffd60055;
                filter: hue-rotate(0deg);
            }
        }
        @keyframes pulse-glow {
            0% {
                box-shadow:
                    0 0 16px 6px #ff980088,
                    0 0 32px 12px #ff5722aa,
                    0 0 48px 24px #ffd60055;
                filter: hue-rotate(0deg);
            }
            50% {
                box-shadow:
                    0 0 36px 14px #ff5722aa,
                    0 0 44px 20px #ff5722cc,
                    0 0 60px 32px #ffd60088;
                filter: hue-rotate(180deg);
            }
            100% {
                box-shadow:
                    0 0 16px 6px #ff980088,
                    0 0 32px 12px #ff5722aa,
                    0 0 48px 24px #ffd60055;
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
            box-shadow:
                0 0 0 0 #ff980000,
                0 0 0 0 #ff572200,
                0 0 0 0 #ffd60000;
            filter: hue-rotate(0deg);
            animation:
                fade-in    3s cubic-bezier(.23,1.14,.69,.98) forwards,
                pulse-glow 4s ease-in-out infinite           3s;
        }
    </style>
    <div style="display:flex; justify-content:center; margin-top:2em;">
        <a class="fake-button" href="http://localhost:8501/mainV5_1_app" target="_self">GO TO GLORIOUS HALL</a>
    </div>
"""
st.markdown(button_html, unsafe_allow_html=True)