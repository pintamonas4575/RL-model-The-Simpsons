import streamlit as st

st.set_page_config(page_title="Home", page_icon="🏠", layout="wide", initial_sidebar_state="collapsed")
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
st.sidebar.page_link("pages/QL_main_hall.py", icon="🖥️", label="QL Main Hall")
st.sidebar.page_link("pages/DQL_main_hall.py", icon="🖥️", label="DQL Main Hall")
st.sidebar.page_link("pages/trained_model_analysis.py", icon="📊", label="Analyze trained model")
st.sidebar.page_link("pages/test_DQN.py", icon="🤖", label="Test a DQN model")

# ************************************* MAIN BODY *************************************
title_html = """
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@900&display=swap" rel="stylesheet">
    <style>
        .enhanced-title {
            font-family: 'Orbitron', 'Courier New', monospace;
            font-size: 3.2rem;
            font-weight: 900;
            text-align: center;
            margin: 30px 0 40px 0;
            padding: 20px;
            background: linear-gradient(45deg, #000000, #1a1a1a, #000000);
            border-radius: 15px;
            position: relative;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(244, 70, 17, 0.3);
        }
        .title-text {
            background: linear-gradient(90deg, 
                #ff6b35 0%, 
                #f7931e 15%, 
                #ffeb3b 30%, 
                #4caf50 45%, 
                #2196f3 60%, 
                #9c27b0 75%, 
                #e91e63 90%, 
                #ff6b35 100%);
            background-size: 300% 300%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            animation: rainbow-flow 4s ease-in-out infinite;
            filter: drop-shadow(0 0 10px rgba(255, 107, 53, 0.6));
            position: relative;
            z-index: 2;
            letter-spacing: 2px;
            line-height: 1.2;
        }
        .enhanced-title::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, 
                transparent, 
                rgba(255, 255, 255, 0.2), 
                transparent);
            animation: shine-sweep 3s infinite;
            z-index: 1;
        }
        .enhanced-title::after {
            content: '';
            position: absolute;
            top: -2px;
            left: -2px;
            right: -2px;
            bottom: -2px;
            background: linear-gradient(45deg, 
                #ff6b35, #f7931e, #ffeb3b, #4caf50, 
                #2196f3, #9c27b0, #e91e63, #ff6b35);
            background-size: 400% 400%;
            border-radius: 17px;
            z-index: -1;
            animation: border-glow 3s ease-in-out infinite;
        }
        @keyframes rainbow-flow {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        @keyframes shine-sweep {
            0% { left: -100%; }
            100% { left: 100%; }
        }
        @keyframes border-glow {
            0%, 100% { 
                background-position: 0% 50%;
                filter: blur(4px) brightness(1);
            }
            50% { 
                background-position: 100% 50%;
                filter: blur(6px) brightness(1.3);
            }
        }
        .highlight-word {
            display: inline-block;
            animation: pulse-glow 2s ease-in-out infinite alternate;
        }
        @keyframes pulse-glow {
            0% { 
                transform: scale(1);
                filter: drop-shadow(0 0 5px rgba(255, 107, 53, 0.4));
            }
            100% { 
                transform: scale(1.05);
                filter: drop-shadow(0 0 15px rgba(255, 107, 53, 0.8));
            }
        }
    </style>
    <div class="enhanced-title">
        <div class="title-text">
            Reinforcement Learning applied to custom dynamic environment
        </div>
    </div>
"""
st.markdown(title_html, unsafe_allow_html=True)

buttons_html = """
    <style>
        .button-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            grid-template-rows: 1fr 1fr;
            gap: 4em 2em;
            justify-items: center;
            align-items: center;
            margin-top: 2.5em;
            margin-bottom: 2em;
            width: 100%;
            max-width: 700px;
            margin-left: auto;
            margin-right: auto;
        }
        .fake-button {
            display: inline-flex;
            justify-content: center;
            align-items: center;
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
            height: 3.5em;
            min-width: 12em;
            text-align: center;
            line-height: normal;
        }
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
    </style>
    <div class="button-grid">
        <a class="fake-button" href="/QL_main_hall" target="_self">GO TO QL HALL</a>
        <a class="fake-button" href="/DQL_main_hall" target="_self">GO TO DQL HALL</a>
        <a class="fake-button" href="/trained_model_analysis" target="_self">TRAINED MODEL ANALYSIS</a>
        <a class="fake-button" href="/test_DQN" target="_self">DQN TESTING</a>
    </div>
"""
st.markdown(buttons_html, unsafe_allow_html=True)

# ************************************* AUTHOR CREDITS *************************************
author_html = """
    <style>
        .author-social-row {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0;
            margin: 150px 0 30px 0;
            width: 100%;
        }
        .author-card {
            background: #f44611;
            border-radius: 15px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.22);
            padding: 14px 28px;
            min-width: 0;
            width: fit-content;
            text-align: left;
            margin-right: 0;
            flex-shrink: 0;
            display: flex;
            align-items: center;
            justify-content: flex-start;
            gap: 18px;
        }
        .author-title {
            font-size: 24px;
            font-weight: bold;
            color: #fff;
            text-shadow: 0 2px 4px rgba(0,0,0,0.18);
            letter-spacing: 1px;
            margin-bottom: 0;
            margin-right: 8px;
        }
        .author-name {
            font-size: 24px;
            font-weight: bold;
            color: #fff;
            text-shadow: 0 0 14px rgba(255,255,255,0.18);
            margin-bottom: 0;
        }
        .social-links {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 18px;
            margin-left: 24px; /* Separar los iconos del fondo */
        }
        .social-links a img {
            width: 40px; height: 40px; border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .social-links a img:hover {
            transform: scale(1.13);
            box-shadow: 0 6px 20px rgba(0,0,0,0.4);
        }
        @media (max-width: 700px) {
            .author-social-row { flex-direction: column; align-items: center; }
            .author-card { margin-right: 0; margin-bottom: 18px; text-align: center; justify-content: center; }
            .social-links { margin-left: 0; margin-top: 12px; }
        }
    </style>
    <div class="author-social-row">
        <div class="author-card">
            <span class="author-title">Author:</span>
            <span class="author-name">Alejandro Mendoza Medina</span>
        </div>
        <div class="social-links">
            <a href="https://github.com/pintamonas4575/RL-model-The-Simpsons" target="_blank">
                <img src="https://github.githubassets.com/favicons/favicon.svg" alt="GitHub">
            </a>
            <a href="https://www.linkedin.com/in/alejandro-mendoza-medina-56b7872a5/" target="_blank">
                <img src="https://static.licdn.com/sc/h/8s162nmbcnfkg7a0k8nq9wwqo" alt="LinkedIn">
        </div>
    </div>
"""
st.markdown(author_html, unsafe_allow_html=True)

footer_html = """
    <style>
        .footer {
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: rgba(0, 0, 0, 0.8);
            color: #888888;
            text-align: center;
            padding: 20px 0;  /* Change this value to move footer down (increased from 10px) */
            font-size: 0.8em;
            border-top: 1px solid #333;
            margin-top: 20px;  /* Change this value to add more space above footer */
        }
    </style>
    <div class="footer">
        © 2025 Alejandro Mendoza all rights reserved. Made for all the people willing to try their models and visualize their results. 
    </div>
"""
st.markdown(footer_html, unsafe_allow_html=True)