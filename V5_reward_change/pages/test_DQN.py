import io
import time
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from environmentV5_app import Scratch_Game_Environment5_Streamlit
from agentV5_2_DQN_app import RL_Agent_52

# ************************************* PAGE CONFIG *************************************
st.set_page_config(page_title="DQN trained data", page_icon="🤖", layout="wide", initial_sidebar_state="collapsed")
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
st.sidebar.page_link("pages/trained_DQN_analysis.py", icon="📊", label="Analyze DQN training")
st.sidebar.page_link("pages/test_DQN.py", icon="🤖", label="Test a DQN model")

# ************************************* MODEL UPLOAD *************************************
title_html = """
    <style>
        .modern-frame {
            border: 2px solid;
            border-image: linear-gradient(45deg, #000, #ff9800, #000, #ff9800) 1;
            border-radius: 16px;
            padding: 2rem 1.5rem;
            background: rgba(0,0,0,0.8);
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 32px rgba(0,0,0,0.4), inset 0 0 16px rgba(255,152,0,0.2);
            position: relative;
            overflow: hidden;
            margin-bottom: 2rem;
        }
        .modern-frame::after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: repeating-linear-gradient(
                45deg,
                rgba(255,152,0,0.1),
                rgba(255,152,0,0.1) 2px,
                transparent 2px,
                transparent 5px
            );
            animation: spin-aspas 8s linear infinite;
            pointer-events: none;
            z-index: 0;
        }
        @keyframes spin-aspas {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .awesome-title {
            text-align: center;
            font-size: 2.5em;
            font-weight: 900;
            letter-spacing: 0.05em;
            margin: 0;
            font-family: 'Segoe UI', 'Roboto', 'Arial', sans-serif;
            color: #fff;
            position: relative;
            z-index: 2;
        }
        .mint {
            color: #50ffb1;
            text-shadow: 0 0 8px #50ffb199;
        }
        .electric {
            color: #00b4db;
            text-shadow: 0 0 8px #00b4db99;
        }
        .rocket, .brain {
            animation: float 3s ease-in-out infinite;
            display: inline-block;
            font-size: 1.2em;
        }
        @keyframes float {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-10px); }
        }
    </style>
    <div class="modern-frame">
        <h1 class="awesome-title">
            <span class="rocket">🚀</span>
            <span class="mint">DQN</span>
            <span class="electric"> Model Testing</span>
            <span class="brain">🧠</span>
        </h1>
    </div>
"""
st.markdown(title_html, unsafe_allow_html=True)

# drag and drop file upload
uploaded_file = st.file_uploader("Upload your trained DQN model file", type=["pt", "pth"], label_visibility="collapsed", accept_multiple_files=True)
if uploaded_file:
    model_files = [file for file in uploaded_file if file.name.endswith(('.pt', '.pth'))]
    if model_files:
        st.session_state["model_files"] = model_files
        st.success(f"Successfully loaded {len(model_files)} model files!")
    else:
        st.error("No valid model files found. Please upload .pt or .pth files.")


st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
start_button_cols = st.columns([1, 1, 1])
with start_button_cols[1]:
    if not st.button("START TRAINING", type="primary", use_container_width=True):
        st.stop()