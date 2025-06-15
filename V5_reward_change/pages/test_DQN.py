import io
import time
import torch
import numpy as np
import streamlit as st
from environmentV5_app import Scratch_Game_Environment5_Streamlit
from agentV5_2_DQN_app import Custom_DQN

device = "cpu"

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
st.sidebar.page_link("pages/trained_model_analysis.py", icon="📊", label="Analyze trained model")
st.sidebar.page_link("pages/test_DQN.py", icon="🤖", label="Test a DQN model")

# ************************************* AGENT UPLOAD AND SETUP *************************************
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
        .robot {
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
            <span class="robot">🤖</span>
            <span class="mint">DQN</span>
            <span class="electric"> Model Testing</span>
            <span class="robot">🤖</span>
        </h1>
    </div>
"""
st.markdown(title_html, unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload your trained POLICY DQN file", type=["pt", "pth"])
if uploaded_file is None:
    st.stop()
else:
    bytes_data = uploaded_file.getvalue()
    buffer = io.BytesIO(bytes_data)
    policy_checkpoint = torch.load(buffer, map_location=device)

# NOTE: PROBAR A PASAR UN CARTÓN DIFERENTE AL ENTRENADO
policy_dqn = Custom_DQN(policy_checkpoint["input_dim"], policy_checkpoint["output_dim"]).to(device)
policy_dqn.load_state_dict(policy_checkpoint["policy_dict"])
policy_dqn.eval()

# ************************************* ENV SETUP *************************************
config_cols = st.columns([1, 1, 1])
with config_cols[1]:
    env_config_html = """
        <style>
            .env-config-title {
                font-size: 28px !important;
                font-weight: bold !important;
                margin-bottom: 10px;
                text-align: center;
                letter-spacing: 1px;
                position: relative;
            }
            .env-config-text {
                color: #f94d0b; /* Naranja intenso de tu imagen */
                filter: drop-shadow(0 0 5px rgba(249, 77, 11, 0.5));
                animation: orange-glow 3.5s ease-in-out infinite alternate;
            }
            @keyframes orange-glow {
                0% {
                    text-shadow: 0 0 8px rgba(249, 77, 11, 0.7), 0 0 15px rgba(255, 213, 79, 0.6);
                    filter: drop-shadow(0 0 5px rgba(249, 77, 11, 0.5));
                }
                50% {
                    text-shadow: 0 0 12px rgba(249, 77, 11, 0.9), 0 0 20px rgba(255, 213, 79, 0.8), 0 0 30px rgba(255, 224, 178, 0.6);
                    filter: drop-shadow(0 0 10px rgba(249, 77, 11, 0.7));
                }
                100% {
                    text-shadow: 0 0 8px rgba(249, 77, 11, 0.7), 0 0 15px rgba(255, 213, 79, 0.6);
                    filter: drop-shadow(0 0 5px rgba(249, 77, 11, 0.5));
                }
            }
        </style>
        <p class='env-config-title'><span class='env-config-text'>Test env Config</span> ⚙️</p>
    """
    st.markdown(env_config_html, unsafe_allow_html=True)
    subcols = st.columns(3)
    with subcols[0]:
        st.markdown("<div style='font-size: 18px; font-weight: bold; margin-bottom: 5px; text-align: center;'>Random Emojis</div>", unsafe_allow_html=True)
        RANDOM_EMOJIS = st.selectbox("", options=[True, False], index=1, label_visibility="collapsed")
    with subcols[1]:
        st.markdown("<div style='font-size: 18px; font-weight: bold; margin-bottom: 5px; text-align: center;'>Frame Size</div>", unsafe_allow_html=True)
        frame_size_html = f"""
            <style>
                .mini-banner1 {{
                    width: 80px;
                    background: linear-gradient(90deg, #F47F26, #D96F1E);  /* naranja intenso con degradado */
                    color: white;
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    font-size: 26px;
                    font-weight: 700;
                    text-align: center;
                    padding: 8px 0;
                    border-radius: 8px;
                    box-shadow: 0 2px 6px rgba(244, 127, 38, 0.6);  /* sombra con naranja intenso */
                    margin: 0 auto;
                    letter-spacing: 1px;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 44px;
                    box-sizing: border-box;
                }}
            </style>
            <div class="mini-banner1">{policy_checkpoint["frame_size"]}</div>
        """
        st.markdown(frame_size_html, unsafe_allow_html=True)
    with subcols[2]:
        st.markdown("<div style='font-size: 18px; font-weight: bold; margin-bottom: 5px; text-align: center;'>Number of frames</div>", unsafe_allow_html=True)
        frames_html = f"""
            <style>
                .mini-banner2 {{
                    width: 80px;
                    background: linear-gradient(90deg, #F47F26, #D96F1E);  /* naranja intenso con degradado */
                    color: white;
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    font-size: 26px;
                    font-weight: 700;
                    text-align: center;
                    padding: 8px 0;
                    border-radius: 8px;
                    box-shadow: 0 2px 6px rgba(244, 127, 38, 0.6);  /* sombra con naranja intenso */
                    margin: 0 auto;
                    letter-spacing: 1px;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 44px;
                    box-sizing: border-box;
                }}
            </style>
            <div class="mini-banner2">{policy_checkpoint["input_dim"]}</div>
        """
        st.markdown(frames_html, unsafe_allow_html=True)

env = Scratch_Game_Environment5_Streamlit(frame_size=policy_checkpoint["frame_size"], scratching_area=(0, 0, 700, 350), random_emojis=RANDOM_EMOJIS)
game_cols = st.columns([0.3, 0.5, 0.3])
with game_cols[1]:
    image_placeholder = st.empty()
    image_placeholder.image(env.get_window_image(), use_container_width=True)

# *********************************** 1 EPISODE LOOP *******************************************
st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
start_button_cols = st.columns([1, 1, 1])
with start_button_cols[1]:
    if not st.button("START TRAINING", type="primary", use_container_width=True):
        st.stop()

ACTION_TRACE = 3
done = False
episode_actions = 0
episode_reward = 0

current_state = env.frames_mask.copy()

start = time.time()
while not done:
    episode_actions += 1

    possible_actions = [i for i, val in enumerate(current_state) if val == -1]
    current_state_tensor = torch.FloatTensor(current_state).unsqueeze(0).to(device)  # [batch, num_actions]
    with torch.no_grad():
        q_values: torch.Tensor = policy_dqn(current_state_tensor)
    q_values_np = q_values[0].cpu().numpy()
    masked_q_values = np.full_like(q_values_np, -np.inf)
    masked_q_values[possible_actions] = q_values_np[possible_actions]
    action_index = np.argmax(masked_q_values)

    next_state, reward, done = env.env_step(action_index)
    episode_reward += reward
    current_state = next_state.copy()

    # Update the game image
    image_placeholder.image(env.get_window_image(), use_container_width=True)
    time.sleep(0.05)

episode_area = (env.scratched_count / env.total_squares) * 100

# ************************************* EPISODE RESULTS *************************************
results_html = f"""
    <style>
        .results-container {{
            background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
            border: 2px solid #f44611;
            border-radius: 15px;
            padding: 25px;
            margin: 20px 0;
            box-shadow: 0 8px 32px rgba(244, 70, 17, 0.3);
        }}
        .results-title {{
            font-size: 28px;
            font-weight: bold;
            color: #f44611;
            text-align: center;
            margin-bottom: 20px;
            text-shadow: 0 0 10px rgba(244, 70, 17, 0.5);
        }}
        .results-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        .result-card {{
            background: rgba(244, 70, 17, 0.1);
            border: 1px solid #f44611;
            border-radius: 10px;
            padding: 15px;
            text-align: center;
        }}
        .result-label {{
            font-size: 24px;
            color: #ffb300;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .result-value {{
            font-size: 35px;
            color: #fff;
            font-weight: bold;
            text-shadow: 0 0 5px rgba(255, 255, 255, 0.3);
        }}
    </style>
    <div class="results-container">
        <div class="results-title">🧪 Testing Results 🧪</div>
        <div class="results-grid">
            <div class="result-card">
                <div class="result-label">Episode Reward</div>
                <div class="result-value">{episode_reward:.0f}</div>
            </div>
            <div class="result-card">
                <div class="result-label">Total Actions</div>
                <div class="result-value">{episode_actions}</div>
            </div>
            <div class="result-card">
                <div class="result-label">Area Scratched</div>
                <div class="result-value">{episode_area:.2f}%</div>
            </div>
        </div>
    </div>
"""
st.markdown(results_html, unsafe_allow_html=True)

# ************************************* OTHER PAGE BUTTONS *************************************
buttons_html = """
    <style>
        .button-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr); /* Tres columnas iguales */
            grid-template-rows: 1fr;
            gap: 2em;
            justify-items: center;
            align-items: center;
            margin: 2.5em auto 2em auto;
            width: 100%;
            max-width: 1920px;
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
            min-width: 13em;
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