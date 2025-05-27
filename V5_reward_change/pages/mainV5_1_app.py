import time
import io
import gc
import base64
import streamlit as st
from PIL import Image
import plotly.graph_objects as go
from io import BytesIO
import pandas as pd
import altair as alt
import numpy as np
import math
import os
import matplotlib.pyplot as plt
from environmentV5_app import Scratch_Game_Environment5_Streamlit
from agentV5_1_Qtable_app import RL_Agent_51_Streamlit
from utils.functionalities import plot_results

# ************************************* PAGE CONFIG *************************************
st.set_page_config(page_title="RL_Scratch_Game", page_icon="💵", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""<style>.stApp {background-color: #000000;}.main .block-container {background-color: #000000;}</style>""", unsafe_allow_html=True)

def get_gradient_color(p: int) -> str:
    """De 0% a 100%, devuelve un color hexadecimal de rojo a verde."""
    # Rojo (#d32f2f) → Amarillo (#ffd600) → Verde (#43a047)
    if p < 50:
        # Rojo a amarillo
        r = 211
        g = int(50 + (214 - 50) * (p / 50))
        b = 47
    else:
        # Amarillo a verde
        r = int(211 - (211 - 67) * ((p - 50) / 50))
        g = int(214 + (160 - 214) * ((p - 50) / 50))
        b = int(47 + (71 - 47) * ((p - 50) / 50))
    return f"rgb({r},{g},{b})"

st.session_state.env = Scratch_Game_Environment5_Streamlit(frame_size=50, scratching_area=(0, 0, 700, 350), background_path="../utils/space.jpg")
st.session_state.agent = RL_Agent_51_Streamlit(num_actions=st.session_state.env.total_squares)
env = st.session_state.env
agent = st.session_state.agent

# ************************************* SIDEBAR MENU *************************************
st.sidebar.markdown("""<span style='font-size:24px; font-weight:bold; color:#ffb300; letter-spacing:1px;'>🌟 Menú de navegación</span>""", unsafe_allow_html=True)
page = st.sidebar.radio(
    "",
    [
        "🏠  Inicio",
        "🖼️  Galería de episodios",
        "📈  Estadísticas",
        "⚙️  Ajustes"
    ],
    captions=[
        "Ir a la página principal",
        "Ver la galería de imágenes de episodios",
        "Visualiza los datos y métricas",
        "Configura la aplicación"
    ]
)
# CSS para mejorar el menú lateral
# background: linear-gradient(180deg, #23272f 60%, #f44611 100%);
st.markdown("""
    <style>
        /* Fondo y bordes del sidebar */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #23272f 60%, #f44611 100%);
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
    """, unsafe_allow_html=True)


if page == "Galería de episodios":
    IMAGES_FOLDER = "../episodes"  # Cambia a tu ruta real
    image_files = [f for f in os.listdir(IMAGES_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    st.title("Galería de episodios")
    # Muestra las imágenes en filas de 3 columnas
    cols = st.columns(3)
    for idx, img in enumerate(image_files):
        with cols[idx % 3]:
            st.image(os.path.join(IMAGES_FOLDER, img), use_column_width=True, caption=f"Episodio {idx+1}")

# ************************************* MAIN APP *************************************
if st.button("🔄 Refresh"):
    st.rerun()

st.markdown("""<style>hr:first-of-type {display: none;}</style>""", unsafe_allow_html=True) # hide first horizontal divider
st.markdown("<h1 style='text-align: center;'>Reinforcement Learning applied to custom dynamic environment </h1>", unsafe_allow_html=True)
rainbow_html = """
    <style>
    @keyframes rotateColors {
        0% { color: red; }
        14% { color: orange; }
        28% { color: yellow; }
        42% { color: green; }
        57% { color: blue; }
        71% { color: indigo; }
        85% { color: violet; }
        100% { color: red; }
    }

    .rainbow span {
        font-size: 30px;
        font-weight: bold;
        animation: rotateColors 7s infinite linear;
    }

    .rainbow span:nth-child(1) { animation-delay: -5s; }
    .rainbow span:nth-child(2) { animation-delay: -4s; }
    .rainbow span:nth-child(3) { animation-delay: -3s; }
    .rainbow span:nth-child(4) { animation-delay: -2s; }
    .rainbow span:nth-child(5) { animation-delay: -1s; }
    .rainbow span:nth-child(6) { animation-delay: 0s; }
    </style>

    <div class="rainbow">
        <span>Streamlit</span>
        <span>can</span>
        <span>write</span>
        <span>text</span>
        <span>in</span>
        <span>colors!</span>
    </div>
"""
# st.markdown(rainbow_html, unsafe_allow_html=True)


config_cols = st.columns([1, 0.6, 1])
with config_cols[1]:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<p style='font-size: 24px; font-weight: bold; margin-bottom: 10px; text-align: center;'>Episodes</p>", unsafe_allow_html=True)
        EPISODES = st.number_input(" ", min_value=1, max_value=1000, value=10, step=1, label_visibility="collapsed")
    with col2:
        st.markdown("<p style='font-size: 24px; font-weight: bold; margin-bottom: 10px; text-align: center;'>Trace Interval</p>", unsafe_allow_html=True)
        TRACE = st.number_input(" ", min_value=1, max_value=50, value=1, step=1, label_visibility="collapsed")

game_cols = st.columns([0.3, 0.5, 0.3])
with game_cols[0]:
    st.markdown(f"""
    <style>
        .full-bg-container {{
            position: relative;
            width: 100%;
            min-height: 340px;
            height: 100%;
            background: #f44611;
            border-radius: 20px;
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
            align-items: stretch;
            padding: 36px 32px 36px 32px;
            margin-bottom: 18px;
        }}
        .env-title {{
            text-align: center;
            font-size: 28px;
            color: #222;
            font-weight: bold;
            margin-bottom: 22px;
            margin-top: 0px;
            letter-spacing: 1.2px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 12px;
        }}
        .arrow-container {{
            display: flex;
            justify-content: center;
            align-items: center;
            margin: 24px 0 24px 0;
            height: 40px;
        }}
        @keyframes bounce {{
            0%, 100% {{ transform: translateY(0); }}
            50% {{ transform: translateY(20px); }}
        }}
        @keyframes arrow-glow {{
            0% {{ filter: drop-shadow(0 0 0px #fff8) brightness(1); }}
            50% {{ filter: drop-shadow(0 0 16px #fff) brightness(1.25); }}
            100% {{ filter: drop-shadow(0 0 0px #fff8) brightness(1); }}
        }}
        .arrow-svg-pro {{
            width: 70px; height: 70px; display: block;
            animation: bounce 1.4s infinite, arrow-glow 2s infinite;
        }}
        .squares-row {{
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 22px;
            font-weight: bold;
            color: #222;
            margin-top: 10px;
            justify-content: center;
        }}
    </style>
    <div class="full-bg-container">
        <div class='env-title'>
            <img src="https://em-content.zobj.net/source/animated-noto-color-emoji/356/globe-showing-europe-africa_1f30d.gif" alt="Globe" style="height:36px;vertical-align:middle;margin-right:8px;">
            Env parameters
            <img src="https://em-content.zobj.net/source/animated-noto-color-emoji/356/globe-showing-europe-africa_1f30d.gif" alt="Globe" style="height:36px;vertical-align:middle;margin-left:8px;">
        </div>
        <div class='env-title' style="font-size:23px; margin-bottom:0;">
            Frame size: <strong>{env.FRAME_SIZE}</strong>
        </div>
        <div class="arrow-container">
            <svg class="arrow-svg-pro" viewBox="0 0 70 70" fill="none">
                <defs>
                    <linearGradient id="arrow-gradient" x1="35" y1="10" x2="35" y2="60" gradientUnits="userSpaceOnUse">
                        <stop stop-color="#fff" />
                        <stop offset="1" stop-color="#ffdf70" />
                    </linearGradient>
                </defs>
                <path d="M35 12 V54" stroke="url(#arrow-gradient)" stroke-width="7" stroke-linecap="round"/>
                <polyline points="20,40 35,58 50,40"
                    fill="none" stroke="url(#arrow-gradient)" stroke-width="7" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
        </div>
        <div class='env-title' style="font-size:23px; margin-bottom:0;">
            Total squares: <strong>{env.total_squares}</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)
with game_cols[1]:
    image_placeholder = st.empty()
    image_placeholder.image(env.get_window_image(), use_container_width=True)
with game_cols[2]:
    st.markdown(f"""
    <style>
        .full-bg-container {{
            position: relative;
            width: 100%;
            min-height: 340px;
            height: 100%;
            background: #f44611;
            border-radius: 20px;
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
            align-items: stretch;
            padding: 36px 32px 36px 32px;
            margin-bottom: 18px;
            box-shadow: 0 4px 32px #0002;
        }}
        .env-title {{
            text-align: center;
            font-size: 28px;
            color: #222;
            font-weight: bold;
            margin-bottom: 28px;
            margin-top: 0px;
            letter-spacing: 1.2px;
            font-family: 'Segoe UI', 'Roboto', Arial, sans-serif;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 12px;
        }}
        .custom-list {{
            margin: 0 auto;
            padding-left: 0;
            max-width: 540px;
            list-style: none;
            display: flex;
            flex-direction: column;
            gap: 18px;
        }}
        .custom-list li {{
            font-size: 20px;
            color: #222;
            line-height: 1.5;
            position: relative;
            padding-left: 44px;
            font-family: 'Segoe UI', 'Roboto', Arial, sans-serif;
            font-weight: bold;
        }}
        .custom-list li:hover {{
            background: none;
        }}
        .custom-list li::before {{
            content: '';
            position: absolute;
            left: 10px;
            top: 50%;
            transform: translateY(-50%);
            width: 18px;
            height: 18px;
            background: radial-gradient(circle at 60% 40%, #fff 60%, #ffe07a 100%);
            border-radius: 50%;
            border: 3px solid #d99118;
            box-shadow: 0 0 10px 3px #ffe07a88;
            animation: bullet-glow 1.6s infinite alternate;
        }}
        @keyframes bullet-glow {{
            0% {{ box-shadow: 0 0 10px 3px #ffe07a88; }}
            100% {{ box-shadow: 0 0 22px 8px #ffe07a; }}
        }}
    </style>
    <div class="full-bg-container">
        <div class='env-title'>
            <img src="https://em-content.zobj.net/thumbs/120/animated-noto-color-emoji/356/robot_1f916.gif" alt="Robot" style="height:36px;vertical-align:middle;margin-right:8px;">
            Agent parameters
            <img src="https://em-content.zobj.net/thumbs/120/animated-noto-color-emoji/356/robot_1f916.gif" alt="Robot" style="height:36px;vertical-align:middle;margin-left:8px;">
        </div>
        <ul class="custom-list">
            <li>Learning rate: <strong>{agent.alpha}</strong></li>
            <li>Discount factor: <strong>{agent.gamma}</strong></li>
            <li>Qtable: <strong>{env.total_squares} × {env.total_squares}</strong></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

center_button_col = st.columns([1, 1, 1])[1]
with center_button_col:
    train_button = st.button("🚀 Start Training", use_container_width=True)
    # if st.button("Train Model", use_container_width=True):
        # st.write("Dummy button pressed!")

# train progressbar
percent_placeholder = st.empty()
progress_placeholder = st.empty()
percent_placeholder.markdown("<div style='text-align:center; font-size: 2rem; font-weight:bold; color: #d32f2f; transition: color 0.5s;'>0%</div>", unsafe_allow_html=True)
progress_placeholder.progress(0)

rewards_cols = st.columns([0.7, 0.3])
actions_cols = st.columns([0.7, 0.3])
areas_cols = st.columns([0.7, 0.3])

# EPISODES = 20
# trace = 1
max_reward, min_actions, min_area_scratched = -99999, 99999, 999 # best
min_reward, max_actions, max_area_scratched = 99999, 0, 0        # worst
# path_to_save = f"V5_version/V5_1_Qtable_{env.total_squares}_{EPISODES}_app"

EPSILON = 0.9

# if st.button('Comenzar entrenamiento', type='primary'):
start = time.time()
with rewards_cols[0]:
    st.markdown(
        "<div style='text-align: center;'>"
        "<h1>🏆 Rewards/Episodes 🏆</h1>"
        "</div>",
        unsafe_allow_html=True
    )
    rewards_placeholder = st.empty()
    rewards_df = pd.DataFrame(columns=['Episode', 'Reward', 'Min Reward', 'Max Reward'])
with actions_cols[0]:
    st.markdown(
        "<div style='text-align: center;'>"
        "<h1>🎯 Actions/Episodes 🎯</h1>"
        "</div>",
        unsafe_allow_html=True
    )
    actions_placeholder = st.empty()
    actions_df = pd.DataFrame(columns=['Episode', 'Actions Done', 'Min Actions', 'Max Actions'])
with areas_cols[0]:
    st.markdown(
        "<div style='text-align: center;'>"
        "<h1>🌍 Areas/Episodes 🌍</h1>"
        "</div>",
        unsafe_allow_html=True
    )
    areas_placeholder = st.empty()
    areas_df = pd.DataFrame(columns=['Episode', 'Area Scratched', 'Min Area Scratched', 'Max Area Scratched'])

# """******************************BEGINNING OF TRAINING******************************"""
for i in range(EPISODES):

    env.env_reset()

    done = False
    episode_actions = 0
    episode_reward = 0

    EPSILON *= np.exp(-0.001 * i)

    current_state = env.frames_mask
    current_action = env.total_squares // 2

    while not done:
        episode_actions += 1

        action_index = agent.choose_action(current_action, current_state, EPSILON)
        next_state, reward, done = env.env_step(action_index)
        agent.update_q_table(current_action, action_index, reward, next_state)

        # image_placeholder.image(env.get_window_image(), caption=f'Paso {episode_actions+1}', width=1000)
        # time.sleep(0.1)

        episode_reward += reward
        current_state = next_state

    episode_area = (env.scratched_count / env.total_squares) * 100

    min_reward = episode_reward if episode_reward < min_reward else min_reward
    min_actions = episode_actions if episode_actions < min_actions else min_actions
    min_area_scratched = episode_area if episode_area < min_area_scratched else min_area_scratched
    max_reward = episode_reward if episode_reward > max_reward else max_reward
    max_actions = episode_actions if episode_actions > max_actions else max_actions
    max_area_scratched = episode_area if episode_area > max_area_scratched else max_area_scratched

    if i % TRACE == 0 or i == EPISODES-1:
        dummy_image: Image.Image = env.get_window_image()
        dummy_image.save(f"dummy_image_episode.png")
    
    # ---------------REWARDS EVOLUTION----------------
    rewards_df.loc[len(rewards_df)] = [i + 1, episode_reward, min(episode_reward, min_reward), max(episode_reward, max_reward)] 
    rewards_chart = alt.Chart(rewards_df).transform_fold(
        ['Reward', 'Min Reward', 'Max Reward'], as_=['Serie', 'Valor']
    ).mark_line(point=True).encode(
        x=alt.X('Episode:Q', title='Episode', axis=alt.Axis(tickMinStep=1, labelAngle=0, labelFontSize=18, titleFontSize=22, grid=False)),
        y=alt.Y('Valor:Q', title='Reward', axis=alt.Axis(labelFontSize=18, titleFontSize=22, grid=False), scale=alt.Scale(zero=False)),
        color=alt.Color(
            'Serie:N',
            legend=alt.Legend(title=""),
            scale=alt.Scale(domain=['Reward', 'Min Reward', 'Max Reward'], range=["#06e7f7", "#ff0000", "#15f10e"])
        )
    ).properties(width=1200, height=400, padding={"top": 20}).configure_view(strokeWidth=0).configure_axis(grid=False).interactive()
    rewards_placeholder.altair_chart(rewards_chart, use_container_width=False)
    # ---------------ACTIONS EVOLUTION----------------
    actions_df.loc[len(actions_df)] = [i + 1, episode_actions, min(episode_actions, min_actions), max(episode_actions, max_actions)]
    actions_chart = alt.Chart(actions_df).transform_fold(
        ['Actions Done', 'Min Actions', 'Max Actions'], as_=['Serie', 'Valor']
    ).mark_line(point=True).encode(
        x=alt.X('Episode:Q', title='Episode', axis=alt.Axis(tickMinStep=1, labelAngle=0, labelFontSize=18, titleFontSize=22, grid=False)),
        y=alt.Y('Valor:Q', title='Actions Done', axis=alt.Axis(labelFontSize=18, titleFontSize=22, grid=False), scale=alt.Scale(zero=False)),
        color=alt.Color(
            'Serie:N',
            legend=alt.Legend(title=""),
            scale=alt.Scale(domain=['Actions Done', 'Min Actions', 'Max Actions'], range=["#06e7f7", "#15f10e", "#ff0000"])
        )
    ).properties(width=1200, height=400, padding={"top": 20}).configure_view(strokeWidth=0).configure_axis(grid=False).interactive()
    actions_placeholder.altair_chart(actions_chart, use_container_width=False)
    # ---------------AREAS EVOLUTION----------------
    areas_df.loc[len(areas_df)] = [i + 1, episode_area, min(episode_area, min_area_scratched), max(episode_area, max_area_scratched)]
    areas_chart = alt.Chart(areas_df).transform_fold(
        ['Area Scratched', 'Min Area Scratched', 'Max Area Scratched'], as_=['Serie', 'Valor']
    ).mark_line(point=True).encode(
        x=alt.X('Episode:Q', title='Episode', axis=alt.Axis(tickMinStep=1, labelAngle=0, labelFontSize=18, titleFontSize=22, grid=False)),
        y=alt.Y('Valor:Q', title='Area Scratched (%)', axis=alt.Axis(labelFontSize=18, titleFontSize=22, grid=False), scale=alt.Scale(zero=False)),
        color=alt.Color(
            'Serie:N',
            legend=alt.Legend(title=""),
            scale=alt.Scale(domain=['Area Scratched', 'Min Area Scratched', 'Max Area Scratched'], range=["#06e7f7", "#15f10e", "#ff0000"])
        )
    ).properties(width=1200, height=400, padding={"top": 20}).configure_view(strokeWidth=0).configure_axis(grid=False).interactive()
    areas_placeholder.altair_chart(areas_chart, use_container_width=False)
    # ---------------UPDATE PROGRESS BAR----------------
    percent = int(100 * (i + 1) / EPISODES) 
    color = get_gradient_color(percent)
    percent_html = f"""
        <div style='
            text-align: center; 
            font-size: 2rem; 
            font-weight: bold; 
            color: {color}; 
            transition: color 0.5s;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
        '>
            {percent}% 
            <img src='https://em-content.zobj.net/source/animated-noto-color-emoji/356/rocket_1f680.gif' 
                style='width: 30px; height: 30px; vertical-align: middle;'>
        </div>
    """
    percent_placeholder.markdown(percent_html, unsafe_allow_html=True)
    progress_placeholder.progress(percent)

    time.sleep(0.08)

# """******************************END OF TRAINING******************************"""
finish_html = f"""
<div style='text-align: center;font-size: 2rem;font-weight: bold;color: #43a047;display: flex;align-items: center;justify-content: center;gap: 10px;'>¡Finished!
    <img src='https://em-content.zobj.net/source/animated-noto-color-emoji/356/hundred-points_1f4af.gif' style='width: 30px; height: 30px; vertical-align: middle;'>
</div>
"""
percent_placeholder.markdown(finish_html, unsafe_allow_html=True)
progress_placeholder.progress(100)

# final image
image_placeholder.image(env.get_window_image(), use_container_width=True)

# """******************************STATS GRAPHICS******************************"""
with rewards_cols[1]:
    min_val = rewards_df['Reward'].min()
    max_val = rewards_df['Reward'].max()
    avg_val = rewards_df['Reward'].mean()
    rewards_resume_html = f"""
    <style>
        .wrapper-center {{
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            height: 100%;
            min-height: 500px;
        }}
        .triangle-container {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 0px;
        }}
        .triangle-row {{
            display: flex;
            justify-content: center;
            align-items: flex-end;
            gap: 80px;
            margin-bottom: 0px;
        }}
        .triangle-bottom {{
            display: flex;
            justify-content: center;
            margin-top: 28px;
        }}
        .stat-circle {{
            width: 175px; height: 175px; border-radius: 88px;
            display: flex; flex-direction: column; justify-content: center; align-items: center;
            color: white; font-size: 40px; font-weight: bold;
            box-shadow: 2px 2px 16px #88888866;
            margin: 0 8px;
            position: relative;
        }}
        .circle-min {{
            background: linear-gradient(135deg, #d32f2f 60%, #ff8a65 100%);
            border: 6px solid #d32f2f;
        }}
        .circle-max {{
            background: linear-gradient(135deg, #388e3c 60%, #66bb6a 100%);
            border: 6px solid #388e3c;
        }}
        .circle-avg {{
            background: linear-gradient(135deg, #ff9800 60%, #ffd54f 100%);
            border: 6px solid #ff9800;
        }}
        .stat-number {{
            font-size: 40px;
            font-weight: bold;
            margin-bottom: 10px;   /* Ajusta este valor para subir/bajar el número */
            margin-top: -12px;     /* Puedes ajustar este valor para subirlo más o menos */
            color: black;
            display: flex;
            align-items: center;
            justify-content: center;
            width: 100%;
        }}
        .stat-label {{
            font-size: 19px;
            font-weight: normal;
            margin-top: 4px;
            color: #000000;
            letter-spacing: 0.5px;
            text-align: center;
            width: 100%;
        }}
    </style>
    <div class="wrapper-center">
        <div class="triangle-container">
            <div class="triangle-row">
            <div class="stat-circle circle-min">
                <div class="stat-number">{min_val}💶</div>
                <div class="stat-label">Min</div>
            </div>
            <div class="stat-circle circle-max">
                <div class="stat-number">{max_val}💶</div>
                <div class="stat-label">Max</div>
            </div>
            </div>
            <div class="triangle-bottom">
            <div class="stat-circle circle-avg">
                <div class="stat-number">{avg_val:.0f}💶</div>
                <div class="stat-label">Mean</div>
            </div>
            </div>
        </div>
    </div>
    """
    st.markdown(rewards_resume_html, unsafe_allow_html=True)
with actions_cols[1]:
    min_val = actions_df['Actions Done'].min()
    max_val = actions_df['Actions Done'].max()
    avg_val = actions_df['Actions Done'].mean()
    actions_resume_html = f"""
    <style>
        .wrapper-center {{
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            height: 100%;
            min-height: 500px;
        }}
        .triangle-container {{
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        .triangle-row {{
            display: flex;
            justify-content: center;
            gap: 80px;
            margin-bottom: 30px;
        }}
        .stat-triangle {{
            width: 175px;
            height: 175px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            color: white;
            font-size: 40px;
            font-weight: bold;
            box-shadow: 2px 2px 16px #88888866;
            margin: 0 8px;
            position: relative;
            /* Triángulo equilátero apuntando hacia arriba */
            clip-path: polygon(50% 0%, 0% 100%, 100% 100%);
        }}
        .triangle-min {{
            background: linear-gradient(135deg, #d32f2f 60%, #ff8a65 100%);
            border: 6px solid #d32f2f;
        }}
        .triangle-max {{
            background: linear-gradient(135deg, #388e3c 60%, #66bb6a 100%);
            border: 6px solid #388e3c;
        }}
        .triangle-avg {{
            background: linear-gradient(135deg, #ff9800 60%, #ffd54f 100%);
            border: 6px solid #ff9800;
        }}
        .stat-label {{
            font-size: 19px;
            font-weight: normal;
            margin-top: 7px;
            color: #f0f0f0;
            letter-spacing: 0.5px;
            position: absolute;
            bottom: 10px;
            left: 50%;
            transform: translateX(-50%);
            width: max-content;
        }}
        .stat-value {{
            position: absolute;
            top: 50px;
            left: 50%;
            transform: translateX(-50%);
            z-index: 1;
            color: black;
        }}
    </style>
    <div class="wrapper-center">
        <div class="triangle-container">
            <div class="triangle-row">
            <div class="stat-triangle triangle-min">
                <div class="stat-value">{min_val}</div>
                <div class="stat-label">Min</div>
            </div>
            <div class="stat-triangle triangle-max">
                <div class="stat-value">{max_val}</div>
                <div class="stat-label">Max</div>
            </div>
            </div>
            <div class="stat-triangle triangle-avg" style="margin-top: 40px;">
                <div class="stat-value">{avg_val:.0f}</div>
                <div class="stat-label">Mean</div>
            </div>
        </div>
    </div>
    """
    st.markdown(actions_resume_html, unsafe_allow_html=True)
with areas_cols[1]:
    min_val = areas_df['Area Scratched'].min()
    max_val = areas_df['Area Scratched'].max()
    avg_val = areas_df['Area Scratched'].mean()
    areas_resume_html = f"""
    <style>
        .wrapper-center {{
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            height: 100%;
            min-height: 500px;
        }}
        .square-container {{
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        .square-row {{
            display: flex;
            justify-content: center;
            gap: 80px;
            margin-bottom: 0px;
        }}
        .square-bottom {{
            display: flex;
            justify-content: center;
            margin-top: 28px;
        }}
        .stat-square {{
            width: 175px;
            height: 175px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            color: white;
            font-size: 40px;
            font-weight: bold;
            box-shadow: 2px 2px 16px #88888866;
            margin: 0 8px;
            border-radius: 12px;
            position: relative;
        }}
        .square-min {{
            background: linear-gradient(135deg, #d32f2f 60%, #ff8a65 100%);
            border: 6px solid #d32f2f;
        }}
        .square-max {{
            background: linear-gradient(135deg, #388e3c 60%, #66bb6a 100%);
            border: 6px solid #388e3c;
        }}
        .square-avg {{
            background: linear-gradient(135deg, #ff9800 60%, #ffd54f 100%);
            border: 6px solid #ff9800;
        }}
        .stat-number {{
            font-size: 40px;
            font-weight: bold;
            margin-bottom: 10px;
            margin-top: -12px;
            display: flex;
            align-items: center;
            justify-content: center;
            width: 100%;
            color: black;
        }}
        .stat-label {{
            font-size: 19px;
            font-weight: normal;
            margin-top: 4px;
            color: #f0f0f0;
            letter-spacing: 0.5px;
            text-align: center;
            width: 100%;
        }}
    </style>
    <div class="wrapper-center">
        <div class="square-container">
            <div class="square-row">
            <div class="stat-square square-min">
                <div class="stat-number">{min_val:.1f}%</div>
                <div class="stat-label">Min</div>
            </div>
            <div class="stat-square square-max">
                <div class="stat-number">{max_val:.1f}%</div>
                <div class="stat-label">Max</div>
            </div>
            </div>
            <div class="square-bottom">
            <div class="stat-square square-avg">
                <div class="stat-number">{avg_val:.1f}%</div>
                <div class="stat-label">Mean</div>
            </div>
            </div>
        </div>
    </div>
    """
    st.markdown(areas_resume_html, unsafe_allow_html=True)

minutes, seconds = divmod(time.time()-start, 60)
st.write(f"***** Total training time: {int(minutes)} minutes and {seconds:.2f} seconds *****")

# ************************************* AUTHOR CREDITS *************************************
st.markdown("""
    <style>
        .author-social-row {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0;
            margin: 50px 0 30px 0;
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
            <span class="author-name">Alejandro Mendoza</span>
        </div>
        <div class="social-links">
            <a href="https://github.com/pintamonas4575/RL-model-The-Simpsons" target="_blank">
                <img src="https://github.githubassets.com/favicons/favicon.svg" alt="GitHub">
            </a>
            <a href="https://www.linkedin.com/in/alejandro-mendoza-medina-56b7872a5/" target="_blank">
                <img src="https://static.licdn.com/sc/h/8s162nmbcnfkg7a0k8nq9wwqo" alt="LinkedIn">
        </div>
    </div>
""", unsafe_allow_html=True)


