import io
import time
import math
import zipfile
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from PIL import Image
from environmentV5_app import Scratch_Game_Environment5_Streamlit
from agentV5_1_Qtable_app import RL_Agent_51_Streamlit

# ************************************* UTILS FUNCTIONS *************************************
def get_gradient_color(p: int) -> str:
    """From 0% to 100%, returns a hexadecimal color from red to green."""
    # Red (#d32f2f) → Yellow (#ffd600) → Green (#43a047)
    if p < 50:
        # Red to yellow
        r = 211
        g = int(50 + (214 - 50) * (p / 50))
        b = 47
    else:
        # Yellow to green
        r = int(211 - (211 - 67) * ((p - 50) / 50))
        g = int(214 + (160 - 214) * ((p - 50) / 50))
        b = int(47 + (71 - 47) * ((p - 50) / 50))
    return f"rgb({r},{g},{b})"

# ************************************* PAGE CONFIG *************************************
st.set_page_config(page_title="QL Main Hall", page_icon="🖥️", layout="wide", initial_sidebar_state="collapsed")
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

# ************************************* MAIN APP *************************************
title_html = """ 
    <h1 style='text-align: center; font-size: 2.8em; font-weight: 900; letter-spacing: 2px; margin-bottom: 0.3em;'>
        🚀 <span style='color:#ff9800;'>Q-Learning</span> <span style='color:#2196f3;'>Training</span> 🧠
    </h1>
"""
st.markdown(title_html, unsafe_allow_html=True)

# st.markdown(title_html, unsafe_allow_html=True)

config_cols = st.columns([1, 0.6, 1])
with config_cols[0]:
    st.markdown("<p style='font-size: 28px; font-weight: bold; margin-bottom: 10px; text-align: center;'>Env Config ⚙️</p>", unsafe_allow_html=True)
    env_config_cols = st.columns(2)
    with env_config_cols[0] as random_emojis_col:
        st.markdown("<p style='font-size: 20px; font-weight: bold; margin-bottom: 5px; text-align: center;'>Random Emojis</p>", unsafe_allow_html=True)
        RANDOM_EMOJIS = st.selectbox(" ", options=[True, False], index=1, label_visibility="collapsed")
    with env_config_cols[1] as frame_size_col:
        st.markdown("<p style='font-size: 20px; font-weight: bold; margin-bottom: 5px; text-align: center;'>Frame Size</p>", unsafe_allow_html=True)
        FRAME_SIZE = st.number_input(" ", value=50, label_visibility="collapsed")
with config_cols[1]:
    st.markdown("<p style='font-size: 28px; font-weight: bold; margin-bottom: 10px; text-align: center;'>Train Config ⚙️</p>", unsafe_allow_html=True)
    train_config_cols = st.columns(2)
    with train_config_cols[0] as episodes_col:
        st.markdown("<p style='font-size: 20px; font-weight: bold; margin-bottom: 5px; text-align: center;'>Episodes</p>", unsafe_allow_html=True)
        EPISODES = st.number_input(" ", min_value=1, max_value=1000, value=10, step=1, label_visibility="collapsed")
    with train_config_cols[1] as trace_col:
        st.markdown("<p style='font-size: 20px; font-weight: bold; margin-bottom: 5px; text-align: center;'>Trace Interval</p>", unsafe_allow_html=True)
        TRACE = st.number_input(" ", min_value=1, max_value=50, value=1, step=1, label_visibility="collapsed")
with config_cols[2]:
    st.markdown("<p style='font-size: 28px; font-weight: bold; margin-bottom: 10px; text-align: center;'>Agent Config ⚙️</p>", unsafe_allow_html=True)
    agent_params_cols = st.columns(3)
    with agent_params_cols[0] as learning_rate_col:
        st.markdown("<p style='font-size: 20px; font-weight: bold; margin-bottom: 5px; text-align: center;'>Learning rate</p>", unsafe_allow_html=True)
        ALPHA = st.number_input(" ", min_value=0.001, max_value=10.0, value=0.1, step=0.1, key="alpha", format="%.3f", label_visibility="collapsed")
    with agent_params_cols[1] as discount_factor_col:
        st.markdown("<p style='font-size: 20px; font-weight: bold; margin-bottom: 5px; text-align: center;'>Discount factor</p>", unsafe_allow_html=True)
        GAMMA = st.number_input(" ", min_value=0.01, max_value=1.0, value=0.9, step=0.01, key="gamma", format="%.2f", label_visibility="collapsed")
    with agent_params_cols[2] as epsilon_col:
        st.markdown("<p style='font-size: 20px; font-weight: bold; margin-bottom: 5px; text-align: center;'>Epsilon</p>", unsafe_allow_html=True)
        EPSILON = st.number_input(" ", min_value=0.01, max_value=1.0, value=0.9, step=0.01, key="epsilon", format="%.2f", label_visibility="collapsed")

st.session_state.env = Scratch_Game_Environment5_Streamlit(frame_size=FRAME_SIZE, scratching_area=(0, 0, 700, 350), random_emojis=RANDOM_EMOJIS)
env = st.session_state.env
st.session_state.agent = RL_Agent_51_Streamlit(num_actions=env.total_squares, alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON)
agent = st.session_state.agent
gallery_images = []

game_cols = st.columns([0.3, 0.5, 0.3])
with game_cols[0]:
    env_params_html = f"""
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
            .env-title-1 {{
                text-align: center;
                font-size: 28px;
                color: #000;
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
                margin: 14px 0 24px 0;
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
            .number-highlight {{
                color: #000;
                background: #FFD700;
                padding: 2px 8px;
                border-radius: 6px;
                font-weight: bold;
                text-shadow: 0 1px 2px rgba(0,0,0,0.3);
            }}
        </style>
        <div class="full-bg-container">
            <div class='env-title-1'>
                <img src="https://em-content.zobj.net/source/animated-noto-color-emoji/356/globe-showing-europe-africa_1f30d.gif" alt="Globe" style="height:36px;vertical-align:middle;margin-right:8px;">
                Env parameters
                <img src="https://em-content.zobj.net/source/animated-noto-color-emoji/356/globe-showing-europe-africa_1f30d.gif" alt="Globe" style="height:36px;vertical-align:middle;margin-left:8px;">
            </div>
            <div class='env-title-1' style="font-size:23px; margin-bottom:0;">
                Frame size: <span class="number-highlight">{env.FRAME_SIZE}</span>
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
            <div class='env-title-1' style="font-size:23px; margin-bottom:0;">
                Total squares: <span class="number-highlight">{env.total_squares}</span>
            </div>
        </div>
    """
    st.markdown(env_params_html, unsafe_allow_html=True)
with game_cols[1]:
    image_placeholder = st.empty()
    image_placeholder.image(env.get_window_image(), use_container_width=True)
with game_cols[2]:
    agent_params_html = f"""
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
        .env-title-2 {{
            text-align: center;
            font-size: 28px;
            color: #000;
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
            color: #000;
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
        .number-highlight {{
            color: #000;
            # background: linear-gradient(45deg, #d99118, #ffe07a);
            background: #FFD700;
            padding: 2px 8px;
            border-radius: 6px;
            font-weight: bold;
            text-shadow: 0 1px 2px rgba(0,0,0,0.3);
        }}
    </style>
    <div class="full-bg-container">
        <div class='env-title-2'>
            <img src="https://em-content.zobj.net/thumbs/120/animated-noto-color-emoji/356/robot_1f916.gif" alt="Robot" style="height:36px;vertical-align:middle;margin-right:8px;">
            Agent parameters
            <img src="https://em-content.zobj.net/thumbs/120/animated-noto-color-emoji/356/robot_1f916.gif" alt="Robot" style="height:36px;vertical-align:middle;margin-left:8px;">
        </div>
        <ul class="custom-list">
            <li>Learning rate: <span class="number-highlight">{agent.alpha}</span></li>
            <li>Discount factor: <span class="number-highlight">{agent.gamma}</span></li>
            <li>Epsilon: <span class="number-highlight">{agent.epsilon}</span></li>
            <li>Qtable: <span class="number-highlight">{env.total_squares}</span> × <span class="number-highlight">{env.total_squares}</span></li>
        </ul>
    </div>
    """
    st.markdown(agent_params_html, unsafe_allow_html=True)

st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
start_button_cols = st.columns([1, 1, 1])
with start_button_cols[1]:
    if not st.button("START TRAINING", type="primary", use_container_width=True):
        st.stop()

# train progressbar
percent_placeholder = st.empty()
progress_placeholder = st.empty()
percent_placeholder.markdown("<div style='text-align:center; font-size: 2rem; font-weight:bold; color: #d32f2f; transition: color 0.5s;'>0%</div>", unsafe_allow_html=True)
progress_placeholder.progress(0)

rewards_cols = st.columns([0.7, 0.3])
actions_cols = st.columns([0.7, 0.3])
areas_cols = st.columns([0.7, 0.3])

max_reward, min_actions, min_area_scratched = -99999, 99999, 999 # best
min_reward, max_actions, max_area_scratched = 99999, 0, 0        # worst
epsilon_history = []

# """******************************BEGINNING OF TRAINING******************************"""
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
    st.markdown(f"""
        <div style='text-align: center;'>
            <h1>🎯 Actions/Episodes 🎯 (Optimum: {len(env.good_frames_idx)})</h1>
        </div>
    """, unsafe_allow_html=True)
    actions_placeholder = st.empty()
    actions_df = pd.DataFrame(columns=['Episode', 'Actions Done', 'Min Actions', 'Max Actions'])
with areas_cols[0]:
    optimum_area = math.ceil(len(env.good_frames_idx) / len(env.squares_images) * 100)
    st.markdown(f"""
        <div style='text-align: center;'>
            <h1>🌍 Areas/Episodes 🌍 (Optimum: {optimum_area}%)</h1>

        </div>
    """, unsafe_allow_html=True)
    areas_placeholder = st.empty()
    areas_df = pd.DataFrame(columns=['Episode', 'Area Scratched', 'Min Area Scratched', 'Max Area Scratched'])

for i in range(EPISODES):

    env.env_reset()

    done = False
    episode_actions = 0
    episode_reward = 0

    agent.epsilon *= np.exp(-0.001 * i)
    epsilon_history.append(agent.epsilon)

    current_state = env.frames_mask
    current_action = env.total_squares // 2

    while not done:
        episode_actions += 1

        action_index = agent.choose_action(current_action, current_state)
        next_state, reward, done = env.env_step(action_index)
        agent.update_q_table(current_action, action_index, reward, next_state)

        episode_reward += reward
        current_state = next_state

    episode_area = (env.scratched_count / env.total_squares) * 100

    min_reward = episode_reward if episode_reward < min_reward else min_reward
    min_actions = episode_actions if episode_actions < min_actions else min_actions
    min_area_scratched = episode_area if episode_area < min_area_scratched else min_area_scratched
    max_reward = episode_reward if episode_reward > max_reward else max_reward
    max_actions = episode_actions if episode_actions > max_actions else max_actions
    max_area_scratched = episode_area if episode_area > max_area_scratched else max_area_scratched
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
            '>{percent}% 
            <img src='https://em-content.zobj.net/source/animated-noto-color-emoji/356/rocket_1f680.gif' 
                style='width: 30px; height: 30px; vertical-align: middle;'>
        </div>
    """
    percent_placeholder.markdown(percent_html, unsafe_allow_html=True)
    progress_placeholder.progress(percent)
    # ---------------SAVE IMAGE TO GALLERY----------------
    if i % TRACE == 0 or i == EPISODES-1:
        image: Image.Image = env.get_window_image()
        gallery_images.append((image, i)) 
        
    time.sleep(0.01)

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
            background: linear-gradient(135deg, #388e3c 60%, #66bb6a 100%);
            border: 6px solid #388e3c;
        }}
        .triangle-max {{
            background: linear-gradient(135deg, #d32f2f 60%, #ff8a65 100%);
            border: 6px solid #d32f2f;
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
            background: linear-gradient(135deg, #388e3c 60%, #66bb6a 100%);
            border: 6px solid #388e3c;
        }}
        .square-max {{
            background: linear-gradient(135deg, #d32f2f 60%, #ff8a65 100%);
            border: 6px solid #d32f2f;
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
col_time, col_graph = st.columns([1, 1])
with col_time:
    st.markdown(f"""
        <style>
            .time-container {{
                background: linear-gradient(135deg, #27ae60, #1e8449);
                border-radius: 15px;
                padding: 20px;
                margin: auto;
                max-width: 500px;
                height: 120px;
                display: flex;
                align-items: center;
                justify-content: center;
                position: relative;
                overflow: hidden;
                box-shadow: 0 8px 32px rgba(0,0,0,0.3);
                border: 2px solid #27ae60;
                transform: translateY(120px); /* desplaza hacia la mitad vertical del gráfico */
            }}
            .time-container::before {{
                content: '';
                position: absolute;
                top: 0;
                left: -120%;
                width: 80px;
                height: 100%;
                background: linear-gradient(120deg, rgba(255,255,255,0) 0%, rgba(255,255,255,0.7) 50%, rgba(255,255,255,0) 100%);
                filter: blur(2px);
                opacity: 0.85;
                animation: slide-shine 3s cubic-bezier(.6,0,.4,1) infinite;
                z-index: 2;
            }}
            .sparkle {{
                position: absolute;
                top: 50%;
                right: -18px;
                transform: translateY(-50%) scale(0.6);
                width: 36px;
                height: 36px;
                pointer-events: none;
                opacity: 0;
                z-index: 3;
                animation: sparkle-pop 3s cubic-bezier(.6,0,.4,1) infinite;
            }}
            .sparkle svg {{
                display: block;
                width: 100%;
                height: 100%;
            }}
            @keyframes slide-shine {{
                0%   {{ left: -120%; opacity: 0.6; }}
                80%  {{ left: 100%; opacity: 0.8; }}
                100% {{ left: 100%; opacity: 0; }}
            }}
            @keyframes sparkle-pop {{
                0%, 84% {{ opacity: 0; transform: translateY(-50%) scale(0.3) rotate(0deg); }}
                85%     {{ opacity: 1; transform: translateY(-50%) scale(1.1) rotate(5deg); }}
                88%     {{ opacity: 1; transform: translateY(-50%) scale(1.3) rotate(-8deg); }}
                91%     {{ opacity: 1; transform: translateY(-50%) scale(1) rotate(0deg); }}
                100%    {{ opacity: 0; transform: translateY(-50%) scale(0.3) rotate(0deg); }}
            }}
            .time-text {{
                font-size: 24px;
                color: #ffffff;
                text-shadow: 0 2px 4px rgba(0,0,0,0.5);
                position: relative;
                z-index: 1;
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 10px;
            }}
            .time-icon {{
                font-size: 28px;
                animation: pulse 2s infinite;
            }}
            .time-value {{
                font-weight: bold;
            }}
            @keyframes pulse {{
                0%, 100% {{ transform: scale(1); }}
                50% {{ transform: scale(1.3); }}
            }}
        </style>
        <div class="time-container">
            <div class="time-text">
                <span class="time-icon">⏱️</span> 
                Total training time: <span class="time-value">{int(minutes)} min {seconds:.2f} sec</span>
            </div>
            <span class="sparkle">
                <svg viewBox="0 0 36 36" fill="none">
                    <g filter="url(#glow)">
                        <circle cx="18" cy="18" r="8" fill="#fff176" fill-opacity="0.9"/>
                        <ellipse cx="18" cy="18" rx="3" ry="10" fill="#fffde7" fill-opacity="0.7" transform="rotate(30 18 18)"/>
                        <ellipse cx="18" cy="18" rx="3" ry="10" fill="#fffde7" fill-opacity="0.7" transform="rotate(-30 18 18)"/>
                    </g>
                    <defs>
                        <filter id="glow" x="0" y="0" width="36" height="36" filterUnits="userSpaceOnUse">
                            <feGaussianBlur stdDeviation="2.5" result="coloredBlur"/>
                            <feMerge>
                                <feMergeNode in="coloredBlur"/>
                                <feMergeNode in="SourceGraphic"/>
                            </feMerge>
                        </filter>
                    </defs>
                </svg>
            </span>
        </div>
    """, unsafe_allow_html=True)
with col_graph:
    epsilon_df = pd.DataFrame({"Episode": range(1, len(epsilon_history) + 1), "Epsilon": epsilon_history})
    epsilon_chart = (
        alt.Chart(epsilon_df.assign(Serie='Epsilon'))
        .mark_line(point=True)
        .encode(
            x=alt.X('Episode:Q', title='Episode',
                    axis=alt.Axis(tickMinStep=1, labelAngle=0, labelFontSize=18, titleFontSize=22, grid=False)),
            y=alt.Y('Epsilon:Q', title='Epsilon',
                    axis=alt.Axis(labelFontSize=18, titleFontSize=22, grid=False)),
            color=alt.Color(
                'Serie:N',
                legend=alt.Legend(title=''),
                scale=alt.Scale(domain=['Epsilon'], range=['#ff0080'])
            )
        )
        .properties(width=900, height=400, padding={"top": 20})
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
        .interactive()
    )
    st.altair_chart(epsilon_chart, use_container_width=False)

# ************************************* SECTION SEPARATOR *************************************
separator_html = """
    <style>
        .smooth-separator {
            width: 100%;
            height: 4px;
            background: linear-gradient(90deg, transparent, #f44611, #ff9800, #f44611, transparent);
            border: none;
            border-radius: 2px;
            margin: 40px 0;
            box-shadow: 0 2px 8px rgba(244, 70, 17, 0.3);
            animation: separator-glow 3s ease-in-out infinite alternate;
        }
        @keyframes separator-glow {
            0% { 
                box-shadow: 0 2px 8px rgba(244, 70, 17, 0.3);
                opacity: 0.8;
            }
            100% { 
                box-shadow: 0 4px 16px rgba(244, 70, 17, 0.6);
                opacity: 1;
            }
        }
    </style>
    <div class="smooth-separator"></div>
"""
st.markdown(separator_html, unsafe_allow_html=True)

# ************************************* EPSIODE GALLERY *************************************
gallery_title_cols = st.columns([1, 2, 1], border=False)
with gallery_title_cols[1]:
    rainbow_html = """
    <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@700&display=swap" rel="stylesheet">
    <style>
    .rainbow-text {
        font-size: 2.7rem;
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

if not gallery_images:
    st.info("No images in caché. Train a model before")
else:
    cols = st.columns(3) # 3 column rows
    for i, (img, episode) in enumerate(gallery_images):
        with cols[i % 3]:
            st.image(img, caption=f"Episode {episode}", use_container_width=True)
        if (i + 1) % 3 == 0 and (i + 1) < len(gallery_images):
            cols = st.columns(3)

zip_buffer = io.BytesIO()
with zipfile.ZipFile(zip_buffer, "w") as zip_file:
    for img, episode in gallery_images:
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes.seek(0)
        zip_file.writestr(f"QL_episode_{episode}.png", img_bytes.read())
zip_buffer.seek(0)

st.markdown("""
    <style>
        div.stDownloadButton > button {
            background-color: #0099ff;
            color: white;
            font-size: 18px;
            border-radius: 10px;
            padding: 10px 24px;
            width: 100%;
            max-width: 300px;
            margin: auto;
            display: block;
        }
    </style>
""", unsafe_allow_html=True)

@st.fragment
def download_zip_fragment(zip_buffer: io.BytesIO) -> None:
    zip_cols = st.columns([1, 2, 1])
    with zip_cols[1]:
        st.download_button(label="Download episode gallery as .ZIP", data=zip_buffer, file_name=f"QL_{EPISODES}_ep_{env.total_squares}.zip")
download_zip_fragment(zip_buffer)

# ************************************* AUTHOR CREDITS *************************************
author_html = """
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

