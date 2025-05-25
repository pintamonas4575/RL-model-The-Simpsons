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
import matplotlib.pyplot as plt
from environmentV5_app import Scratch_Game_Environment5_Streamlit
from agentV5_1_Qtable_app import RL_Agent_51_Streamlit
from utils.functionalities import plot_results

st.set_page_config(page_title="RL_Scratch_Game", page_icon="💵", layout="wide")

def get_gradient_color(p):
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

"""***********************************************************"""

st.markdown("<h1 style='text-align: center;'>Entrenamiento de un Agente de Aprendizaje por Refuerzo con Scratch Game</h1>", unsafe_allow_html=True)

if st.button("🔄 Refrescar todo"):
    st.rerun()

# app bg image
bg_image_html = """
    <style>
    .stApp {
        background-image: url('https://images.unsplash.com/photo-1506744038136-46273834b3fb?auto=format&fit=crop&w=1350&q=80');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
"""
# st.markdown(bg_image_html, unsafe_allow_html=True)

st.session_state.env = Scratch_Game_Environment5_Streamlit(frame_size=50, scratching_area=(0, 0, 700, 350), background_path="../utils/space.jpg")
st.session_state.agent = RL_Agent_51_Streamlit(num_actions=st.session_state.env.total_squares)
env = st.session_state.env
agent = st.session_state.agent


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


game_cols = st.columns([0.3, 0.5, 0.3], border=True)
with game_cols[0]:
    st.markdown(f"""
        <div class="custom-col">
            <h3 style='text-align: center; color: #fff; margin-bottom: 40px;'>🧩 Env parameters 🧩</h3>
            <div class="params-list">
                <div class="param-item">
                    <span class="param-emoji">🖼️</span>
                    <span class="param-text">Frame size: <strong>{env.FRAME_SIZE}</strong></span>
                </div>
                <div class="param-item">
                    <span class="param-emoji">🔢</span>
                    <span class="param-text">Total squares: <strong>{env.total_squares}</strong></span>
                </div>
            </div>
        </div>
        <style>
        .custom-col {{
            background: #232f4b;
            border-radius: 18px;
            padding: 24px 12px;
            min-height: 350px;
            box-shadow: 0 2px 16px #0002;
            width: 100%;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }}
        .params-list {{
            display: flex;
            flex-direction: column;
            align-items: left;
            gap: 20px;
        }}
        .param-item {{
            display: flex;
            align-items: center;
            gap: 15px;
            font-size: 20px;
            color: #fff;
            margin-left: 100px;
        }}
        .param-emoji {{
            font-size: 24px;
        }}
        .param-text {{
            font-size: 18px;
        }}
        </style>
    """, unsafe_allow_html=True)

# with game_cols[0]:
#     st.markdown("<h3 style='text-align: center;'>🧩 Env parameters 🧩</h3>", unsafe_allow_html=True)
#     st.markdown(f"<p style='text-align: center;'>Total squares: <strong>{env.total_squares}</strong></p>", unsafe_allow_html=True)
with game_cols[1]:
    image_placeholder = st.empty()
    image_placeholder.image(env.get_window_image(), use_container_width=True)
with game_cols[2]:
    st.markdown("<h3 style='text-align: center;'>🤖 Agent parameters 🤖</h3>", unsafe_allow_html=True)
    pass

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

EPISODES = 20
trace = 1
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

    if i % trace == 0 or i == EPISODES-1:
        pass
        # TODO: adapt env method to optionally save the image
        # env.get_window_image(True, f"episodes/V5_1_episode_{i}.png")
    
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

"""***********************************************************"""

