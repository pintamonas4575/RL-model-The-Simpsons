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

if st.button("🔄 Refrescar todo"):
    st.rerun()

st.session_state.env = Scratch_Game_Environment5_Streamlit(frame_size=130, scratching_area=(0, 0, 700, 350), background_path="../utils/space.jpg")
st.session_state.agent = RL_Agent_51_Streamlit(num_actions=st.session_state.env.total_squares)
env = st.session_state.env
agent = st.session_state.agent

# st.markdown("<h1 style='white-space: nowrap;'>Entrenamiento RL Agent con Scratch Game</h1>", unsafe_allow_html=True)

# st.markdown("<h1 style='font-size:150px; text-align:center;'>🤖</h1>", unsafe_allow_html=True)
# custom_width = 150
# st.markdown(
#     f"<img src='https://raw.githubusercontent.com/pintamonas4575/pintamonas4575/main/assets/winking-face.gif' "
#     f"width='{custom_width}' height='{custom_width}' alt='Winking Face GIF'>",
#     unsafe_allow_html=True
# )

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
    st.markdown("<h3 style='text-align: center;'>Parámetros del entorno</h3>", unsafe_allow_html=True)
with game_cols[1]:
    image_placeholder = st.empty()
    image_placeholder.image(env.get_window_image(), caption=f'Imagen inicial', use_container_width=True)
with game_cols[2]:
    st.markdown("<h3 style='text-align: center;'>🤖 Parámetros del agente 🤖</h3>", unsafe_allow_html=True)
    pass

# train progressbar
percent_placeholder = st.empty()
progress_placeholder = st.empty()

rewards_cols = st.columns([0.7, 0.3])
actions_cols = st.columns([0.7, 0.3])
areas_cols = st.columns([0.7, 0.3])

EPISODES = 25
trace = 1
actions_done, min_actions_done = [], []
areas_scratched, min_areas_scratched = [], []
max_reward, min_actions, min_area_scratched = -99999, 99999, 999
path_to_save = f"V5_version/V5_1_Qtable_{env.total_squares}_{EPISODES}_app"

epsilon = 0.9

start = time.time()
# if st.button('Comenzar entrenamiento', type='primary'):
with rewards_cols[0]:
    st.markdown(
        "<div style='text-align: center;'>"
        "<h1>🏆 Rewards/Episodes 🏆</h1>"
        "</div>",
        unsafe_allow_html=True
    )
    reward_placeholder = st.empty()
    rewards_df = pd.DataFrame(columns=['Episode', 'Reward', 'Max Reward'])
with actions_cols[0]:
    st.markdown(
        "<div style='text-align: center;'>"
        "<h1>🎯 Actions/Episodes 🎯</h1>"
        "</div>",
        unsafe_allow_html=True
    )
    action_placeholder = st.empty()
    actions_df = pd.DataFrame(columns=['Episode', 'Actions Done', 'Min Actions'])
with areas_cols[0]:
    st.markdown(
        "<div style='text-align: center;'>"
        "<h1>🌍 Areas/Episodes 🌍</h1>"
        "</div>",
        unsafe_allow_html=True
    )
    areas_placeholder = st.empty()
    areas_df = pd.DataFrame(columns=['Episode', 'Area Scratched', 'Min Area Scratched'])

# """******************************BEGINNING OF TRAINING******************************"""

for i in range(EPISODES):

    env.env_reset()

    done = False
    episode_actions = 0
    episode_reward = 0

    current_state = env.frames_mask
    current_action = env.total_squares // 2

    while not done:
        episode_actions += 1

        action_index = agent.choose_action(current_action, current_state, epsilon)
        next_state, reward, done = env.env_step(action_index)
        agent.update_q_table(current_action, action_index, reward, next_state)

        # image_placeholder.image(env.get_window_image(), caption=f'Paso {episode_actions+1}', width=1000)
        # time.sleep(0.1)

        episode_reward += reward
        current_state = next_state

    episode_percentage = (env.scratched_count / env.total_squares) * 100

    max_reward = episode_reward if episode_reward > max_reward else max_reward
    min_actions = episode_actions if episode_actions < min_actions else min_actions
    min_area_scratched = episode_percentage if episode_percentage < min_area_scratched else min_area_scratched

    if i % trace == 0 or i == EPISODES-1:
        pass
        # TODO: adapt env method to optionally save the image
        # env.get_window_image(True, f"episodes/V5_1_episode_{i}.png")
    
    # ---------------data for graphics----------------
    actions_done.append(episode_actions)
    areas_scratched.append(episode_percentage)
    min_actions_done.append(min_actions)
    min_areas_scratched.append(min_area_scratched)
    # ---------------data for graphics----------------
    # ---------------REWARDS EVOLUTION----------------
    # add row to DataFrame
    rewards_df.loc[len(rewards_df)] = [i + 1, episode_reward, max_reward] 

    rewards_chart = alt.Chart(rewards_df).transform_fold(
        ['Reward', 'Max Reward'], as_=['Serie', 'Valor']
    ).mark_line(point=True).encode(
        x=alt.X('Episode:Q', title='Episode', axis=alt.Axis(tickMinStep=1, labelAngle=0, labelFontSize=18, titleFontSize=22, grid=False)),
        y=alt.Y('Valor:Q', title='Reward', axis=alt.Axis(labelFontSize=18, titleFontSize=22, grid=False)),
        color=alt.Color(
            'Serie:N',
            legend=alt.Legend(title=""),
            scale=alt.Scale(domain=['Reward', 'Max Reward'], range=["#06e7f7", "#15f10e"])
        )
    ).properties(width=1200, height=400).configure_view(strokeWidth=0).configure_axis(grid=False).interactive()
    reward_placeholder.altair_chart(rewards_chart, use_container_width=False)
    # ---------------ACTIONS EVOLUTION----------------
    actions_df.loc[len(actions_df)] = [i + 1, episode_actions, min_actions]
    actions_chart = alt.Chart(actions_df).transform_fold(
        ['Actions Done', 'Min Actions'], as_=['Serie', 'Valor']
    ).mark_line(point=True).encode(
        x=alt.X('Episode:Q', title='Episode', axis=alt.Axis(tickMinStep=1, labelAngle=0, labelFontSize=18, titleFontSize=22, grid=False)),
        y=alt.Y('Valor:Q', title='Actions Done', axis=alt.Axis(labelFontSize=18, titleFontSize=22, grid=False)),
        color=alt.Color(
            'Serie:N',
            legend=alt.Legend(title=""),
            scale=alt.Scale(domain=['Actions Done', 'Min Actions'], range=["#06e7f7", "#15f10e"])
        )
    ).properties(width=1200, height=400).configure_view(strokeWidth=0).configure_axis(grid=False).interactive()
    action_placeholder.altair_chart(actions_chart, use_container_width=False)
    # ---------------AREAS EVOLUTION----------------
    areas_df.loc[len(areas_df)] = [i + 1, episode_percentage, min_area_scratched]
    areas_chart = alt.Chart(areas_df).transform_fold(
        ['Area Scratched', 'Min Area Scratched'], as_=['Serie', 'Valor']
    ).mark_line(point=True).encode(
        x=alt.X('Episode:Q', title='Episode', axis=alt.Axis(tickMinStep=1, labelAngle=0, labelFontSize=18, titleFontSize=22, grid=False)),
        y=alt.Y('Valor:Q', title='Area Scratched (%)', axis=alt.Axis(labelFontSize=18, titleFontSize=22, grid=False)),
        color=alt.Color(
            'Serie:N',
            legend=alt.Legend(title=""),
            scale=alt.Scale(domain=['Area Scratched', 'Min Area Scratched'], range=["#06e7f7", "#15f10e"])
        )
    ).properties(width=1200, height=400).configure_view(strokeWidth=0).configure_axis(grid=False).interactive()
    areas_placeholder.altair_chart(areas_chart, use_container_width=False)
    # -------------------------------------------------------------------------------------------------------------
    percent = int(100 * (i + 1) / EPISODES) 
    color = get_gradient_color(percent)
    percent_html = f"<div style='text-align:center; font-size: 2rem; font-weight:bold; color: {color}; transition: color 0.5s;'>{percent}%</div>"
    percent_placeholder.markdown(percent_html, unsafe_allow_html=True)
    progress_placeholder.progress(percent)

    time.sleep(0.08)

# """******************************END OF TRAINING******************************"""

aux_html = "<div style='text-align:center; font-size: 2rem; font-weight:bold; color: #43a047;'>¡Finished!</div>"
percent_placeholder.markdown(aux_html, unsafe_allow_html=True)
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
            min-height: 500px; /* Puedes ajustar según el layout */
        }}
        .triangle-container {{
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        .triangle-row {{
            display: flex;
            justify-content: center;
            gap: 80px;  /* Más separación */
            margin-bottom: 30px;
        }}
        .stat-circle {{
            width: 175px; height: 175px; border-radius: 88px;
            display: flex; flex-direction: column; justify-content: center; align-items: center;
            color: white; font-size: 40px; font-weight: bold; 
            box-shadow: 2px 2px 16px #88888866;
            margin: 0 8px;
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
        .stat-label {{
            font-size: 19px; font-weight: normal; margin-top: 7px; color: #f0f0f0;
            letter-spacing: 0.5px;
        }}
        </style>
        <div class="wrapper-center">
          <div class="triangle-container">
            <div class="triangle-row">
              <div class="stat-circle circle-min">
                  {min_val}💶
                  <div class="stat-label">Min</div>
              </div>
              <div class="stat-circle circle-max">
                  {max_val}💶
                  <div class="stat-label">Max</div>
              </div>
            </div>
            <div class="stat-circle circle-avg" style="margin-top: 40px;">
                {avg_val:.0f}💶
                <div class="stat-label">Mean</div>
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
        margin-bottom: 30px;
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
        border-radius: 12px; /* Opcional: bordes ligeramente redondeados */
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
    .stat-label {{
        font-size: 19px;
        font-weight: normal;
        margin-top: 7px;
        color: #f0f0f0;
        letter-spacing: 0.5px;
    }}
    </style>
    <div class="wrapper-center">
      <div class="square-container">
        <div class="square-row">
          <div class="stat-square square-min">
              {min_val:.1f}%
              <div class="stat-label">Min</div>
          </div>
          <div class="stat-square square-max">
              {max_val:.1f}%
              <div class="stat-label">Max</div>
          </div>
        </div>
        <div class="stat-square square-avg" style="margin-top: 40px;">
            {avg_val:.1f}%
            <div class="stat-label">Mean</div>
        </div>
      </div>
    </div>
    """
    st.markdown(areas_resume_html, unsafe_allow_html=True)


minutes, seconds = divmod(time.time()-start, 60)
st.write(f"***** Total training time: {int(minutes)} minutes and {seconds:.2f} seconds *****")

"""***********************************************************"""

