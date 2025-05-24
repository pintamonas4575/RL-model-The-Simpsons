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

st.set_page_config(page_title="RL_Scratch_game", page_icon="💵", layout="wide")

def pil_to_base64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

"""***********************************************************"""

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

html = """
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
        animation: rotateColors 7s infinite linear reverse;
    }

    .rainbow span:nth-child(1) { animation-delay: 0s; }
    .rainbow span:nth-child(2) { animation-delay: -1s; }
    .rainbow span:nth-child(3) { animation-delay: -2s; }
    .rainbow span:nth-child(4) { animation-delay: -3s; }
    .rainbow span:nth-child(5) { animation-delay: -4s; }
    .rainbow span:nth-child(6) { animation-delay: -5s; }
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
# st.markdown(html, unsafe_allow_html=True)


# cols = st.columns(3, border=True)
game_cols = st.columns([0.3, 0.3, 0.3], border=True)
with game_cols[1]:
    image_placeholder = st.empty()
    image_placeholder.image(env.get_window_image(), caption=f'Imagen inicial', width=700)

chart_cols = st.columns([0.7, 0.3], border=True)

EPISODES = 25
trace = 1
rewards, max_rewards = [], []
actions_done, min_actions_done = [], []
areas_scratched, min_areas_scratched = [], []
max_reward, min_actions, min_area_scratched = -99999, 99999, 999
path_to_save = f"V5_version/V5_1_Qtable_{env.total_squares}_{EPISODES}_app"

epsilon = 0.9

start = time.time()
# if st.button('Comenzar entrenamiento', type='primary'):
if True:

    placeholder = st.empty()

    with chart_cols[0]:
        st.markdown(
            "<div style='text-align: center;'>"
            "<h1>🏆 Rewards 🏆</h1>"
            "</div>",
            unsafe_allow_html=True
        )
        data = pd.DataFrame({'Rewards': [], 'Max Reward': []}, columns=['Rewards', 'Max Reward'])
        chart = st.line_chart(data, width=350, height=300, x_label='Episodio', y_label='Recompensa')


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

        # st.success("¡Juego terminado!")

        episode_percentage = (env.scratched_count / env.total_squares) * 100

        max_reward = episode_reward if episode_reward > max_reward else max_reward
        min_actions = episode_actions if episode_actions < min_actions else min_actions
        min_area_scratched = episode_percentage if episode_percentage < min_area_scratched else min_area_scratched

        if i % trace == 0 or i == EPISODES-1:
            pass
            # st.write(f"-----------------EPISODE {i}---------------------")
            # st.write(f"Nº cuadrados entorno: {env.total_squares}")
            # st.write(f"Acciones realizadas: {episode_actions}")
            # st.write(f"Recompensa total: {episode_reward}")
            # st.write(f"Porcentaje de área rascada: {episode_percentage:.2f}%")

            # TODO: adapt env method to optionally save the image
            # env.get_window_image(True, f"episodes/V5_1_episode_{i}.png")
        
        # ---------------data for graphics----------------
        rewards.append(episode_reward)
        actions_done.append(episode_actions)
        areas_scratched.append(episode_percentage)
        max_rewards.append(max_reward)
        min_actions_done.append(min_actions)
        min_areas_scratched.append(min_area_scratched)
        # ---------------data for graphics----------------
        # ---------------NEW data for graphics----------------
        # data = pd.DataFrame({
        #     'Episode': range(1, len(rewards) + 1),
        #     'Rewards': rewards,
        #     'Max Reward': max_rewards
        # })
        # new_row = {'Episode': i+1, 'Rewards': episode_reward, 'Max Reward': max_reward}
        # chart = st.line_chart(data, x='Episode', y='Rewards', use_container_width=True)


        new_row = {'Rewards': episode_reward, 'Max Reward': max_reward}
        chart.add_rows(pd.DataFrame([new_row], index=[i+1]))
        # chart.add_rows(pd.DataFrame([new_row]))
        # ---------------NEW data for graphics----------------
        # data = pd.DataFrame({
        #     'Episodio': list(range(1, len(rewards)+1)),
        #     'Rewards': rewards,
        #     'Max Reward': max_rewards
        # })
        
        # # zoom = alt.selection_interval(bind='scales', encodings=['x'])
        # zoom = alt.selection_interval(bind='scales', encodings=['x', 'y'])
        # chart = alt.Chart(data).transform_fold(
        #     ['Rewards', 'Max Reward'], as_=['Serie', 'Valor']
        # ).mark_line(point=True).encode(
        #     x=alt.X('Episodio:O', title='Episode', axis=alt.Axis(tickMinStep=1, labelAngle=0)),
        #     y=alt.Y('Valor:Q', title='Reward'),
        #     color=alt.Color(
        #         'Serie:N',
        #         legend=alt.Legend(title='Legend'),
        #         scale=alt.Scale(domain=['Rewards', 'Max Reward'], range=['#21c05e', '#ff9800'])
        #     )
        # ).properties(width=350, height=300).add_params(zoom)

        # placeholder.altair_chart(chart, use_container_width=True)
        # ---------------NEW data for graphics----------------
        # data = pd.DataFrame({
        # 'Episode': list(range(1, len(rewards)+1)),
        # 'Rewards': rewards,
        # 'Max Reward': max_rewards
        # })

        # fig = go.Figure()
        # fig.add_trace(go.Scatter(x=data['Episode'], y=data['Rewards'], mode='lines+markers', name='Rewards'))
        # fig.add_trace(go.Scatter(x=data['Episode'], y=data['Max Reward'], mode='lines+markers', name='Max Reward'))
        # fig.update_layout(
        #     xaxis_title='Episodio',
        #     yaxis_title='Recompensa',
        #     legend_title='Serie',
        #     width=700,
        #     height=350
        # )
        # st.plotly_chart(fig, use_container_width=True)
        # ---------------NEW data for graphics----------------


        # time.sleep(1)


# image_placeholder.image(env.get_window_image(), caption=f'Final de entrenamiento', width=700)


# st.write("*" * 50)
# st.write(f"Max reward: {max_reward}")
# st.write(f"Min actions done: {min_actions}")
# st.write(f"Min scratched area: {min_area_scratched:.2f}%")

minutes, seconds = divmod(time.time()-start, 60)
st.write(f"***** Total training time: {int(minutes)} minutes and {seconds:.2f} seconds *****")

"""***********************************************************"""

