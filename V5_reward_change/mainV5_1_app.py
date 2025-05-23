import time
import io
import base64
import streamlit as st
from PIL import Image
from environmentV5_app import Scratch_Game_Environment5_Streamlit
from agentV5_1_Qtable_app import RL_Agent_51_Streamlit
from utils.functionalities import plot_results

st.set_page_config(page_title="RL_Scratch_game", page_icon="💵", layout="wide")

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


cols = st.columns(3)  # columnas laterales para centrar la central
with cols[1]:
    image_placeholder = st.empty()
    image_placeholder.image(env.get_window_image(), caption=f'Imagen inicial', width=700, use_container_width=True)

EPISODES = 2
trace = 1
rewards, max_rewards = [], []
actions_done, min_actions_done = [], []
areas_scratched, min_areas_scratched = [], []
max_reward, min_actions, min_area_scratched = -99999, 99999, 999
path_to_save = f"V5_version/V5_1_Qtable_{env.total_squares}_{EPISODES}_app"

epsilon = 0.9

start = time.time()
for i in range(EPISODES):

    env.env_reset()

    done = False
    episode_actions = 0
    episode_reward = 0

    current_state = env.frames_mask
    current_action = env.total_squares // 2

    print(f"Estado inicial: {current_state}")

    # simular un episodio entero
    while not done:
        episode_actions += 1

        action_index = agent.choose_action(current_action, current_state, epsilon)
        next_state, reward, done = env.env_step(action_index)
        agent.update_q_table(current_action, action_index, reward, next_state)

        image_placeholder.image(env.get_window_image(), caption=f'Paso {episode_actions+1}')
        time.sleep(0.1)

        episode_reward += reward
        current_state = next_state

    st.success("¡Juego terminado!")

    episode_percentage = (env.scratched_count / env.total_squares) * 100

    max_reward = episode_reward if episode_reward > max_reward else max_reward
    min_actions = episode_actions if episode_actions < min_actions else min_actions
    min_area_scratched = episode_percentage if episode_percentage < min_area_scratched else min_area_scratched

    if i % trace == 0 or i == EPISODES-1:
        st.write(f"-----------------EPISODE {i}---------------------")
        st.write(f"Nº cuadrados entorno: {env.total_squares}")
        st.write(f"Acciones realizadas: {episode_actions}")
        st.write(f"Recompensa total: {episode_reward}")
        st.write(f"Porcentaje de área rascada: {episode_percentage:.2f}%")

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



st.write("*" * 50)
st.write(f"Max reward: {max_reward}")
st.write(f"Min actions done: {min_actions}")
st.write(f"Min scratched area: {min_area_scratched:.2f}%")

minutes, seconds = divmod(time.time()-start, 60)
st.write(f"***** Total training time: {int(minutes)} minutes and {seconds:.2f} seconds *****")


"""***********************************************************"""

