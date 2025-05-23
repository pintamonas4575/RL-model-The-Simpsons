import streamlit as st
import time
from environmentV5_app import Scratch_Game_Environment5_Streamlit

# Inicialización del entorno en session_state
if 'env' not in st.session_state:
    st.session_state.env = Scratch_Game_Environment5_Streamlit(
        frame_size=40,
        scratching_area=(0, 0, 700, 350),
        background_path="../utils/space.jpg"
    )
env = st.session_state.env

image_placeholder = st.empty()
image_placeholder.image(env.get_window_image(), caption='Imagen inicial')
for i in range(50):
    env.scratch_frame(i)
    image_placeholder.image(env.get_window_image(), caption=f'Después de quitar el cuadrado {i}')
    time.sleep(0.3)

time.sleep(0.5)
env.env_reset()
image_placeholder.image(env.get_window_image(), caption='Imagen después de reiniciar el entorno')
