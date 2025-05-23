import streamlit as st
from environmentV5_st import Scratch_Game_Environment5_Streamlit
import time

# Inicialización del entorno en session_state
if 'env' not in st.session_state:
    st.session_state.env = Scratch_Game_Environment5_Streamlit(
        frame_size=40,
        scratching_area=(0, 0, 700, 350),
        background_path="../utils/space.jpg"
    )
env = st.session_state.env

image_placeholder = st.empty()
image_placeholder.image(env.get_window_image_and_save(), caption='Imagen inicial')
for i in range(3):
    time.sleep(2)
    env.scratch_frame(i)
    image_placeholder.image(env.get_window_image_and_save(), caption=f'Después de quitar el cuadrado {i}')

time.sleep(2)
env.env_reset()
image_placeholder.image(env.get_window_image_and_save(), caption='Imagen después de reiniciar el entorno')
