import streamlit as st
from environmentV5_st import Scratch_Game_Environment5_Streamlit

# Inicialización del entorno en session_state
if 'env' not in st.session_state:
    st.session_state.env = Scratch_Game_Environment5_Streamlit(
        frame_size=40,
        scratching_area=(0, 0, 700, 350),
        background_path="../utils/space.jpg"
    )

# Mostrar imagen actual del entorno
env = st.session_state.env
st.image(env.get_window_image_and_save(), caption='Scratch & Win')

# Simular quitar un cuadrado (p.ej, frame 0) al pulsar un botón
if st.button('Simular quitar cuadrado 0'):
    env.scratch_frame(0)

# También puedes hacer lo mismo con cualquier otro índice
i = st.number_input('Índice de frame a rascar', min_value=0, max_value=env.total_squares-1, step=1)
if st.button('Rascar frame elegido'):
    env.scratch_frame(i)
