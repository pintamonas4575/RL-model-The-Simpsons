
import streamlit as st
import numpy as np
import pandas as pd
import time
import streamlit.components.v1 as components
import base64
import matplotlib.pyplot as plt
from agentV5_1_Qtable import RL_Agent_51
from environmentV5_st import ScratchGameEnvironmentHeadless

st.set_page_config(layout="wide")
np.random.seed(0)

# Inicializa entorno si no existe
if "env" not in st.session_state:
    st.session_state.env = ScratchGameEnvironmentHeadless(frame_size=50, scratching_area=(110,98,770,300))

colors = st.session_state.env.get_visual_grid()
cols = st.session_state.env.cols

html = '''
<div style="display: grid; grid-template-columns: ''' + ' '.join(['50px'] * cols) + '''; grid-gap: 0px; line-height: 0; border-collapse: collapse; margin: 0; padding: 0;">
'''

for color in colors:
    html += f'''
    <div style="
        width: 50px; height: 50px;
        background-color: {color};
        margin: 0; padding: 0;
        border: none;
    "></div>
    '''
html += '</div>'

st.markdown("### Entorno: Scratch & Win (solo visual)")
components.html(html, height=st.session_state.env.rows * 50 + 10, width=1000, scrolling=False)

"""*******************************************************"""
st.markdown("<h1 style='text-align: center;'><em>Streamlit is <span style='color: blue;'>cool</span></em> 😎</h1>", unsafe_allow_html=True)

image_path = "../utils/space.jpg"
with open(image_path, "rb") as img_file:
    base64_image = base64.b64encode(img_file.read()).decode()
    html_code = f'''
    <div style="display: flex; justify-content: center;">
        <img src="data:image/jpeg;base64,{base64_image}" width="1000" alt="Game">
    </div>
    '''
    st.markdown(html_code, unsafe_allow_html=True)

