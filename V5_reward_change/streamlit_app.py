
import streamlit as st
import numpy as np
import pandas as pd
from datetime import time


st.set_page_config(layout="wide")
np.random.seed(0)

st.title("Streamlit App")
left_column, right_column = st.columns(2)
left_column.button('Press me!')
# Or even better, call Streamlit functions inside a "with" block:
with right_column:
    chosen = st.radio(
        'Sorting hat',
        ("Gryffindor", "Ravenclaw", "Hufflepuff", "Slytherin"))
    st.write(f"You are in {chosen} house!")


st.markdown("""
<style>
  .big-font {
    font-size:30px !important;
    color: red;
    text-align: left;
  }
  .custom-box {
    border: 2px solid #4CAF50;
    padding: 15px;
    border-radius: 10px;
  }
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="big-font custom-box">¡Texto personalizado!</div>', unsafe_allow_html=True)

# Add a selectbox to the sidebar:
add_selectbox = st.sidebar.selectbox(
    'How would you like to be contacted?',
    ('Email', 'Home phone', 'Mobile phone')
)

# Add a slider to the sidebar:
add_slider = st.sidebar.slider(
    'Select a range of values',
    0.0, 100.0, (25.0, 75.0)
)

st.title("Tests de Streamlit")
# st.html()
st.markdown(""":+1:
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
""", unsafe_allow_html=True)

# Divide la pantalla en dos columnas de diferentes anchos
col1, col2 = st.columns([3, 2])

with col1:
    st.header("Dataframe")
    dataframe = pd.DataFrame(
        np.random.randn(10, 20),
        columns=('col %d' % i for i in range(20)))
    st.dataframe(dataframe.style.highlight_max(axis=0))

with col2:
    st.header("Line Chart")
    chart_data = pd.DataFrame(
        np.random.randn(20, 3),
        columns=['a', 'b', 'c'])
    st.line_chart(chart_data)

# Otro conjunto de columnas para sliders
col3, col4 = st.columns(2)

with col3:
    x = st.slider('x', min_value=5, max_value=10)
    st.write(x, 'squared is', x * x)

with col4:
    appointment = st.slider(
        "Schedule your appointment:", value=(time(11, 30), time(12, 45))
    )
    st.write("You're scheduled for:", appointment)

