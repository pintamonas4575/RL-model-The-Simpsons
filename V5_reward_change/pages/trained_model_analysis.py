import pandas as pd
import altair as alt
import streamlit as st

# ************************************* PAGE CONFIG *************************************
st.set_page_config(page_title="DQN trained data", page_icon="🤖", layout="wide", initial_sidebar_state="collapsed")
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
st.sidebar.page_link("pages/trained_model_analysis.py", icon="📊", label="Analyze trained model")
st.sidebar.page_link("pages/test_DQN.py", icon="🤖", label="Test a DQN model")

# ********************************** CSV FILE UPLOAD **********************************
title_html = """
    <style>
        .modern-frame {
            border: 2px solid;
            border-image: linear-gradient(45deg, #000, #ff9800, #000, #ff9800) 1;
            border-radius: 16px;
            padding: 2rem 1.5rem;
            background: rgba(0,0,0,0.8);
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 32px rgba(0,0,0,0.4), inset 0 0 16px rgba(255,152,0,0.2);
            position: relative;
            overflow: hidden;
            margin-bottom: 2rem;
        }
        .modern-frame::after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: repeating-linear-gradient(
                45deg,
                rgba(255,152,0,0.1),
                rgba(255,152,0,0.1) 2px,
                transparent 2px,
                transparent 5px
            );
            animation: spin-aspas 8s linear infinite;
            pointer-events: none;
            z-index: 0;
        }
        @keyframes spin-aspas {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .awesome-title {
            text-align: center;
            font-size: 2.5em;
            font-weight: 900;
            letter-spacing: 0.05em;
            margin: 0;
            font-family: 'Segoe UI', 'Roboto', 'Arial', sans-serif;
            color: #fff;
            position: relative;
            z-index: 2;
        }
        .mint {
            color: #50ffb1;
            text-shadow: 0 0 8px #50ffb199;
        }
        .electric {
            color: #00b4db;
            text-shadow: 0 0 8px #00b4db99;
        }
        .chart {
            animation: float 3s ease-in-out infinite;
            display: inline-block;
            font-size: 1.2em;
        }
        @keyframes float {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-10px); }
        }
    </style>
    <div class="modern-frame">
        <h1 class="awesome-title">
            <span class="chart">📊</span>
            <span class="mint">Trained model</span>
            <span class="electric">Analysis</span>
            <span class="chart">📊</span>
        </h1>
    </div>
"""
st.markdown(title_html, unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("<div class='center-upload'></div>", unsafe_allow_html=True)  # Esto solo es visual, no funcional
    uploaded_file = st.file_uploader("📁 Drop your CSV file here or click to browse", type=['csv'])
    if uploaded_file is not None:
        file_name: str = uploaded_file.name
        if not file_name.lower().endswith('.csv'):
            st.error("File must be CSV.")
            st.stop()
        train_df = pd.read_csv(uploaded_file)
    else:
        st.stop()

rewards_cols = st.columns([0.7, 0.3])
actions_cols = st.columns([0.7, 0.3])
areas_cols = st.columns([0.7, 0.3])

with rewards_cols[0]:
    st.markdown(
        "<div style='text-align: center;'>"
        "<h1>🏆 Rewards/Episodes 🏆</h1>"
        "</div>",
        unsafe_allow_html=True
    )
    rewards_placeholder = st.empty()
with actions_cols[0]:
    st.markdown(f"""
        <div style='text-align: center;'>
            <h1>🎯 Actions/Episodes 🎯)</h1>
        </div>
    """, unsafe_allow_html=True)
    actions_placeholder = st.empty()
with areas_cols[0]:
    st.markdown(f"""
        <div style='text-align: center;'>
            <h1>🌍 Areas/Episodes 🌍</h1>

        </div>
    """, unsafe_allow_html=True)
    areas_placeholder = st.empty()

# ************************************* REWARDS CHART *************************************
rewards_chart = alt.Chart(train_df[['Episode', 'Reward', 'Min Reward', 'Max Reward']]).transform_fold(
        ['Reward', 'Min Reward', 'Max Reward'], as_=['Serie', 'Valor']
    ).mark_line(point=True).encode(
        x=alt.X('Episode:Q', title='Episode', axis=alt.Axis(labelFontSize=18, titleFontSize=22)),
        y=alt.Y('Valor:Q', title='Reward', axis=alt.Axis(labelFontSize=18, titleFontSize=22), scale=alt.Scale(zero=False)),
        color=alt.Color(
            'Serie:N',
            legend=alt.Legend(title=None),
            scale=alt.Scale(domain=['Reward', 'Min Reward', 'Max Reward'], range=["#06e7f7", "#ff0000", "#15f10e"])
        )
    ).properties(width=1200, height=400, padding={"top": 20}).configure_axis(grid=False).interactive()
rewards_placeholder.altair_chart(rewards_chart, use_container_width=False)
# ************************************* ACTIONS CHART *************************************
actions_chart = alt.Chart(train_df[['Episode', 'Actions Done', 'Min Actions', 'Max Actions']]).transform_fold(
        ['Actions Done', 'Min Actions', 'Max Actions'], as_=['Serie', 'Valor']
    ).mark_line(point=True).encode(
        x=alt.X('Episode:Q', title='Episode', axis=alt.Axis(labelFontSize=18, titleFontSize=22)),
        y=alt.Y('Valor:Q', title='Actions Done', axis=alt.Axis(labelFontSize=18, titleFontSize=22), scale=alt.Scale(zero=False)),
        color=alt.Color(
            'Serie:N',
            legend=alt.Legend(title=None),
            scale=alt.Scale(domain=['Actions Done', 'Min Actions', 'Max Actions'], range=["#06e7f7", "#15f10e", "#ff0000"])
        )
    ).properties(width=1200, height=400, padding={"top": 20}).configure_axis(grid=False).interactive()
actions_placeholder.altair_chart(actions_chart, use_container_width=False)
# ************************************* AREAS CHART *************************************
areas_chart = alt.Chart(train_df[['Episode', 'Area Scratched', 'Min Area Scratched', 'Max Area Scratched']]).transform_fold(
        ['Area Scratched', 'Min Area Scratched', 'Max Area Scratched'], as_=['Serie', 'Valor']
    ).mark_line(point=True).encode(
        x=alt.X('Episode:Q', title='Episode', axis=alt.Axis(labelFontSize=18, titleFontSize=22)),
        y=alt.Y('Valor:Q', title='Area Scratched (%)', axis=alt.Axis(labelFontSize=18, titleFontSize=22), scale=alt.Scale(zero=False)),
        color=alt.Color(
            'Serie:N',
            legend=alt.Legend(title=None),
            scale=alt.Scale(domain=['Area Scratched', 'Min Area Scratched', 'Max Area Scratched'], range=["#06e7f7", "#15f10e", "#ff0000"])
        )
    ).properties(width=1200, height=400, padding={"top": 20}).configure_axis(grid=False).interactive()
areas_placeholder.altair_chart(areas_chart, use_container_width=False)

# *********************************TRAINING STATS*********************************
with rewards_cols[1]:
    min_val = train_df['Reward'].min()
    max_val = train_df['Reward'].max()
    avg_val = train_df['Reward'].mean()
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
                <div class="stat-number">{min_val:.0f}💶</div>
                <div class="stat-label">Min</div>
            </div>
            <div class="stat-circle circle-max">
                <div class="stat-number">{max_val:.0f}💶</div>
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
    min_val = train_df['Actions Done'].min()
    max_val = train_df['Actions Done'].max()
    avg_val = train_df['Actions Done'].mean()
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
                <div class="stat-value">{min_val:.0f}</div>
                <div class="stat-label">Min</div>
            </div>
            <div class="stat-triangle triangle-max">
                <div class="stat-value">{max_val:.0f}</div>
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
    min_val = train_df['Area Scratched'].min()
    max_val = train_df['Area Scratched'].max()
    avg_val = train_df['Area Scratched'].mean()
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

# ************************************* AUTHOR CREDITS *************************************
author_html = """
    <style>
        .author-social-row {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0;
            margin: 150px 0 30px 0;
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