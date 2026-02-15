from directory import pages_dir, static_dir
import streamlit as st
import os, base64

st.set_page_config(
    page_title="IsyaratAI",
    layout="wide",
    initial_sidebar_state="collapsed"
)

background_gif = os.path.join(static_dir, 'animation.gif')
with open(background_gif, "rb") as f:
    encoded_gif = base64.b64encode(f.read()).decode()

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/gif;base64,{encoded_gif}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}

    .title {{
        font-size: 5em !important;
        font-family: 'Courier New', monospace !important;
        color: CanvasText !important;
    }}

    .subtitle {{
        font-size: 1.5rem !important;
        font-family: 'Courier New', monospace !important;
        color: CanvasText !important;
        margin-bottom: 2rem;
    }}

    .container {{
        position: fixed;
        top: 47.5%;
        left: 50%;
        transform: translate(-50%, -50%);
        justify-content: center;
        align-items: center;
        text-align: center;
        color: white;
        z-index: 1000;
    }}

    div.stButton > button {{
        position: fixed;
        top: 56.25%;
        left: 41.25%;
        width: 15em;
        background: linear-gradient(135deg, #c7eff0, #f5d5fd, #fdc4ec, #92ccdd);
        color: black;
        padding: 15px 30px;
        border-radius: 8px;
        border: none;
        cursor: pointer;

        transition: 
            transform 0.25s ease,
            box-shadow 0.25s ease,
            background-color 0.25s ease;
    }}

    div.stButton > button * {{
        font-size: 1em;
        font-family: 'Courier New', monospace !important;
        font-weight: bold;
    }}
    
    div.stButton > button:hover {{
        transform: translateY(-6px) scale(1.08);
        box-shadow: 
            0 12px 24px rgba(0, 0, 0, 0.35),
            0 0 20px rgba(76, 175, 80, 0.6);
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="container">
        <h1 class="title">Isyarat AI</h1>
        <p class="subtitle">Welcome to the future of inclusive communication</p>
    </div>
    """,
    unsafe_allow_html=True
)

start_btn = st.button("Let's start", key="start-btn")
if start_btn:
    st.switch_page(os.path.join(pages_dir, 'demo.py'))
