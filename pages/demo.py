import os, torch, warnings
import streamlit as st
import numpy as np
from directory import weight_dir
from model import load_model
from utils import get_device, generate_video_and_subtitle

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="IsyaratAI",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title('DEMO.')

if "videos" not in st.session_state:
    st.session_state.videos = []

if "selected_name" not in st.session_state:
    st.session_state.selected_name = None

if "gestures" not in st.session_state:
    st.session_state.gestures = [
        'hi', 'beli', 'pukul', 'nasi_lemak',
        'lemak', 'kereta', 'nasi', 'marah',
        'anak_lelaki', 'baik', 'jangan', 'apa_khabar',
        'main', 'pinjam', 'buat', 'ribut',
        'pandai_2', 'emak_saudara', 'jahat', 'panas',
        'assalamualaikum', 'lelaki', 'bomba', 'emak',
        'sejuk', 'masalah', 'beli_2', 'anak_perempuan',
        'perempuan', 'panas_2'
    ]

weight_file = os.path.join(weight_dir, 'trained_model.pth')
device = get_device()
model = load_model(device, weight_file)

# user interface setup
tab1, tab2 = st.tabs(["Upload", "Camera"])

with tab1:
    videos = st.file_uploader("Feed me with your videos", accept_multiple_files=True, type=["mp4", "mov"])
    st.session_state.videos.extend(videos)

    if videos:
        st.subheader("Video Player")
        left_col, right_col = st.columns([1, 3])
        with left_col:
                video_names = [v.name for v in videos]
                selected_name = st.radio("Select a video", video_names)
                st.session_state.selected_name = selected_name

        with right_col:
            if st.session_state.selected_name:
                selected_video = next(v for v in st.session_state.videos if v.name == st.session_state.selected_name)

                # generate video with landmark and subtitle
                output_memory_file, vtt_file = generate_video_and_subtitle(selected_video, device=device, model=model)

                # load video and subtitle into media player
                st.video(output_memory_file, subtitles=vtt_file.name)
            else:
                st.info("No video selected")
    
with tab2:
    st.header("A dog")
    st.image("https://static.streamlit.io/examples/dog.jpg", width=200)
