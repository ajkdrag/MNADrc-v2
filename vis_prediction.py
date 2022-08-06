import os
import time
import cv2
import streamlit as st
import matplotlib.pyplot as plt
from stqdm import stqdm
from pathlib import Path
from inference import *


class Config:
    model_file = "../artifacts/MNADrc/Ped2_prediction_model.pth"
    model_keys_file = "../artifacts/MNADrc/Ped2_prediction_keys.pt"
    data_dir = "../data/ped2/testing/st_frames"
    vid_dir = "../data/ped2/testing/st_videos"
    gt_file = "../data/ped2/ped2.mat"
    dataset_type = "ped2"

    gpus = "0"
    h = 256
    w = 256
    t_length = 5
    th = 0.01
    loc_thresh = 230
    alpha = 0.6
    batch_size = 1
    num_workers_test = 1


def st_display_video(vid_name):
    with open(vid_name,'rb') as stream:
        video_bytes = stream.read()
        st.video(video_bytes)


def vid_to_frames(vid_name, out_dir):
    vid_name_wo_ext = Path(vid_name).stem
    frames_dir = Path(out_dir) / vid_name_wo_ext
    frames_dir.mkdir(exist_ok=True, parents=True)

    cap = cv2.VideoCapture(vid_name)
    idx = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # frame = cv2.resize(frame, (Config.h, Config.w))
        frame_file = frames_dir / f"{idx:03d}.jpg"
        cv2.imwrite(str(frame_file), frame)
        idx += 1
    
    cap.release()


def upload_video(key, split_frames=True):
    session_key = f'{key}_video_path'
    if session_key not in st.session_state:
        st.session_state[session_key] = None

    uploaded_file = st.file_uploader('Choose a file', key=key)

    session_data = st.session_state[session_key]
    if session_data != None and Path(session_data).exists():
        agree = st.checkbox(
            'Previous file found! Do you want to use previous video file?')
        if agree:
            st_display_video(session_data)
            return Path(session_data).stem

    if uploaded_file is not None:
        base_path = Path(Config.vid_dir)
        base_path.mkdir(parents=True, exist_ok=True)
        video_file = base_path / uploaded_file.name
        session_data = str(video_file)

        with video_file.open(mode="wb") as f:
            f.write(uploaded_file.read())

        st.session_state[session_key] = session_data
        st.write(f'\n\nUploaded video file: {session_data}')

        st_display_video(session_data)

        if split_frames:
            vid_to_frames(session_data, Config.data_dir)
        
        return Path(session_data).stem


def run_inference():
    with st.spinner("Loading model"):
        torch_setup(Config)
        model, m_items = get_model(Config)
        time.sleep(1)
    
    st.info("Loaded Model")

    with st.spinner("Loading data"):
        dataset_batch = get_dataset_batch(Config)
        gt = get_gt(Config)
        videos_dict, videos_list = get_videos(Config)
        ds = {}
        init_datastructures(Config, videos_dict, videos_list, gt, ds)
        time.sleep(1)

    st.info("Loaded Data")


    st.markdown(
        """
        <style>
            .stProgress > div > div > div > div {
                background-color: green;
            }
        </style>""",
        unsafe_allow_html=True,
    )
    with stqdm(total=len(dataset_batch), desc="Running inference", mininterval=1) as pbar:
        vis_file = evaluate(Config, model, 
                      m_items, dataset_batch, 
                      videos_dict, videos_list, 
                      ds, save_diff=True, tqdm_pbar=pbar)
    
    st.info("Finished Inference")

    with st.spinner("Plotting results"):
        if vis_file:
            st_display_video(vis_file)

        anomaly_score_total_list = get_anomaly_scores(Config, 
                                    videos_list, ds, save=False, plot=False)
        print(anomaly_score_total_list)
        fig = plt.figure()
        plt.plot(anomaly_score_total_list)
        st.pyplot(fig)
        
        if not Config.anomalous_data:
            st.write("AUC is not defined in this case.")
            AUC_val = "NA"
        else:
            try:
                AUC_val = calc_AUC(anomaly_score_total_list, ds)
            except ValueError as err:
                st.write(err)
                AUC_val = "NA"

    st.info(f"AUC: {AUC_val}")


def run():
    print("In run ...")

    video_widget_key = "anomalous_vid" if Config.anomalous_data else "normal_vid"
    
    # save video (and frames) to disk
    video_file_name = upload_video(video_widget_key)
    Config.desired_folder = video_file_name

    placeholder = st.empty()

    if video_file_name:
        btn = placeholder.button("Detect Anomaly", disabled=False, key="1")
        if btn:
            placeholder.button('Detect Anomaly', disabled=True, key='2')
            run_inference()
            placeholder.button('Detect Anomaly', disabled=False, key='3')
            placeholder.empty()


def ui():
    new_title = '<p style="font-size: 42px;">Anomaly Detection in Pedestrian crossings</p>'
    read_me_0 = st.markdown(new_title, unsafe_allow_html=True)
    read_me = st.markdown("""
    This project was built using Streamlit and OpenCV 
    to demonstrate anomaly detection in a pedestrian dataset (UCSDPed2)."""
    )
    st.sidebar.title("Select Activity")
    choices = ("About","AD (Ped2 Normal)","AD (Ped2 Anomaly)")
    choice  = st.sidebar.selectbox("MODE", choices)
    
    
    if choice == choices[1]:
        read_me_0.empty()
        read_me.empty()
        Config.anomalous_data = False
        run()
    elif choice == choices[2]:
        read_me_0.empty()
        read_me.empty()
        Config.anomalous_data = True
        run()
    elif choice == "About":
        print("In About section ...")


if __name__=="__main__":
    print("Starting UI ...")
    ui()

