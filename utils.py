import mediapipe as mp
import numpy as np
import streamlit as st
import torch, cv2, tempfile, io, av

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def mediapipe_detection(image,model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Color conversion from BGR to RGB
    image.flags.writeable = False                   # Image is no longer writeable
    results = model.process(image)                  # Make prediction
    image.flags.writeable = True                    # Image is no longer writeable
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)   # Color conversion RGB to BGR
    return image, results


def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)  # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)    # Draw right connections


def draw_styled_landmarks(image,results):
    # Draw pose connection
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(0,0,255), thickness=1,circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80,110,10), thickness=1,circle_radius=1)
                              )
    # Draw left hand connection
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(0,0,255), thickness=1,circle_radius=2),
                              mp_drawing.DrawingSpec(color=(80,110,10), thickness=1,circle_radius=1)
                              )
    # Draw right hand connection
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(0,0,255), thickness=1,circle_radius=2),
                              mp_drawing.DrawingSpec(color=(80,110,10), thickness=1,circle_radius=1)
                              )


def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)

    return np.concatenate([pose, lh, rh])


# device configuration
@st.cache_resource
def get_device():
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print("device:", device)
    return device


# generate a subtitle file
def create_vtt(predictions, vtt_path):
    def format_timestamp(seconds):
        hrs = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hrs:01}:{mins:02}:{secs:02}.{millis:03}"

    with open(vtt_path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for start, end, label in predictions:
            f.write(f"{format_timestamp(start)} --> {format_timestamp(end)}\n")
            f.write(f"{label}\n\n")



# generate video with landmark and subtitle
def generate_video_and_subtitle(selected_video, device, model):
    sequence = []

    # temporary file for OpenCV to read the upload
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(selected_video.read())
    
    # open video with OpenCV
    cap = cv2.VideoCapture(tfile.name)
    tfile.close()

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2) * 2
    height = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) // 2) * 2
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # print(f"Total frames: {frame_count}")

    output_memory_file = io.BytesIO()
    output_container = av.open(output_memory_file, 'w', format="mp4")
    stream = output_container.add_stream('h264', rate=int(fps))
    stream.width = width
    stream.height = height
    stream.pix_fmt = 'yuv420p'

    sequence, predictions = [], []
    frame_count, start_frame = 0, 0
    current_gesture = None

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        image, results = mediapipe_detection(frame, holistic)

        # write the frame with landmarks to the output video
        draw_styled_landmarks(image, results)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        av_frame = av.VideoFrame.from_ndarray(image, format='rgb24')

        # encode and mux
        for packet in stream.encode(av_frame):
            output_container.mux(packet)

        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]
        frame_count += 1
        
        # perform prediction every 30 frames
        if len(sequence) == 30:
            input_data = torch.tensor(np.expand_dims(sequence, axis=0), dtype=torch.float32).to(device)
            
            with torch.no_grad():
                res = model(input_data)
                probabilities = torch.softmax(res, dim=1)
                max_val, max_idx = torch.max(probabilities, dim=1)
                predicted_label = st.session_state.gestures[max_idx.item()]
                confidence = float(max_val.item())

                # print(f"frames: {len(sequence)}, Prediction: {predicted_label} ({confidence:.2f})")

                # threshold logic
                detected_label = predicted_label if confidence > 0.9 else None

                if detected_label == current_gesture:
                    pass
                else:
                    if current_gesture is not None:
                        end_time = (frame_count - 1) / fps
                        start_time = start_frame / fps
                        predictions.append((start_time, end_time, current_gesture))
                    
                    current_gesture = detected_label
                    start_frame = frame_count

    # handle the last gesture after loop ends
    # if current_gesture is not None:
    #     predictions.append((start_frame/fps, frame_count/fps, current_gesture))

    # flush the encoder
    for packet in stream.encode(None):
        output_container.mux(packet)
    output_container.close()
    cap.release()

    # reset pointer to start of the file
    output_memory_file.seek(0)

    # create the subtitle file
    vtt_file = tempfile.NamedTemporaryFile(delete=False, suffix=".vtt")
    create_vtt(predictions, vtt_file.name)

    return output_memory_file, vtt_file