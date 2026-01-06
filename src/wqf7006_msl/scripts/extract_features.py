import argparse
import os
from functools import partial
from multiprocessing import Pool

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

# -------------------------
# MediaPipe utilities
# -------------------------

mp_holistic = mp.solutions.holistic


def extract_keypoints(results):
    pose = (
        np.array(
            [
                [res.x, res.y, res.z, res.visibility]
                for res in results.pose_landmarks.landmark
            ]
        ).flatten()
        if results.pose_landmarks
        else np.zeros(33 * 4)
    )

    lh = (
        np.array(
            [[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]
        ).flatten()
        if results.left_hand_landmarks
        else np.zeros(21 * 3)
    )

    rh = (
        np.array(
            [[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]
        ).flatten()
        if results.right_hand_landmarks
        else np.zeros(21 * 3)
    )

    return np.concatenate([pose, lh, rh])


def get_frame_indices(total_frames, num_frames, sampling):
    if sampling == "first":
        return list(range(min(num_frames, total_frames)))
    elif sampling == "uniform":
        return np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()
    else:
        raise ValueError(f"Unknown sampling: {sampling}")


# -------------------------
# Worker function
# -------------------------


def _process_video_first(
    args,
    video_path,
    output_dir,
    num_frames,
):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    saved = 0
    frame_idx = 0

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        while cap.isOpened() and saved < num_frames:
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)

            # HARD CONSTRAINT: require hand landmarks
            if results.left_hand_landmarks or results.right_hand_landmarks:
                keypoints = extract_keypoints(results)

                out_path = os.path.join(output_dir, f"{frame_idx:02d}.npy")
                np.save(out_path, keypoints)

                saved += 1

            frame_idx += 1

    cap.release()
    return saved


def _process_video_uniform(
    args,
    video_path,
    output_dir,
    num_frames,
):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    keypoints_list = []  # (frame_idx, keypoints)
    frame_idx = 0

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)

            # HARD CONSTRAINT: must have hand landmarks
            if results.left_hand_landmarks or results.right_hand_landmarks:
                keypoints = extract_keypoints(results)
                keypoints_list.append((frame_idx, keypoints))

            frame_idx += 1

    cap.release()

    # ---------- uniform sampling AFTER extraction ----------
    if len(keypoints_list) >= num_frames:
        idx = np.linspace(0, len(keypoints_list) - 1, num_frames).astype(int)
        keypoints_list = [keypoints_list[i] for i in idx]

    # ---------- save using ORIGINAL frame index ----------
    for frame_idx, kp in keypoints_list:
        npy_path = os.path.join(output_dir, f"{frame_idx:02d}.npy")
        np.save(npy_path, kp)

    return len(keypoints_list)


def process_video(
    args,
    video_path,
    output_dir,
    sampling,
    num_frames,
):
    if sampling == "first":
        return _process_video_first(
            args,
            video_path,
            output_dir,
            num_frames,
        )
    elif sampling == "uniform":
        return _process_video_uniform(
            args,
            video_path,
            output_dir,
            num_frames,
        )
    else:
        raise ValueError(f"Unknown sampling: {sampling}")


# -------------------------
# Main
# -------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe features from BIM videos"
    )

    parser.add_argument("--video-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--sampling",
        choices=["first", "uniform"],
        default="first",
    )
    parser.add_argument("--num-frames", type=int, default=30)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--gloss",
        nargs="+",
        help="List of glosses to process (must all exist)",
    )

    args = parser.parse_args()

    video_root = args.video_root
    output_root = os.path.join(
        args.output_root,
        f"first_{args.num_frames}"
        if args.sampling == "first"
        else f"uniform_{args.num_frames}",
    )

    # ---------- validate glosses ----------
    available_glosses = sorted(
        d for d in os.listdir(video_root) if os.path.isdir(os.path.join(video_root, d))
    )

    if args.gloss is not None:
        requested = sorted(args.gloss)
        missing = sorted(set(requested) - set(available_glosses))

        if missing:
            raise ValueError(
                f"Invalid gloss(es): {missing}\nAvailable glosses: {available_glosses}"
            )

        glosses_to_process = requested
    else:
        glosses_to_process = available_glosses

    # ---------- build tasks ----------
    tasks = []

    for gloss in glosses_to_process:
        gloss_dir = os.path.join(video_root, gloss)

        for video in os.listdir(gloss_dir):
            if not video.endswith(".mp4"):
                continue

            video_path = os.path.join(gloss_dir, video)
            out_dir = os.path.join(
                output_root,
                gloss,
                video,
            )

            tasks.append((video_path, out_dir))

    worker = partial(
        process_video,
        args,
        sampling=args.sampling,
        num_frames=args.num_frames,
    )

    with Pool(args.num_workers) as pool:
        list(
            tqdm(
                pool.starmap(worker, tasks),
                total=len(tasks),
            )
        )


if __name__ == "__main__":
    main()
