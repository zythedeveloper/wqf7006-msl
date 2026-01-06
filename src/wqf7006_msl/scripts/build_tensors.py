import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def build_xy(features_root: Path, num_frames: int):
    """
    Build X and y from extracted feature folders.

    Expected structure:
    features_root/
        gloss/
            video.mp4/
                00.npy
                01.npy
                ...

    Returns
    -------
    X : np.ndarray, shape (N, T, D)
    y : np.ndarray, shape (N,)
    label_map : dict[str, int]
    """

    glosses = sorted(d.name for d in features_root.iterdir() if d.is_dir())
    if not glosses:
        raise RuntimeError(f"No gloss folders found in {features_root}")

    label_map = {g: i for i, g in enumerate(glosses)}

    X, y = [], []

    print(f"Found {len(glosses)} glosses")
    print("Building dataset...")

    for gloss in tqdm(glosses, desc="Glosses"):
        gloss_dir = features_root / gloss

        for video_dir in sorted(gloss_dir.iterdir()):
            if not video_dir.is_dir():
                continue

            npy_files = sorted(video_dir.glob("*.npy"))
            if len(npy_files) == 0:
                continue

            # Load frame features
            frames = [np.load(f) for f in npy_files]
            D = frames[0].shape[0]

            # Pad or trim to fixed length
            if len(frames) < num_frames:
                pad = np.zeros((num_frames - len(frames), D))
                frames = np.vstack([frames, pad])
            else:
                frames = np.stack(frames[:num_frames])

            X.append(frames)
            y.append(label_map[gloss])

    X = np.stack(X)  # (N, T, D)
    y = np.array(y)  # (N,)

    print("\nDataset built:")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    return X, y, label_map


def save_base_tensors(output_root: Path, X, y, label_map):
    output_root.mkdir(parents=True, exist_ok=True)

    np.save(output_root / "X.npy", X)
    np.save(output_root / "y.npy", y)

    with open(output_root / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    print(f"\nSaved base tensors to {output_root}")


def stratified_split_and_save(
    output_root: Path,
    X,
    y,
    test_size: float,
    seed: int,
):

    print("\nPerforming stratified train/test split...")
    print(f"Test size: {test_size}, Seed: {seed}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=seed,
    )

    np.save(output_root / "X_train.npy", X_train)
    np.save(output_root / "y_train.npy", y_train)
    np.save(output_root / "X_test.npy", X_test)
    np.save(output_root / "y_test.npy", y_test)

    print("Saved split tensors:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test : {X_test.shape}")


def main():
    parser = argparse.ArgumentParser(
        description="Build X.npy and y.npy from extracted MSL features"
    )

    parser.add_argument(
        "--features-root",
        type=Path,
        required=True,
        help="Path to features (e.g. features/first_30)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Path to output tensors (e.g. tensors/first_30)",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=30,
        help="Number of frames per video (default: 30)",
    )

    # ---- split options ----
    parser.add_argument(
        "--split",
        action="store_true",
        help="Also create stratified train/test split",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set ratio (default: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting",
    )

    args = parser.parse_args()

    # ---- build tensors ----
    X, y, label_map = build_xy(
        features_root=args.features_root,
        num_frames=args.num_frames,
    )

    save_base_tensors(
        output_root=args.output_root,
        X=X,
        y=y,
        label_map=label_map,
    )

    # ---- optional split ----
    if args.split:
        stratified_split_and_save(
            output_root=args.output_root,
            X=X,
            y=y,
            test_size=args.test_size,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
