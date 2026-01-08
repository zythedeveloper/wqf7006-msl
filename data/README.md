Get the dataset from [huggingface](https://huggingface.co/datasets/jherng/malaysian-sign-language).

```bash
wqf7006-msl/
└── data/
    └── BIM Dataset V3/
        ├── video/                          # Raw video files (organized by gloss/sign)
        │   ├── abang/
        │   │   ├── abang_1_1_1.mp4
        │   │   ├── ...
        │   │   └── abang_4_6_3.mp4
        │   ├── ada/
        │   ├── ...
        │   └── ambil/
        │
        ├── features/                       # Extracted MediaPipe features
        │   ├── first_30/                   # First 30 frames with hand landmarks
        │   │   ├── abang/
        │   │   │   ├── abang_1_1_1.npy    # Shape: (30, 258) - all frames in one file
        │   │   │   ├── abang_4_6_3.npy    # Shape: (22, 258) - might be less than 30 frames, padding is not done
        │   │   │   └── ...
        │   │   ├── ada/
        │   │   └── ...
        │   │
        │   └── uniform_30/                # Uniformly sampled 30 frames
        │       ├── abang/
        │       │   ├── abang_1_1_1.npy    # Shape: (30, 258) - all frames in one file
        │       │   ├── abang_4_6_3.npy    # Shape: (22, 258) - might be less than 30 frames, padding is not done
        │       │   └── ...
        │       └── ...
        │
        └── tensors/                        # Processed tensors for ML training
            ├── first_30/
            │   ├── X.npy                   # Full dataset features (N, T, D), T is the number of specified frames (30 by default), padding is done here
            │   ├── y.npy                   # Full dataset labels (N,)
            │   ├── X_train.npy             # Training features
            │   ├── y_train.npy             # Training labels
            │   ├── X_test.npy              # Test features
            │   ├── y_test.npy              # Test labels
            │   └── label_map.json          # Gloss to label index mapping
            │
            └── uniform_30/
                └── [same structure as first_30/]
```
