# WQF7006 Computer Vision and Image Processing - Assignment

Case Study: Malaysian Sign Language

## Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager

## Setup

Install the dev dependencies:

```bash
uv sync --dev --extra cpu
# If you have a CUDA-enabled GPU, you can install the GPU version:
# uv sync --dev --extra cu130
```

Add extra packages if needed:

```bash
uv add <package-name>
```

## Development

Use jupyter lab/notebook for development:

```bash
# Do this
.venv/Scripts/activate  # On Windows
source .venv/bin/activate  # On macOS/Linux
jupyter lab # jupyter notebook

# Or this
uv run jupyter lab  # uv run jupyter notebook
```

Or use any IDE/text editor :)

## Dataset

```bash
uv run msl-extract-feats \ 
    --video-root "data/BIM Dataset V3/video" \
    --output-root "data/BIM Dataset V3/features" \ 
    --sampling first \
    --num-frames 30 \
    --num-workers 4 \
    --gloss hi beli pukul nasi_lemak lemak kereta nasi marah anak_lelaki baik jangan apa_khabar main pinjam buat ribut pandai_2 emak_saudara jahat panas assalamualaikum lelaki bomba emak sejuk masalah beli_2 panas_2 perempuan bagaimana

uv run msl-extract-feats \ 
    --video-root "data/BIM Dataset V3/video" \
    --output-root "data/BIM Dataset V3/features" \ 
    --sampling uniform \
    --num-frames 30 \
    --num-workers 4 \
    --gloss hi beli pukul nasi_lemak lemak kereta nasi marah anak_lelaki baik jangan apa_khabar main pinjam buat ribut pandai_2 emak_saudara jahat panas assalamualaikum lelaki bomba emak sejuk masalah beli_2 panas_2 perempuan bagaimana

uv run msl-build-tensors \
    --features-root "data/BIM Dataset V3/features/first_30/" \
    --output-root "/data/BIM Dataset V3/tensors/first_30" \
    --num-frames 30 \
    --split \
    --test-size 0.1 \
    --seed 42

uv run msl-build-tensors \
    --features-root "data/BIM Dataset V3/features/uniform_30/" \
    --output-root "/data/BIM Dataset V3/tensors/uniform_30" \
    --num-frames 30 \
    --split \
    --test-size 0.1 \
    --seed 42

```
