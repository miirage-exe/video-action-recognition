# ──────────────────────────────────────────────────────────────────────────────
# Video Action Recognition — GPU Training Environment
# Base: PyTorch 2.5.1 + CUDA 12.4 + cuDNN 9
# ──────────────────────────────────────────────────────────────────────────────
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# System libraries required by OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Python dependencies
RUN pip install --no-cache-dir \
    opencv-python-headless==4.10.0.84 \
    scikit-learn==1.5.2 \
    matplotlib==3.9.2 \
    tqdm==4.66.5

# Paths injected via environment variables (same variables used in the notebook)
ENV DATA_DIR=/workspace/data/UCF-101
ENV SPLIT_DIR=/workspace/data/UCF-101-TrainTestlist
ENV OUT_DIR=/workspace/outputs

# Copy the generated training script (run `make train.py` before building)
COPY train.py .

RUN mkdir -p data outputs

CMD ["python", "-u", "train.py"]
