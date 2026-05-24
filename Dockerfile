# ──────────────────────────────────────────────────────────────────────────────
# Video Action Recognition — GPU Jupyter Environment
# Base: PyTorch 2.5.1 + CUDA 12.4 + cuDNN 9
# ──────────────────────────────────────────────────────────────────────────────
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# System libraries required by OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Python dependencies (pinned for reproducibility)
RUN pip install --no-cache-dir \
    opencv-python-headless==4.10.0.84 \
    scikit-learn==1.5.2 \
    matplotlib==3.9.2 \
    tqdm==4.66.5 \
    jupyterlab==4.2.5

# Copy the notebook into the image
COPY ActionRecognition_GPU.ipynb .

# ── Data layout inside the container ─────────────────────────────────────────
# Mount your local ./data folder to /workspace/data via docker-compose.
# Then update the two path variables in Cell 3 of the notebook:
#
#   DATA_DIR  = Path("/workspace/data/UCF-101")
#   SPLIT_DIR = Path("/workspace/data/UCF-101-TrainTestlist")
# ─────────────────────────────────────────────────────────────────────────────
RUN mkdir -p data

EXPOSE 8888

# Launch JupyterLab (no token so the browser opens directly)
CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", "--port=8888", \
     "--no-browser", "--allow-root", \
     "--NotebookApp.token=''", "--NotebookApp.password=''"]
