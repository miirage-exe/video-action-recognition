# Video Action Recognition

Video understanding requires reasoning about both spatial content (what objects are present)
and temporal dynamics (how they move and interact over time). Action recognition; classifying a
short video clip into one of many activity categories; is the foundational task. Early deep
learning approaches applied 2D CNNs frame-by-frame and aggregated predictions. 3D CNNs
(C3D, I3D) process spatiotemporal volumes directly. More recently, Video Transformers
(TimeSformer, VideoMAE) treat video as a sequence of patch tokens across space and time.
ObjectivesUse the UCF-101 dataset (13,320 videos, 101 action categories, 3 train/test splits) or a subset
of Kinetics-400. Implement at least two approaches: a frame-level CNN baseline (e.g., ResNet
features averaged over frames) and a temporal model (3D CNN). Compare their top-1 accuracy
and per-class performance. Analyze which actions benefit most from temporal modeling. Upon
completing this project, we expect: (i) a brief description of the video understanding literature;
(ii) a working action recognition system with spatial-only and spatiotemporal variants; (iii)
experimental evaluations comparing approaches, with analysis of which actions require
temporal context.

---

## How to run

There are two ways to run this project: **Docker** (full training from scratch, GPU required) and **Google Colab** (inspect pre-computed results or retrain in the cloud).

### Option A - Docker (GPU training)

This is the primary training path, and is what we used for this project. It runs `train.py` inside a containerized PyTorch + CUDA environment and writes all outputs to `./outputs/` on the host.

**Prerequisites**

- Docker with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed.
- The UCF-101 dataset placed under `./data/` with the following layout:

```
data/
├── UCF-101/                   # one subfolder per class, .avi video files
└── UCF-101-TrainTestlist/     # classInd.txt, trainlist0{1,2,3}.txt, testlist0{1,2,3}.txt
```

The dataset can be downloaded from https://www.crcv.ucf.edu/data/UCF101.php.

**Steps**

```bash
# 1. Generate train.py from the notebook (strips Colab-only cells tagged docker-skip)
make train.py

# 2. Build the Docker image
docker compose build

# 3. Run training (GPU required)
docker compose up
```

`make` is timestamp-aware: it skips step 1 if the notebook has not changed since the last run.

Training produces the following files under `./outputs/`:

| File | Description |
|------|-------------|
| `frame_cnn.pth` / `c3d_model.pth` | Trained model weights |
| `frame_all_labels.npy`, `frame_all_predictions.npy` | Per-sample ground truth and predictions (Frame CNN) |
| `c3d_all_labels.npy`, `c3d_all_predictions.npy` | Per-sample ground truth and predictions (C3D) |
| `frame_per_class_acc.npy`, `c3d_per_class_acc.npy` | Per-class accuracy arrays |
| `frame_cnn_history.json`, `c3d_history.json` | Epoch-level loss/accuracy logs |
| Training curve PNGs, confusion matrix PNGs | Visualizations |

---

### Option B - Google Colab (inspect results or retrain in the cloud)

Use this path to explore the pre-computed results included in this repository, or to retrain using a Colab GPU without a local setup.

**Inspecting pre-computed results (no training needed)**

Pre-computed outputs for several runs are committed to the repository:

| Folder | Description |
|--------|-------------|
| `outputs_10_epochs_without_augmented/` | 10-epoch baseline run, no augmentation |
| `outputs_10_epochs_with_augmented/` | 10-epoch run with data augmentation |
| `outputs_40_epochs_with_augmented/` | 40-epoch run with data augmentation |

To load and visualize these results in Colab without retraining, open `ActionRecognition_GPU.ipynb` and update the `OUT_DIR` variable in the **Step 4 — Configuration** cell to point to one of those folders:

```python
OUT_DIR = "outputs_10_epochs_with_augmented"
```

Then run the following cells up to **evaluation and visualization** (the confusion matrix, per-class accuracy, and training curve plots).

**Retraining from scratch on Colab**

1. Upload the repository to Google Drive.
2. Open `ActionRecognition_GPU.ipynb` in Colab (Runtime > Change runtime type > GPU).
3. Run all cells from top to bottom. The Colab-specific cells (Drive mount, data download) run automatically; the `docker-skip`-tagged cells are only excluded when using Docker.
4. Outputs are saved to the path set in `OUT_DIR` (defaults to a folder on your Drive).

---

## References

- Tran, D., Bourdev, L., Fergus, R., Torresani, L., Paluri, M. "Learning Spatiotemporal Features with 3D
Convolutional Networks." ICCV, pp. 4489–4497, 2015. (C3D)
- Tong, Z., Song, Y., Wang, J., Wang, L. "VideoMAE: Masked Autoencoders are Data-Efficient Learners
for Self-Supervised Video Pre-Training." NeurIPS, 2022.
- Dataset: https://www.crcv.ucf.edu/data/UCF101.php
