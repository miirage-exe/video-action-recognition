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

References
- Tran, D., Bourdev, L., Fergus, R., Torresani, L., Paluri, M. "Learning Spatiotemporal Features with 3D
Convolutional Networks." ICCV, pp. 4489–4497, 2015. (C3D)
- Tong, Z., Song, Y., Wang, J., Wang, L. "VideoMAE: Masked Autoencoders are Data-Efficient Learners
for Self-Supervised Video Pre-Training." NeurIPS, 2022.
- Dataset: https://www.crcv.ucf.edu/data/UCF101.php
