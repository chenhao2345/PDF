# PDF: Prompt-guided Decoupled Feature Learning for Cloth-Changing Person Re-identification

Official PyTorch implementation of the paper **"PDF: Prompt-guided Decoupled Feature Learning for Cloth-Changing Person Re-identification"**.

## 1. Installation

### Requirements
*   Python 3.8
*   PyTorch 1.8.0
*   torchvision 0.9.0
*   CUDA 10.2 or higher

### Environment Setup
```bash
# Create a conda environment
conda create -n pdf_reid python=3.8
conda activate pdf_reid

# Install PyTorch and dependencies
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch
pip install yacs timm scikit-image tqdm ftfy regex matplotlib
```

## 2. Prepare Datasets

Download the cloth-changing ReID datasets into 'data' folder. The directory structure should look like this:

```text
PDF/
└── data/
    ├── LTCC_ReID/
    │   ├── train/
    │   ├── test/
    │   └── query/
    ├── prcc/
    │   ├── rgb/
    │       ├── val/
    │       ├── test/
    │       └── train/
    ├── last/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── VC-Clothes/
        ├── train/
        ├── query/
        └── gallery/
```

> **Note:** Please ensure the root path in your configuration files (`configs/*.yml`) points to your `data/` folder.

## 3. Training

We provide a convenient shell script to start the training process. This script handles the prompt-guided learning and feature decoupling.

To train the model on the specified dataset (e.g., LTCC), run:

```bash
# Run training
sh train.sh
```

*You can modify `train.sh` to switch between different datasets or hyper-parameters by changing the `--config_file` argument.*

## 4. Evaluation

To evaluate a trained model, use the `test.sh` script. Ensure you have the checkpoint file path correctly set in the script or config file.

```bash
# Run evaluation
sh test.sh
```

For manual testing of a specific weight:
```bash
CUDA_VISIBLE_DEVICES=0 python test.py --config_file configs/ltcc.yml TEST.WEIGHT 'logs/ltcc/ViT-B-16_60.pth'
```


## Acknowledgement
Our code is built upon [CLIP-ReID](https://github.com/Syliz517/CLIP-ReID). We thank the authors for their great work.

---
