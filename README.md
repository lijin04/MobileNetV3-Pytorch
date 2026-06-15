# Searching for MobileNetV3 using PyTorch

PyTorch implementation of [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244) paper.

**Authors:** Andrew Howard (Google Research), Mark Sandler (Google Research), Grace Chu (Google Research), Liang-Chieh Chen (Google Research), Bo Chen (Google Research), Mingxing Tan (Google Brain), Weijun Wang (Google Research), Yukun Zhu (Google Research), Ruoming Pang (Google Brain), Vijay Vasudevan (Google Brain), Quoc V. Le (Google Brain), Hartwig Adam (Google Research)

## MobileNetV3 Block

![MobileNetV3 Block](https://user-images.githubusercontent.com/22078438/57360577-6f30d000-71b5-11e9-89a6-24034a3ecdde.PNG)

## Supported Datasets

| Dataset | Classes | Training set | Test set | Status |
|---------|---------|-------------|----------|--------|
| CIFAR-10 | 10 | 50,000 | 10,000 | ✅ Supported |
| CIFAR-100 | 100 | 50,000 | 10,000 | ✅ Supported |
| ImageNet | 1,000 | ~1.2M | 50,000 | ✅ Supported |
| Oxford 102 Flower | 102 | 1,020 (official split) | 6,149 (+1,020 val) | ✅ Supported |
| **Tiny ImageNet** | **200** | **100,000** | **10,000** | **✅ Supported** |

## Experiments

### CIFAR-100 (resized to 224×224)

| Dataset | Model | Acc@1 | Acc@5 | Epochs | Parameters |
|---------|-------|-------|-------|--------|------------|
| CIFAR-100 | MobileNetV3 (LARGE) | 70.44% | 91.34% | 80 | 3.99M |
| CIFAR-100 | MobileNetV3 (SMALL) | 67.04% | 89.41% | 55 | 1.7M |
| Tiny ImageNet | MobileNetV3 (LARGE) | **61.47%** | **81.52%** | 300 | 3.99M |
| ImageNet | MobileNetV3 (LARGE) | WORK IN PROGRESS | | | 5.15M |
| ImageNet | MobileNetV3 (SMALL) | WORK IN PROGRESS | | | 2.94M |

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

| Package | Version | Usage |
|---------|---------|-------|
| `torch` | >= 1.8.0 | Core framework, model definition, training |
| `torchvision` | >= 0.9.0 | Dataset loading, image transforms |
| `Pillow` | >= 8.0.0 | Image I/O (inference) |
| `tqdm` | >= 4.60.0 | Progress bar |
| `matplotlib` | >= 3.0.0 | Loss / accuracy curve plotting |
| `scipy` | >= 1.4.0 | `.mat` file reading (Flower102 dataset) |

## Usage

### Dataset Preparation

#### CIFAR-10 / CIFAR-100

The dataset will be automatically downloaded by torchvision when you first run training. Place it at `data/cifar/`:

```
data/
└── cifar/
    ├── cifar-10-batches-py/   (auto-downloaded)
    └── cifar-100-python/      (auto-downloaded)
```

#### Oxford 102 Flower

1. Download the dataset and place images in `data/flower102/jpg/`
2. Place `imagelabels.mat` and `setid.mat` in `data/flower102/`
3. Run the organization script:

```bash
python organize_flower102.py
```

This will create the following structure:

```
data/flower102/
├── train/1/ ~ 102/    ← 1,020 images
├── val/1/ ~ 102/      ← 1,020 images
└── test/1/ ~ 102/     ← 6,149 images
```

#### Tiny ImageNet

1. Download [Tiny ImageNet](https://www.kaggle.com/c/tiny-imagenet) and place it in `data/tiny-imagenet-200/`
2. Run the organization script to fix the validation set:

```bash
python organize_tiny_imagenet.py
```

This will create:

```
data/tiny-imagenet-200/
├── train/n01443537/ ~ n13133613/  ← 100,000 images (500/class)
├── val/n01443537/ ~ n13133613/    ← 10,000 images (50/class)
└── test/images/                    ← unlabeled test images
```

#### ImageNet

Download the ImageNet dataset and organize it as:

```
data/
├── train/
│   ├── n01440764/
│   ├── n01443537/
│   └── ...
└── val/
    ├── n01440764/
    ├── n01443537/
    └── ...
```

### Train

#### Single GPU

```bash
python main.py --dataset-mode TINY_IMAGENET --epochs 100 --batch-size 256 --learning-rate 0.1 --workers 4
```

#### Multi-GPU (Distributed Data Parallel)

Use `torchrun` to launch multi-GPU training with `DistributedDataParallel`:

```bash
# 4 GPUs
torchrun --nproc_per_node=4 main.py --dataset-mode TINY_IMAGENET --epochs 100 --batch-size 256 --learning-rate 0.1 --workers 4 --distributed

# 8 GPUs
torchrun --nproc_per_node=8 main.py --dataset-mode TINY_IMAGENET --epochs 100 --batch-size 256 --learning-rate 0.1 --workers 4 --distributed
```

> **Note:** When using `--distributed`, the learning rate is automatically scaled by `world_size` (lr × num_gpus). Each GPU processes `--batch-size` samples independently.

#### Continue Training from Checkpoint

```bash
# Single GPU
python main.py --dataset-mode TINY_IMAGENET --epochs 100 --batch-size 256 --learning-rate 0.1 --load-pretrained True

# Multi-GPU
torchrun --nproc_per_node=4 main.py --dataset-mode TINY_IMAGENET --epochs 100 --batch-size 256 --learning-rate 0.1 --workers 4 --distributed --load-pretrained True
```

#### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--dataset-mode` | `TINY_IMAGENET` | Dataset to use (`CIFAR10`, `CIFAR100`, `IMAGENET`, `FLOWER102`, `TINY_IMAGENET`) |
| `--epochs` | 100 | Number of epochs |
| `--batch-size` | 256 | Batch size **per GPU** (total = batch_size × num_gpus) |
| `--learning-rate` | 0.1 | Base learning rate (auto-scaled by world_size when distributed) |
| `--dropout` | 0.8 | Dropout rate |
| `--model-mode` | `LARGE` | Network size (`LARGE` or `SMALL`) |
| `--load-pretrained` | False | Load pretrained checkpoint to continue training |
| `--evaluate` | False | Evaluation mode (skip training) |
| `--multiplier` | 1.0 | Width multiplier |
| `--data` | `data/tiny-imagenet-200` | Dataset root path |
| `--workers` | 4 | Data loading workers per GPU |
| `--print-interval` | 20 | Logging frequency (batches) |
| `--distributed` | False | Enable multi-GPU distributed training (use with `torchrun`) |

#### Examples

```bash
# CIFAR-100 with Large model
python main.py --dataset-mode CIFAR100 --epochs 100 --batch-size 128

# CIFAR-10 with Small model
python main.py --dataset-mode CIFAR10 --model-mode SMALL --epochs 100

# Oxford 102 Flower
python main.py --dataset-mode FLOWER102 --data data/flower102 --epochs 200 --batch-size 32 --learning-rate 0.01

# Tiny ImageNet (single GPU)
python main.py --dataset-mode TINY_IMAGENET --epochs 100 --batch-size 256 --learning-rate 0.1 --workers 4

# Tiny ImageNet (4 GPUs)
torchrun --nproc_per_node=4 main.py --dataset-mode TINY_IMAGENET --epochs 100 --batch-size 256 --learning-rate 0.1 --workers 4 --distributed

# ImageNet
python main.py --dataset-mode IMAGENET --data /path/to/imagenet --epochs 200 --batch-size 256
```

### Test / Evaluate

```bash
python main.py --evaluate True --dataset-mode CIFAR100
```

- The checkpoint is automatically named `best_model_{MODE}_{DATASET}_ckpt.t7`, e.g. `best_model_LARGE_TINY_IMAGENET_ckpt.t7`.

### Inference on a Single Image

Classify a single image using a trained model:

```bash
python inference.py --image path/to/image.jpg --model-mode LARGE --dataset-mode TINY_IMAGENET
```

#### Inference Options

| Option | Default | Description |
|--------|---------|-------------|
| `--image` | **(required)** | Path to input image |
| `--model-mode` | `LARGE` | Network size (`LARGE` or `SMALL`) |
| `--dataset-mode` | `CIFAR100` | Dataset mode (`CIFAR10`, `CIFAR100`, `FLOWER102`, `TINY_IMAGENET`) |
| `--topk` | 5 | Show top-K predictions |
| `--multiplier` | 1.0 | Width multiplier |
| `--checkpoint` | auto | Path to checkpoint file |

#### Inference Examples

```bash
# Classify with CIFAR-100 Large model
python inference.py --image test.jpg --model-mode LARGE

# Classify with Tiny ImageNet Large model
python inference.py --image test.JPEG --model-mode LARGE --dataset-mode TINY_IMAGENET
```

Example output:
```
使用设备: cuda
图像尺寸: (64, 64)
加载模型: ./checkpoint/best_model_LARGE_TINY_IMAGENET_ckpt.t7
模型训练精度: Acc@1=52.76, Acc@5=75.09

预测结果 (Top-5):
==================================================
  1. n02124075            ( 32)   置信度: 85.32%
  2. n04507155            ( 18)   置信度: 6.15%
  3. n02279972            ( 13)   置信度: 3.21%
  4. n07875152            ( 23)   置信度: 2.10%
  5. n07753592            ( 46)   置信度: 1.05%
==================================================
预测类别: n02124075
```

> **Note:** Tiny ImageNet class IDs are WordNet synset IDs (e.g. `n02124075` = Egyptian cat). Look up the full name in `data/tiny-imagenet-200/words.txt`.

### Training Curves

After training completes, loss and accuracy curves are automatically saved to `reporting/`:

```
reporting/
└── best_model_LARGE_TINY_IMAGENET_curves.png
```

### Count Model Parameters

```python
import torch
from model import MobileNetV3

def get_model_parameters(model):
    total_parameters = 0
    for layer in list(model.parameters()):
        layer_parameter = 1
        for l in list(layer.size()):
            layer_parameter *= l
        total_parameters += layer_parameter
    return total_parameters

tmp = torch.randn((128, 3, 224, 224))
model = MobileNetV3(model_mode="LARGE", multiplier=1.0)
print("Number of model parameters: ", get_model_parameters(model))
```

## Multi-GPU Performance

| GPUs | Batch Size (per GPU) | Total Batch | Time per Epoch | 100 Epochs |
|:----:|:-------------------:|:-----------:|:--------------:|:----------:|
| 1× RTX 3090 | 256 | 256 | ~3.5 min | ~6 hr |
| 4× RTX 3090 | 256 | 1024 | ~1 min | ~1.5 hr |
| 8× RTX 3090 | 256 | 2048 | ~30 sec | ~0.8 hr |

## Anti-Overfitting Techniques

For small datasets like Tiny ImageNet (500 images/class) and Flower102 (10 images/class), the following techniques are applied:

| Technique | Description |
|-----------|-------------|
| **RandomResizedCrop** | Random crop & resize (scale 0.6~1.0), prevents memorization |
| **ColorJitter** | Random brightness/contrast/saturation/hue shift |
| **RandomRotation** | ±15° random rotation |
| **Label Smoothing** | Softens target labels (ε=0.1), prevents overconfidence |
| **Cosine Annealing LR** | Smooth learning rate decay instead of step decay |

## Project Structure

```
MobileNetV3-Pytorch-master/
├── main.py                    # Training & evaluation (supports single & multi-GPU)
├── model.py                   # MobileNetV3 model definition (LARGE & SMALL)
├── preprocess.py              # Data loading & preprocessing
├── inference.py               # Single image inference script
├── organize_flower102.py      # Flower102 dataset organizer
├── organize_tiny_imagenet.py  # Tiny ImageNet dataset organizer
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── data/
│   ├── cifar/                 # CIFAR-10/100 dataset
│   ├── flower102/             # Oxford 102 Flower dataset
│   └── tiny-imagenet-200/     # Tiny ImageNet dataset
├── checkpoint/                # Saved model checkpoints
└── reporting/                 # Training logs & curve plots
```

## TODO

- [ ] ImageNet full training & evaluation
- [ ] Code refactoring for general-purpose usage
