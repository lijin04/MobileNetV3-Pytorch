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
| ImageNet | MobileNetV3 (LARGE) | WORK IN PROGRESS | | | 5.15M |
| ImageNet | MobileNetV3 (SMALL) | WORK IN PROGRESS | | | 2.94M |

## Requirements

- Python 3.x
- PyTorch >= 1.0.1
- torchvision
- scipy (for Flower102 dataset)
- tqdm
- PIL

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

```bash
python main.py
```

#### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--dataset-mode` | `IMAGENET` | Dataset to use (`CIFAR10`, `CIFAR100`, `IMAGENET`, `FLOWER102`) |
| `--epochs` | 100 | Number of epochs |
| `--batch-size` | 512 | Batch size (reduce if OOM, e.g. 128 for CIFAR, 32 for Flower102) |
| `--learning-rate` | 0.1 | Initial learning rate |
| `--dropout` | 0.8 | Dropout rate |
| `--model-mode` | `LARGE` | Network size (`LARGE` or `SMALL`) |
| `--load-pretrained` | False | Load pretrained checkpoint |
| `--evaluate` | False | Evaluation mode (skip training) |
| `--multiplier` | 1.0 | Width multiplier |
| `--data` | `data/flower102` | Dataset root path |
| `--workers` | 4 | Data loading workers |
| `--print-interval` | 5 | Logging frequency (batches) |

#### Examples

```bash
# CIFAR-100 with Large model
python main.py --dataset-mode CIFAR100 --epochs 100 --batch-size 128

# CIFAR-10 with Small model
python main.py --dataset-mode CIFAR10 --model-mode SMALL --epochs 100

# Oxford 102 Flower
python main.py --dataset-mode FLOWER102 --data data/flower102 --epochs 200 --batch-size 32 --learning-rate 0.01

# Tiny ImageNet (64x64 -> 224x224, no pretrained needed)
python main.py --dataset-mode TINY_IMAGENET --epochs 100 --batch-size 128 --learning-rate 0.1 --workers 0

# ImageNet
python main.py --dataset-mode IMAGENET --data /path/to/imagenet --epochs 200 --batch-size 256
```

### Test / Evaluate

```bash
python main.py --evaluate True --dataset-mode CIFAR100
```

- Place the saved model file in the `checkpoint/` folder.
- The checkpoint is automatically named `best_model_LARGE_ckpt.t7` or `best_model_SMALL_ckpt.t7`.

### Inference on a Single Image

Classify a single image using a trained model:

```bash
python inference.py --image path/to/image.jpg --model-mode LARGE --dataset-mode CIFAR100
```

#### Inference Options

| Option | Default | Description |
|--------|---------|-------------|
| `--image` | **(required)** | Path to input image |
| `--model-mode` | `LARGE` | Network size (`LARGE` or `SMALL`) |
| `--dataset-mode` | `CIFAR100` | Dataset mode (`CIFAR10`, `CIFAR100`, `FLOWER102`) |
| `--topk` | 5 | Show top-K predictions |
| `--multiplier` | 1.0 | Width multiplier |
| `--checkpoint` | auto | Path to checkpoint file |

#### Inference Examples

```bash
# Classify with CIFAR-100 Large model
python inference.py --image test.jpg --model-mode LARGE

# Classify with Flower102 Small model
python inference.py --image flower.jpg --model-mode SMALL --dataset-mode FLOWER102
```

Example output:
```
使用设备: cpu
图像尺寸: (400, 300)
加载模型: ./checkpoint/best_model_LARGE_ckpt.t7

预测结果 (Top-5):
==================================================
  1. cat                   (3)   置信度: 85.32%
  2. dog                   (5)   置信度: 6.15%
  3. fox                   (34)  置信度: 3.21%
==================================================
预测类别: cat
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

## Project Structure

```
MobileNetV3-Pytorch-master/
├── main.py                    # Training & evaluation script
├── model.py                   # MobileNetV3 model definition
├── preprocess.py              # Data loading & preprocessing
├── inference.py               # Single image inference script
├── organize_flower102.py      # Flower102 dataset organizer
├── organize_tiny_imagenet.py  # Tiny ImageNet dataset organizer
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
