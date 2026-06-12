import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import argparse
import sys
import os

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import MobileNetV3


def load_class_names(dataset_mode):
    """加载类别名称"""
    if dataset_mode == "CIFAR10":
        return ['airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck']
    elif dataset_mode == "FLOWER102":
        return [str(i) for i in range(1, 103)]
    elif dataset_mode == "TINY_IMAGENET":
        # 从 wnids.txt 读取类别ID，并按字母排序以匹配 ImageFolder 顺序
        wnids_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'data', 'tiny-imagenet-200', 'wnids.txt')
        if os.path.exists(wnids_path):
            with open(wnids_path, 'r') as f:
                classes = sorted([line.strip() for line in f.readlines()])
                return classes
        return [str(i) for i in range(200)]
    else:  # CIFAR100
        return ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
                'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
                'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
                'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
                'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
                'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
                'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
                'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
                'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
                'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
                'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
                'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank',
                'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip',
                'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']


def get_num_classes(dataset_mode):
    if dataset_mode == "CIFAR10":
        return 10
    elif dataset_mode == "CIFAR100":
        return 100
    elif dataset_mode == "FLOWER102":
        return 102
    elif dataset_mode == "TINY_IMAGENET":
        return 200
    return 1000


def get_normalize_params(dataset_mode):
    if dataset_mode in ["CIFAR10", "CIFAR100"]:
        if dataset_mode == "CIFAR10":
            return (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        else:
            return (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
    else:
        return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def get_args():
    parser = argparse.ArgumentParser("MobileNetV3 图像分类推理")
    parser.add_argument("--image", type=str, required=True, help="输入图像路径")
    parser.add_argument("--dataset-mode", type=str, default="CIFAR100", choices=["CIFAR10", "CIFAR100", "FLOWER102", "TINY_IMAGENET"], help="数据集模式")
    parser.add_argument("--model-mode", type=str, default="LARGE", choices=["LARGE", "SMALL"], help="模型大小")
    parser.add_argument("--multiplier", type=float, default=1.0, help="模型宽度倍率")
    parser.add_argument("--topk", type=int, default=5, help="显示前K个预测结果")
    parser.add_argument("--checkpoint", type=str, default=None, help="指定checkpoint路径，默认自动选择")
    return parser.parse_args()


def main():
    args = get_args()

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 类别名称
    classes = load_class_names(args.dataset_mode)
    num_classes = get_num_classes(args.dataset_mode)
    mean, std = get_normalize_params(args.dataset_mode)

    # 图像预处理（与训练时一致）
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # 加载图像
    try:
        image = Image.open(args.image).convert("RGB")
    except Exception as e:
        print(f"无法打开图像: {e}")
        return

    print(f"图像尺寸: {image.size}")
    input_tensor = transform(image).unsqueeze(0).to(device)  # [1, 3, 224, 224]

    # 加载模型
    model = MobileNetV3(model_mode=args.model_mode, num_classes=num_classes,
                        multiplier=args.multiplier, dropout_rate=0.0).to(device)
    model = nn.DataParallel(model).to(device)

    # 确定checkpoint路径
    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        ckpt_path = f'./checkpoint/best_model_{args.model_mode}_{args.dataset_mode}_ckpt.t7'

    if not os.path.exists(ckpt_path):
        print(f"错误: 找不到模型文件 {ckpt_path}")
        print("请先训练模型或指定正确的 --checkpoint 路径")
        return

    # 加载权重
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    print(f"加载模型: {ckpt_path}")
    print(f"模型训练精度: Acc@1={checkpoint.get('best_acc1', 'N/A')}, Acc@5={checkpoint.get('best_acc5', 'N/A')}")

    # 推理
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        topk_probs, topk_indices = torch.topk(probabilities, args.topk, dim=1)

    print(f"\n预测结果 (Top-{args.topk}):")
    print("=" * 50)
    for i in range(args.topk):
        idx = topk_indices[0][i].item()
        prob = topk_probs[0][i].item() * 100
        label = classes[idx] if idx < len(classes) else f"未知({idx})"
        print(f"  {i+1}. {label:20s} ({idx:3d})  置信度: {prob:.2f}%")

    print("=" * 50)
    print(f"预测类别: {classes[topk_indices[0][0].item()]}")


if __name__ == "__main__":
    main()
