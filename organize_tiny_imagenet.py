"""
整理 Tiny ImageNet 验证集为 ImageFolder 格式：
data/tiny-imagenet-200/
├── train/n01443537/    (500张/类, 直接可用)
├── val/n01443537/      (需要整理)
└── test/images/        (无标签)
"""
import os
import shutil

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'tiny-imagenet-200')

# 整理验证集：根据 val_annotations.txt 把图片复制到 val/类别/ 目录
val_dir = os.path.join(data_dir, 'val')
val_images_dir = os.path.join(val_dir, 'images')
val_annotations = os.path.join(val_dir, 'val_annotations.txt')

if not os.path.exists(val_annotations):
    print("错误: 找不到 val_annotations.txt，请确认数据集完整")
    exit(1)

print("正在整理验证集...")
with open(val_annotations, 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) < 2:
            continue
        img_name, class_id = parts[0], parts[1]
        src = os.path.join(val_images_dir, img_name)
        dst_dir = os.path.join(val_dir, class_id)
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, img_name)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)

# 清理旧的 images 目录和注释文件（避免 ImageFolder 误读为类别）
if os.path.exists(val_images_dir):
    shutil.rmtree(val_images_dir)
    print("  已删除旧的 images/ 目录")
if os.path.exists(val_annotations):
    os.remove(val_annotations)
    print("  已删除 val_annotations.txt")

# 验证
print("验证整理结果...")
val_class_count = len([d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))])
val_file_count = sum(len(files) for _, _, files in os.walk(val_dir))
print(f"  验证集: {val_class_count} 个类别, {val_file_count} 张图片")

train_class_count = len([d for d in os.listdir(os.path.join(data_dir, 'train')) if os.path.isdir(os.path.join(data_dir, 'train', d))])
train_file_count = sum(len(files) for _, _, files in os.walk(os.path.join(data_dir, 'train')))
print(f"  训练集: {train_class_count} 个类别, {train_file_count} 张图片")

print("\n完成！")
