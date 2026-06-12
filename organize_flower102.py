"""
整理 Oxford 102 Flower 数据集为 ImageFolder 格式：
data/flower102/
  ├── train/class_id/  (1020张)
  ├── val/class_id/    (1020张)
  └── test/class_id/   (6149张)
"""
import scipy.io
import shutil
import os

# 路径
data_dir = os.path.dirname(os.path.abspath(__file__))
flower_dir = os.path.join(data_dir, 'data', 'flower102')
jpg_dir = os.path.join(flower_dir, 'jpg')

# 加载标签
labels = scipy.io.loadmat(os.path.join(flower_dir, 'imagelabels.mat'))['labels'][0]  # shape: (8189,)

# 加载数据集划分
setid = scipy.io.loadmat(os.path.join(flower_dir, 'setid.mat'))
trnid = setid['trnid'][0]    # 1020
valid = setid['valid'][0]    # 1020
tstid = setid['tstid'][0]    # 6149

splits = {
    'train': trnid,
    'val': valid,
    'test': tstid,
}

for split_name, ids in splits.items():
    split_dir = os.path.join(flower_dir, split_name)
    print(f"正在整理 {split_name} 集 ({len(ids)} 张)...")
    count = 0
    for img_id in ids:
        # 标签从1开始，类别目录也以1开始
        label = int(labels[img_id - 1])
        class_dir = os.path.join(split_dir, str(label))
        os.makedirs(class_dir, exist_ok=True)

        src = os.path.join(jpg_dir, f"image_{img_id:05d}.jpg")
        dst = os.path.join(class_dir, f"image_{img_id:05d}.jpg")

        if os.path.exists(src):
            shutil.copy2(src, dst)
            count += 1
        else:
            print(f"  警告: 找不到 {src}")
    print(f"  {split_name} 完成，共复制 {count} 张")

print("\n全部完成！目录结构:")
for split_name in splits:
    split_dir = os.path.join(flower_dir, split_name)
    class_count = len([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])
    file_count = sum(len(files) for _, _, files in os.walk(split_dir))
    print(f"  {split_name}/: {class_count} 个类别, {file_count} 张图片")
