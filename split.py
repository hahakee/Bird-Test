import os
import random
import shutil

# 数据集路径
source_dataset_dir = r'D:\BirdTest\ultralytics-main\ultralytics-main\images'
train_dir = r'D:\BirdTest\ultralytics-main\ultralytics-main\images1\train'
valid_dir = r'D:\BirdTest\ultralytics-main\ultralytics-main\images1\val'

# 定义训练集和验证集的比例
train_ratio = 0.8  # 80%用于训练，20%用于验证

# 创建训练集和验证集的根目录
os.makedirs(train_dir, exist_ok=True)
os.makedirs(valid_dir, exist_ok=True)

# 获取所有类别（每个类别是一个子文件夹）
categories = os.listdir(source_dataset_dir)

# 遍历每个类别
for category in categories:
    category_dir = os.path.join(source_dataset_dir, category)
    if not os.path.isdir(category_dir):  # 确保是文件夹
        continue

    # 获取该类别下的所有图片文件
    images = os.listdir(category_dir)
    random.shuffle(images)  # 随机打乱图片列表

    # 计算训练集和验证集的分割点
    train_count = int(len(images) * train_ratio)

    # 创建训练集和验证集的类别子目录
    train_category_dir = os.path.join(train_dir, category)
    valid_category_dir = os.path.join(valid_dir, category)
    os.makedirs(train_category_dir, exist_ok=True)
    os.makedirs(valid_category_dir, exist_ok=True)

    # 将图片复制到训练集或验证集目录
    for i, image in enumerate(images):
        src_path = os.path.join(category_dir, image)
        if i < train_count:
            dst_path = os.path.join(train_category_dir, image)
        else:
            dst_path = os.path.join(valid_category_dir, image)
        shutil.copy(src_path, dst_path)  # 复制图片

print("数据集成功划分为训练集和验证集！")
