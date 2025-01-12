import os
import shutil
from sklearn.model_selection import train_test_split

# 定义图片文件夹和目标文件夹路径
data_folder = r'D:\WorkSpace\vit\data'

test_folder = r'D:\WorkSpace\vit\test'

# 获取所有类别（即文件夹的名称）
categories = [category for category in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, category))]

# 函数：根据类别划分文件夹中的图片
def move_files(category, file_list, source_folder, target_folder):
    category_folder = os.path.join(source_folder, category)
    for file_name in file_list:
        source_path = os.path.join(category_folder, file_name)
        target_path = os.path.join(target_folder, category, file_name)

        # 如果目标文件夹不存在，创建它
        if not os.path.exists(os.path.dirname(target_path)):
            os.makedirs(os.path.dirname(target_path))

        # 移动文件
        shutil.move(source_path, target_path)

# 遍历每个类别
for category in categories:

    image_files = [f for f in os.listdir(os.path.join(data_folder, category)) if os.path.isfile(os.path.join(data_folder, category, f))]

    train_files, test_files = train_test_split(image_files, test_size=0.15, random_state=42)
    move_files(category, test_files, data_folder, test_folder)

print("数据集已成功划分并按类别移动到对应文件夹！")
