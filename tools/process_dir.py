import os
import shutil
import pandas as pd

# CSV文件路径和图片文件夹路径
csv_file = r"D:\archive\HAM10000_metadata.csv"
image_folder = r"D:\archive\Skin Cancer\Skin Cancer"
target_folder = r'D:\WorkSpace\vit\data'

# 标签到目标文件夹的映射关系
label_to_folder = {
    'nv': 'data/黑色素细胞痣',
    'akiec': 'data/光化性角化病和上皮内癌',
    'bcc':'data/基底细胞癌',
    'bkl':'data/良性角化病样病变',
    'df':'data/皮肤纤维瘤',
    'mel':'data/黑色素瘤',
    'vasc':'data/血管病变'
}

# 读取CSV文件
df = pd.read_csv(csv_file)

# 遍历CSV文件中的每一行
for index, row in df.iterrows():
    image_name = row['image_id']  # 假设CSV文件中有一列是图片的文件名
    label = row['dx']  # 假设CSV文件中有一列是标签

    # 如果标签存在于标签到文件夹的映射中
    if label in label_to_folder:
        # 获取目标文件夹路径
        target_folder = os.path.join(target_folder, label_to_folder[label])

        # 如果目标文件夹不存在，创建它
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

        # 构建图片的完整路径
        image_path = os.path.join(image_folder, image_name+'.jpg')

        # 构建目标文件的路径
        destination_path = os.path.join(target_folder, image_name+'.jpg')

        # 移动文件
        shutil.move(image_path, destination_path)

print("图片已根据标签成功分类并放入对应的文件夹！")
