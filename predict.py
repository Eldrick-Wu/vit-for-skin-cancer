import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from vit_model import vit_base_patch16_224_in21k as create_model


def predict(img_path):
    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    # load image
    img = Image.open(img_path)
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)
    """
     # read class_indict
    json_path = 'class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
    with open(json_path, "r" ) as f:
        class_indict = json.load(f)
    """
    model.eval()
    with torch.no_grad():
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()


    '''
    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    print(print_res)
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    plt.show()
    '''
    return str(predict_cla)


# 读取所有测试图像和真实标签
def get_image_paths_and_labels(root_dir,class_name_to_index):
    image_paths = []
    true_labels = []

    # 遍历每个类别的文件夹
    for label_name in os.listdir(root_dir):
        label_path = os.path.join(root_dir, label_name)
        if os.path.isdir(label_path):
            # 获取类别名称对应的索引
            if label_name in class_name_to_index:
                label_index = class_name_to_index[label_name]

                # 获取该文件夹下的所有图片路径
                for img_name in os.listdir(label_path):
                    if img_name.endswith(('.png', '.jpg', '.jpeg')):  # 根据实际文件格式修改
                        img_path = os.path.join(label_path, img_name)
                        image_paths.append(img_path)
                        true_labels.append(label_index)

    return image_paths, true_labels


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = create_model(num_classes=7, has_logits=False).to(device)
    model_weight_path = "./weights/model-9.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device, weights_only=True))
    with open('class_indices.json', 'r',encoding='utf8') as f:
        class_indices = json.load(f)

    # 反转索引以从类别名称到索引进行映射
    class_name_to_index = {v: k for k, v in class_indices.items()}
    # 假设测试文件夹路径为 'test'
    test_dir = 'test'
    image_paths, y_true = get_image_paths_and_labels(test_dir,class_name_to_index)
    # 生成预测标签
    y_pred = [predict(img_path) for img_path in image_paths]

    report = classification_report(y_true, y_pred, target_names=class_name_to_index.values())
    print(report)

