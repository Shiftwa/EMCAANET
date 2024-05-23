import os

def get_subfolders(directory):
    """
    Get all subfolder names in the specified directory.

    Args:
        directory (str): Path to the directory.

    Returns:
        list: List of subfolder names.
    """
    subfolders = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
    return subfolders

# # Example usage
# if __name__ == "__main__":
#     directory = "/root/autodl-tmp/imagenet/val"
#     subfolders = get_subfolders(directory)
#     print("Subfolders:", subfolders)
# from torchvision import datasets, transforms

# # 定义数据转换
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])

# # 创建 ImageFolder 实例
# valdir = '/root/autodl-tmp/imagenet/val'
# val_dataset = datasets.ImageFolder(
#     valdir,
#     transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         normalize,
#     ])
# )

# # 获取类名列表和类名到标签的映射
# class_names = val_dataset.classes
# class_to_idx = val_dataset.class_to_idx

# print("Class names:", class_names)
# print("Class to index mapping:", class_to_idx)
def read_labels_file(file_path):
    labels_dict = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(' ')
            label_id = parts[0]
            label_name = ' '.join(parts[1:])
            labels_dict[label_id] = label_name
    return labels_dict

file_path = 'mapping.txt'  # 替换成你的文件路径
labels_dict = read_labels_file(file_path)

# 给定的类别标识符列表
label_ids = ['n01558993', 'n01692333', 'n01729322', 'n01735189', 'n01749939', 'n01773797', 'n01820546', 'n01855672', 'n01978455', 'n01980166', 'n01983481', 'n02009229', 'n02018207', 'n02085620', 'n02086240', 'n02086910', 'n02087046', 'n02089867', 'n02089973', 'n02090622', 'n02091831', 'n02093428', 'n02099849', 'n02100583', 'n02104029', 'n02105505', 'n02106550', 'n02107142', 'n02108089', 'n02109047', 'n02113799', 'n02113978', 'n02114855', 'n02116738', 'n02119022', 'n02123045', 'n02138441', 'n02172182', 'n02231487', 'n02259212', 'n02326432', 'n02396427', 'n02483362', 'n02488291', 'n02701002', 'n02788148', 'n02804414', 'n02859443', 'n02869837', 'n02877765', 'n02974003', 'n03017168', 'n03032252', 'n03062245', 'n03085013', 'n03259280', 'n03379051', 'n03424325', 'n03492542', 'n03494278', 'n03530642', 'n03584829', 'n03594734', 'n03637318', 'n03642806', 'n03764736', 'n03775546', 'n03777754', 'n03785016', 'n03787032', 'n03794056', 'n03837869', 'n03891251', 'n03903868', 'n03930630', 'n03947888', 'n04026417', 'n04067472', 'n04099969', 'n04111531', 'n04127249', 'n04136333', 'n04229816', 'n04238763', 'n04336792', 'n04418357', 'n04429376', 'n04435653', 'n04485082', 'n04493381', 'n04517823', 'n04589890', 'n04592741', 'n07714571', 'n07715103', 'n07753275', 'n07831146', 'n07836838', 'n13037406', 'n13040303']

# 使用之前读取的标签字典获取对应的描述
labels = []
for label_id in label_ids:
    if label_id in labels_dict:
        labels.append(labels_dict[label_id])
    else:
        labels.append("Unknown")

# 打印描述列表
print(len(labels))


