import torch
import torchvision.transforms as transforms
from PIL import Image
import argparse
import os
from emcaa_resnet import emcaanet_resnet50

import matplotlib.pyplot as plt
import numpy as np

# Argument parser for inference
parser = argparse.ArgumentParser(description='PyTorch ImageNet Inference')
parser.add_argument('--model-path', type=str, required=True, help='path to the trained model')
parser.add_argument('--image-path', type=str, required=False, help='path to the input image')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use')
parser.add_argument('--labels', type=str, default='labels.txt', help='path to the labels file')

def load_model(model_path, gpu=None):
    # Load the trained model
    model = emcaanet_resnet50()
    if gpu is not None:
        model = model.cuda(gpu)
        checkpoint = torch.load(model_path, map_location=f'cuda:{gpu}')
    else:
        checkpoint = torch.load(model_path, map_location='cpu')
    
    # model.load_state_dict(checkpoint['state_dict'])
    model.load_state_dict({k.replace('module.',''):v for k,v in checkpoint['state_dict'].items()})
 
# 相当于用''代替'module.'。
#直接使得需要的键名等于期望的键名。

    
    model.eval()
    return model
# 函数用于在图像上显示真实标签和预测的Top1标签
def show_images_with_labels(image_paths, true_labels, predicted_labels):
    # 设置图形的大小和列数
    num_images = len(image_paths)
    num_cols = 3
    num_rows = (num_images + num_cols - 1) // num_cols

    # 创建一个新的Matplotlib图形
    plt.figure(figsize=(15, 5*num_rows))

    # 遍历每个图像
    for i, (image_path, true_label, predicted_label) in enumerate(zip(image_paths, true_labels, predicted_labels), 1):
        # 读取图像
        image = plt.imread(image_path)
        
        # 绘制图像
        plt.subplot(num_rows, num_cols, i)
        plt.imshow(image)
        plt.title('True Label: {}\nPredicted Label (Top1): {}'.format(true_label, predicted_label))
        plt.axis('off')

    # 调整布局
    plt.tight_layout()
    # plt.show()
    folder = "/root/autodl-tmp/math3544/classsfy_result/4.png"
    plt.savefig(folder, bbox_inches='tight')

def preprocess_image(image_path):
    # Preprocess the input image
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path)
    image = preprocess(image).unsqueeze(0)  # Add batch dimension
    return image

def load_labels(labels_path):
    # Load labels from file
    with open(labels_path) as f:
        labels = [line.strip() for line in f.readlines()]
    return labels

def predict(model, image, gpu=None):
    # Perform inference
    if gpu is not None:
        image = image.cuda(gpu)
    with torch.no_grad():
        output = model(image)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    return probabilities

def main():
    args = parser.parse_args()

    # Load model
    model = load_model(args.model_path, args.gpu)
    
    # Preprocess image
    # image = preprocess_image(args.image_path)
    
    # Load labels
    # labels = load_labels(args.labels)
    labels = ['robin, American robin, Turdus migratorius', 'Gila monster, Heloderma suspectum', 'hognose snake, puff adder, sand viper', 'garter snake, grass snake', 'green mamba', 'garden spider, Aranea diademata', 'lorikeet', 'goose', 'rock crab, Cancer irroratus', 'fiddler crab', 'American lobster, Northern lobster, Maine lobster, Homarus americanus', 'little blue heron, Egretta caerulea', 'American coot, marsh hen, mud hen, water hen, Fulica americana', 'Chihuahua', 'Shih-Tzu', 'papillon', 'toy terrier', 'Walker hound, Walker foxhound', 'English foxhound', 'borzoi, Russian wolfhound', 'Saluki, gazelle hound', 'American Staffordshire terrier, Staffordshire terrier, American pit bull terrier, pit bull terrier', 'Chesapeake Bay retriever', 'vizsla, Hungarian pointer', 'kuvasz', 'komondor', 'Rottweiler', 'Doberman, Doberman pinscher', 'boxer', 'Great Dane', 'standard poodle', 'Mexican hairless', 'coyote, prairie wolf, brush wolf, Canis latrans', 'African hunting dog, hyena dog, Cape hunting dog, Lycaon pictus', 'red fox, Vulpes vulpes', 'tabby, tabby cat', 'meerkat, mierkat', 'dung beetle', 'walking stick, walkingstick, stick insect', 'leafhopper', 'hare', 'wild boar, boar, Sus scrofa', 'gibbon, Hylobates lar', 'langur', 'ambulance', 'bannister, banister, balustrade, balusters, handrail', 'bassinet', 'boathouse', 'bonnet, poke bonnet', 'bottlecap', 'car wheel', 'chime, bell, gong', 'cinema, movie theater, movie theatre, movie house, picture palace', 'cocktail shaker', 'computer keyboard, keypad', 'Dutch oven', 'football helmet', 'gasmask, respirator, gas helmet', 'hard disc, hard disk, fixed disk', 'harmonica, mouth organ, harp, mouth harp', 'honeycomb', 'iron, smoothing iron', 'jean, blue jean, denim', 'lampshade, lamp shade', 'laptop, laptop computer', 'milk can', 'mixing bowl', 'modem', 'moped', 'mortarboard', 'mousetrap', 'obelisk', 'park bench', 'pedestal, plinth, footstall', 'pickup, pickup truck', 'pirate, pirate ship', 'purse', 'reel', 'rocking chair, rocker', 'rotisserie', 'safety pin', 'sarong', 'ski mask', 'slide rule, slipstick', 'stretcher', 'theater curtain, theatre curtain', 'throne', 'tile roof', 'tripod', 'tub, vat', 'vacuum, vacuum cleaner', 'window screen', 'wing', 'head cabbage', 'cauliflower', 'pineapple, ananas', 'carbonara', 'chocolate sauce, chocolate syrup', 'gyromitra', 'stinkhorn, carrion fungus']
    labels2= ['n01558993', 'n01692333', 'n01729322', 'n01735189', 'n01749939', 'n01773797', 'n01820546', 'n01855672', 'n01978455', 'n01980166', 'n01983481', 'n02009229', 'n02018207', 'n02085620', 'n02086240', 'n02086910', 'n02087046', 'n02089867', 'n02089973', 'n02090622', 'n02091831', 'n02093428', 'n02099849', 'n02100583', 'n02104029', 'n02105505', 'n02106550', 'n02107142', 'n02108089', 'n02109047', 'n02113799', 'n02113978', 'n02114855', 'n02116738', 'n02119022', 'n02123045', 'n02138441', 'n02172182', 'n02231487', 'n02259212', 'n02326432', 'n02396427', 'n02483362', 'n02488291', 'n02701002', 'n02788148', 'n02804414', 'n02859443', 'n02869837', 'n02877765', 'n02974003', 'n03017168', 'n03032252', 'n03062245', 'n03085013', 'n03259280', 'n03379051', 'n03424325', 'n03492542', 'n03494278', 'n03530642', 'n03584829', 'n03594734', 'n03637318', 'n03642806', 'n03764736', 'n03775546', 'n03777754', 'n03785016', 'n03787032', 'n03794056', 'n03837869', 'n03891251', 'n03903868', 'n03930630', 'n03947888', 'n04026417', 'n04067472', 'n04099969', 'n04111531', 'n04127249', 'n04136333', 'n04229816', 'n04238763', 'n04336792', 'n04418357', 'n04429376', 'n04435653', 'n04485082', 'n04493381', 'n04517823', 'n04589890', 'n04592741', 'n07714571', 'n07715103', 'n07753275', 'n07831146', 'n07836838', 'n13037406', 'n13040303']
    # Predict
   

    # 示例图像路径列表、真实标签列表和预测的Top1标签列表
    image_paths = ["/root/autodl-tmp/imagenet/val/n01558993/ILSVRC2012_val_00006030.JPEG", "/root/autodl-tmp/imagenet/val/n01692333/ILSVRC2012_val_00006500.JPEG", "/root/autodl-tmp/imagenet/val/n01729322/ILSVRC2012_val_00011295.JPEG","/root/autodl-tmp/imagenet/val/n01980166/ILSVRC2012_val_00001985.JPEG"]  # 图像路径列表
    predicted_labels = []
    for image_path in image_paths:
        img = preprocess_image(image_path)
        probabilities = predict(model, img, args.gpu)
         # Get top 5 predictions
        top1_prob, top1_catid = torch.topk(probabilities, 1)
        predict_label = labels[top1_catid[0]].split(",")[0]+' '+f"{top1_prob[0].item()*100:.2f}%"
        predicted_labels.append(predict_label) 
        print(predicted_labels)
        
    true_labels = ["robin", "Gila monster", "hognose snake","fidller crab"]  # 真实标签列表
    # 预测的Top1标签列表
   
    # 显示图像和标签
    show_images_with_labels(image_paths, true_labels, predicted_labels)

   



if __name__ == '__main__':
    main()




