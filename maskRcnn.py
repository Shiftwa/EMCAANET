import torch
import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torch import nn
from torchvision.models.resnet import resnet50
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
import os
from PIL import Image

# 自定义修改后的 ResNet
class ModifiedResNet(nn.Module):
    def __init__(self, pretrained_path=None):
        super(ModifiedResNet, self).__init__()
        self.backbone = resnet50(pretrained=False)
        if pretrained_path:
            self.backbone.load_state_dict(torch.load(pretrained_path))

        # 如果有进一步修改，可以在这里进行
        # 例如： self.backbone.layer1[0].conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        c2 = self.backbone.layer1(x)
        c3 = self.backbone.layer2(c2)
        c4 = self.backbone.layer3(c3)
        c5 = self.backbone.layer4(c4)

        return c2, c3, c4, c5

# 手动构建 FPN
class BackboneWithFPNCustom(BackboneWithFPN):
    def __init__(self, backbone):
        return_layers = {
            '0': 'c2',
            '1': 'c3',
            '2': 'c4',
            '3': 'c5',
        }
        in_channels_list = [256, 512, 1024, 2048]
        out_channels = 256
        super(BackboneWithFPNCustom, self).__init__(backbone, return_layers, in_channels_list, out_channels)

# 定义 COCO 数据集类
class COCODataset(torch.utils.data.Dataset):
    def __init__(self, root, annFile, transforms=None):
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgToAnns.keys())
        self.transforms = transforms

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert("RGB")

        num_objs = len(anns)

        boxes = []
        masks = []
        for i in range(num_objs):
            xmin, ymin, width, height = anns[i]['bbox']
            xmax = xmin + width
            ymax = ymin + height
            boxes.append([xmin, ymin, xmax, ymax])
            masks.append(coco.annToMask(anns[i]))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([img_id])
        area = torch.as_tensor([ann['area'] for ann in anns], dtype=torch.float32)
        iscrowd = torch.as_tensor([ann['iscrowd'] for ann in anns], dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)

def get_transform(train):
    transforms = []
    transforms.append(F.to_tensor)
    if train:
        transforms.append(F.random_horizontal_flip(0.5))
    return torchvision.transforms.Compose(transforms)

# 加载数据集
root = 'path/to/coco/images'
annFile = 'path/to/coco/annotations/instances_train2017.json'
dataset = COCODataset(root, annFile, transforms=get_transform(train=True))
data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))

# 加载自定义的 ResNet 并构建 FPN
pretrained_resnet_path = 'path/to/your/pretrained/resnet50.pth'
modified_resnet = ModifiedResNet(pretrained_resnet_path)
backbone_with_fpn = BackboneWithFPNCustom(modified_resnet)

# 定义 Mask R-CNN 模型
model = MaskRCNN(backbone_with_fpn, num_classes=2)  # 类别数 + 背景

# 将模型移到 GPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# 设置优化器
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    i = 0
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"Epoch: {epoch}, Iteration: {i}, Loss: {losses.item()}")
        i += 1

    lr_scheduler.step()

    print(f"Epoch {epoch} finished")

print("Training completed")

# 保存模型
torch.save(model.state_dict(), "maskrcnn_modified_resnet50_coco.pth")
