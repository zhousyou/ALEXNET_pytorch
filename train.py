import torch
import torch.backends
import torch.backends.mps
from torchvision import transforms,datasets
from create_data import CustomDataset
from torch.utils.data import random_split,DataLoader
from model.Alexnet import Alexnet
import torch.nn as nn
import torch.optim as optim

#选择设备
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

#设置超参数
batch_size = 128
lr = 1e-3
epoch = 50
num_classes = 2
logs = 'logs/'
data_root = 'data/dataset.txt'
img_dir = 'data/image/train'

#数据预处理
transform = transforms.Compose(
    transforms.ToTensor(),
    transforms.Resize((224,224))
)

#加载数据集
dataset = CustomDataset(img_dir, data_root, transform)

#划分数据集
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(
    dataset=dataset,
    lengths=[train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

#创建dataloader

train_loader = DataLoader(
    train_dataset,
    batch_size,
    shuffle=True,
    num_workers=2
)

val_loader = DataLoader(
    val_dataset,
    batch_size,
    shuffle=False,
    num_workers=2
)

#定义模型
model = Alexnet()
model = model.to(device)

#定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr=lr)

#定义训练的函数
def train():
    best_acc = 0.0
    for e in range(epoch):
        model.train()
        running_loss = 0.0

        for image, label in train_loader:
            image = image.to(device)
            label = label.to(device)

            #前向传播
            outputs = model(image)
            loss = criterion(outputs, label)

            #反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * image.size(0)

#验证的函数
def validation():
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for image, label in val_loader:
            image = image.to(device)
            label = label.to(device)

            outputs = model(image)
            loss = criterion(outputs, label)





