import torch
import torch.backends
# import torch.backends.mps
from torchvision import transforms,datasets
from create_data import CustomDataset
from torch.utils.data import random_split,DataLoader
from model.Alexnet import Alexnet
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time

#选择设备
# device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#设置超参数
batch_size = 128
lr = 1e-3
epoch = 50
num_classes = 2
logs = 'logs/'
data_root = 'data/dataset.txt'
img_dir = 'data/image/train'

#数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224,224))
])

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

def test_train():
    model.train()
    total_loss = 0
    for img, label in train_loader:
        print(img.shape, label.shape)  #img:128,3,224,224 label:128
        optimizer.zero_grad()
        img = img.to(device)
        label = label.to(device)  
        #前向传播
        outputs = model(img)  #128*2
        max_vaules, max_indexed = torch.max(outputs, 1)
        print(max_indexed.shape)
        # print(outputs.shape)
        #计算损失
        loss = criterion(outputs, label)
        print(loss)
    # img, label = train_dataset[0]
    # print(img.shape)
    # print(label)

        


#定义训练的函数
def train():
    best_acc = 0.0
    for e in range(epoch):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {e}")
        start_time = time.time()

        for image, label in progress_bar:
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
            # print(running_loss)

        #验证
        epoch_loss = running_loss/len(train_loader.dataset)
        # val_loss, val_acc = validation()
        
        progress_bar.set_postfix({
            "LOSS": f"{epoch_loss:.4f}",
            "Time": time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
        })
        progress_bar.refresh()  # 强制刷新进度条
        print(f'Epoch [{e+1}/{epoch}], LOSS: {epoch_loss: .4f}')
        # print(f'Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        # 保存最佳模型
        # if val_acc > best_acc:
        #     best_acc = val_acc
        #     torch.save(model.state_dict(), 'best_model.pth')


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
            _, predicted = torch.argmax(outputs,1).item()  #torch.max()返回两个值：max_values, max_indexes;outputs.shape:(128,2);predicted.shpae:128
            total += label.size(0)   #label.shape:128
            correct += (label==predicted).sum().item()
            running_loss += loss.item()*image.size(0)
        val_loss = running_loss/len(val_loader.dataset)
        val_acc = correct*100/total
        return val_loss, val_acc


if __name__=="__main__":
    # test_train()
    # print(train_loader[0])
    train()
