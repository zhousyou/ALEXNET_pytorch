import torch
from torchvision import transforms,datasets

#选择设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#设置超参数
batch_size = 128
lr = 1e-3
epoch = 50
num_classes = 2
logs = 'logs/'
data_root = 'data/dataset.txt'


#数据预处理
transform = transforms.Compose(
    transforms.ToTensor()
)

#加载数据集




