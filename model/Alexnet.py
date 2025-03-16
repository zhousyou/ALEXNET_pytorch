import  torch
import torch.nn as nn
from torchsummary import summary
import cv2
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import torchvision

class Alexnet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96,kernel_size=11,
                                stride=4, padding=2),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5,
                      stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),

            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3,
                      stride=1, padding=1),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3,
                      stride=1, padding=1),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3,
                      stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),

            nn.Flatten(),
            nn.Linear(9216, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096,1024),
            nn.ReLU(),
            nn.Linear(1024,2),
            # nn.Softmax()
        )
    def forward(self, x):
        return self.net(x)
# model = Alexnet()
# summary(model, (3, 224, 224))


# img = cv2.imread("data/image/train/cat.0.jpg")
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img = Image.open("data/image/train/cat.0.jpg").convert('RGB')
# print(img.shape)
# img = Image.fromarray(img)
# model = Alexnet()
# transform = transforms.Compose([
#     transforms.Resize((224,224)),
#     transforms.ToTensor()
# ])
# img = transform(img)
# img_batch = img.unsqueeze(0)
# # dataloader = DataLoader(img, 1)
# # for img in dataloader:
# output = model(img_batch)
# ans=torch.argmax(output).item()
# print(type(output), ans)