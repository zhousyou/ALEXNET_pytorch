import torch
from model.Alexnet import Alexnet
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image

class InterfacePipeline():
    def __init__(self, model_path, class_names, device='cuda'):
        self.class_names = class_names
        self.device = torch.device(device)

        #加载模型
        self.model = Alexnet()
        self.model.load_state_dict(torch.load(model_path))
        self.model = self.model.to(self.device)
        self.model = self.model.eval()

        #预处理
        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])
    
    def predict(self, img_path):

        img = Image.open(img_path)
        img_tensor = self.transform(img)
        img_batch = img_tensor.unsqueeze(0).to(self.device)

        #推理
        with torch.no_grad():
            output = self.model(img_batch)
       
        probabilities = F.softmax(output)
        print(probabilities)
        predicted = torch.argmax(output).item()

        return {
            'classes':self.class_names[predicted],
            'probabilities':torch.max(probabilities).item()
        }
    
if __name__ =="__main__":
    CLASS_NAMES = ['cat', 'dog']
    pipeline = InterfacePipeline(
        model_path='best_model.pth',
        class_names=CLASS_NAMES
    )
    result = pipeline.predict('test_img/test4.jpg')
    print(f" 预测结果:{result['classes']}, 置信度：{result['probabilities']}")
