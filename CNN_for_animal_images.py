import torch
from torch import nn
import matplotlib.pyplot as plt
from torch.utils import data
from torchvision import transforms
import torchvision
import os
from PIL import Image

transformer = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

class AnimalDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.labels = [self.extract_label(img) for img in os.listdir(root) if self.extract_label(img) != -1]
        self.classes = list(set(self.labels))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        self.img_paths = [os.path.join(root, img) for img in os.listdir(root)]
        print(self.img_paths[:10])
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_path = self.img_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"✗ 加载图片失败: {img_path}, 错误: {e}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        label_str = self.labels[idx]
        label = self.class_to_idx[label_str]
        for cls, idx in self.class_to_idx.items():
            print(cls, idx)
        if self.transform:
            image = self.transform(image)
        
        return image, label

    def extract_label(self, img_name):
        for i in range(len(img_name)):
            if img_name[i].isalpha():
                continue
            else:
                return img_name[:i].lower()
        return -1

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def get_dataloader_workers():
    return 4

def load_data_animal(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.ToTensor()
    animal_train = AnimalDataset(root='./animal_data', transform=trans)
    return data.DataLoader(animal_train, batch_size, shuffle=True, num_workers=get_dataloader_workers())

batch_size = 16
lr = 0.001
num_epochs = 10

animal_iter = load_data_animal(batch_size=batch_size)
image, label = next(iter(animal_iter))
print(image.shape, label)