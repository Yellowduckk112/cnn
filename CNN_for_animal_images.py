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
        print(f"✓ 成功加载数据集: {len(self.labels)} 张图片, 共 {len(self.classes)} 个类别")
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        self.img_paths = [os.path.join(root, img) for img in os.listdir(root)]
    
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
    def __init__(self, num_classes=5):
        super(CNN, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 输入: 3*224*224
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2) 
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1, 2))
        )
        self.lin = nn.Sequential(
            nn.Linear(256 * 6 * 6, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.lin(x)
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
    trans = transforms.Compose(trans)
    animal_train = AnimalDataset(root='./animal_data/train_data', transform=trans)
    animal_test = AnimalDataset(root='./animal_data/test_data', transform=trans)
    return data.DataLoader(animal_train, batch_size, shuffle=True, num_workers=get_dataloader_workers()), \
        data.DataLoader(animal_test, batch_size, shuffle=False, num_workers=get_dataloader_workers())

def train(net, train_iter, test_iter, num_epochs, lr, device):
    test_accs = []
    test_losses = []
    train_accs = []
    train_losses = []
    net = net.to(device)
    print("training on ", device)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    
    net.train()
    for epoch in range(num_epochs):
        train_loss_sum = 0
        train_acc_sum = 0
        n = 0

        for X, y in train_iter:
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_loss_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]

        print(f'epoch {epoch + 1}, loss {train_loss_sum / n:.4f}, train acc {train_acc_sum / n:.3f}')
        train_accs.append(train_acc_sum / n)
        train_losses.append(train_loss_sum / n)

        test_loss_sum = 0
        test_acc_sum = 0
        n = 0
        net.eval()
        with torch.no_grad():
            for X, y in test_iter:
                X, y = X.to(device), y.to(device)
                y_hat = net(X)
                l = loss(y_hat, y)
                test_loss_sum += l.cpu().item()
                test_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
                n += y.shape[0]

        print(f'test loss {test_loss_sum / n:.4f}, test acc {test_acc_sum / n:.3f}')
        test_accs.append(test_acc_sum / n)
        test_losses.append(test_loss_sum / n)

    return train_accs, train_losses, test_accs, test_losses

if __name__ == "__main__":
    batch_size = 16
    lr = 0.01
    num_epochs = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_iter, test_iter = load_data_animal(batch_size=batch_size)

    net = CNN(num_classes=5)
    train_accs, train_losses, test_accs, test_losses = train(net, train_iter, test_iter, num_epochs, lr, device)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='train loss')
    plt.plot(test_losses, label='test loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='train acc')
    plt.plot(test_accs, label='test acc')
    plt.legend()
    plt.show()
    # image, label = next(iter(train_iter))
    # print(image.shape, label)