import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
])

train_data = datasets.FashionMNIST(root='data', 
                                   train=True,        # 학습용 데이터셋 설정(True)
                                   download=True, 
                                   transform=transform                
                                  )

test_data = datasets.FashionMNIST(root='data', 
                                  train=False,        # 검증용 데이터셋 설정(False)
                                  download=True, 
                                  transform=transform
                                 )

labels_map = {
    0: "t-shirt/top",
    1: "trouser",
    2: "pullover",
    3: "dress",
    4: "coat",
    5: "sandal",
    6: "shirt",
    7: "sneaker",
    8: "bag",
    9: "ankle boot",
}

batch_size = 32
num_workers = 4

train_loader = torch.utils.data.DataLoader(train_data, 
                                           batch_size=batch_size,
                                           shuffle=True, 
                                           num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, 
                                          batch_size=batch_size,
                                          shuffle=False, 
                                          num_workers=num_workers)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DNNModel(nn.Module):
    def __init__(self):
        super(DNNModel, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)
        self.output = nn.Linear(32, 10)
    
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.output(x)
        return x

model = DNNModel()
model.to(device)

optimizer = optim.Adam(model.parameters(), lr = 0.0005)

loss_fn = nn.CrossEntropyLoss()

def model_train(model, data_loader, loss_fn, optimizer, device):
    model.train()
    running_loss = 0
    corr = 0
    
    for img, lbl in data_loader:
        img, lbl = img.to(device), lbl.to(device)
        
        optimizer.zero_grad()
        output = model(img)
        loss = loss_fn(output, lbl)
        loss.backward()
        optimizer.step()

        _, pred = output.max(dim=1)
        corr += pred.eq(lbl).sum().item()
        
        running_loss += loss.item() * img.size(0)
        
    acc = corr / len(data_loader.dataset)
    
    return running_loss / len(data_loader.dataset), acc

def model_evaluate(model, data_loader, loss_fn, device):
    model.eval()
    
    with torch.no_grad():
        corr = 0
        running_loss = 0
        
        for img, lbl in data_loader:
            img, lbl = img.to(device), lbl.to(device)
            
            output = model(img)
            _, pred = output.max(dim=1)
            corr += torch.sum(pred.eq(lbl)).item()
            running_loss += loss_fn(output, lbl).item() * img.size(0)
        
        acc = corr / len(data_loader.dataset)
        
        return running_loss / len(data_loader.dataset), acc

num_epochs = 20

min_loss = np.inf

# Epoch 별 훈련 및 검증을 수행합니다.
for epoch in range(num_epochs):
    # Model Training
    # 훈련 손실과 정확도를 반환 받습니다.
    train_loss, train_acc = model_train(model, train_loader, loss_fn, optimizer, device)

    # 검증 손실과 검증 정확도를 반환 받습니다.
    val_loss, val_acc = model_evaluate(model, test_loader, loss_fn, device)   
    
    # val_loss 가 개선되었다면 min_loss를 갱신하고 model의 가중치(weights)를 저장합니다.
    if val_loss < min_loss:
        print(f'[INFO] val_loss has been improved from {min_loss:.5f} to {val_loss:.5f}. Saving Model!')
        min_loss = val_loss
    
    # Epoch 별 결과를 출력합니다.
    print(f'epoch {epoch+1:02d}, val_loss: {val_loss:.5f}, val_accuracy: {val_acc:.5f}')