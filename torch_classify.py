import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import pandas as pd
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.raw_data = dataframe
        self.labels = list(map(lambda x : [1] if x=='good' else [0], self.raw_data['Quality']))
        self.raw_data = self.raw_data.drop(labels='Quality', axis = 1)
        self.raw_data = self.raw_data.drop(labels='A_id', axis = 1)
        self.raw_data = self.raw_data.to_numpy()
        self.labels = np.array(self.labels)

        self.raw_data = torch.from_numpy(self.raw_data).float()
        self.labels = torch.from_numpy(self.labels).float()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = self.raw_data[idx]
        label = self.labels[idx]
        if self.transform:
            item = self.transform(item)
        if self.target_transform:
            label = self.target_transform(label)
        sample = {'item' : item, 'label' : label}
        return item, label

def split_data(csv_data):
    splited_idx = int(len(csv_data) * 0.7)
    splitted_train = csv_data[:splited_idx]
    splitted_test = csv_data[splited_idx:]
    return splitted_train, splitted_test

csv_dir = '/home/jetson/pytorch/data/apple_quality.csv'

csv_data = pd.read_csv(csv_dir)
csv_train, csv_test = split_data(csv_data)

train_data = CustomDataset(csv_train)
test_data = CustomDataset(csv_test)

batch_size = 16

train_dataloader = DataLoader(dataset = train_data, batch_size=batch_size)
test_dataloader = DataLoader(dataset = test_data, batch_size=batch_size)
#--> __getitem__()을 통해서 item-label을 mapping

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Linear(7, 1)

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) *len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy : {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}\n")


epochs = 5

for t in range(epochs):
    print(f"Epoch {t+1}\n---------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

# model.eval()
# x, y = test_data[0][0], test_data[0][1]
# start = time.time()
# with torch.no_grad():
#     x = x.to(device)
#     pred = model(x)
#     predicted, actual = classes[pred[0].argmax(0)], classes[y]
#     print(f'Predicted: "{predicted}", Actual: "{actual}"')

# end = time.time()

# print(f"spend time : {end - start:.5f}sec")