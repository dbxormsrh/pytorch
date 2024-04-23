import pandas as pd
import hgtk
from tqdm import tqdm
import fasttext
from torch.utils.data import random_split, DataLoader, Dataset, TensorDataset
from torch import nn
import torch.nn.functional as F
import torch
import numpy as np
import matplotlib.pyplot as plt

def decompose(forms:list):
    word = ''
    for form in forms:
        try:
            if hgtk.checker.is_hangul(form):
                for s in form:
                    a, b, c = hgtk.letter.decompose(s)
                    if not a:
                        a = '-'
                    if not b:
                        b = '-'
                    if not c:
                        c = '-'
                    word = word + a + b + c
        except TypeError as e:
            print(form)
    return word

def transform(word_sequence):
    return [(compose(word), similarity) for (similarity, word) in word_sequence]


class CustomDataset(Dataset):
    def __init__(self, csv_dir, num_word, transform = None, target_transform=None):
        self.df = pd.read_csv(csv_dir).sample(frac=1)[:5120]
        self.transform = transform
        self.target_transform = target_transform
        self.num_word = num_word
            
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, i):
        sent = self.df['morphologized_sent'].iloc[i]
        label = self.df['label'].iloc[i]
        label = torch.tensor(label, dtype=torch.float32)
        padded_vec = torch.zeros((self.num_word, fast_model.get_dimension()), dtype = torch.float32)
        
        sent2vec = [] 
        for w in sent:
            if w.rstrip():
                sent2vec.append(fast_model.get_word_vector(decompose(w)))
        sent2vec = np.array(sent2vec)
        len_sent = len(sent2vec)
        if len_sent > self.num_word:
            len_sent = self.num_word
        padded_vec[(self.num_word - len_sent):] = torch.from_numpy(sent2vec[:len_sent])
        
        return (padded_vec, label)

class SentimentLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_dim):
        super(SentimentLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.output_dim = output_dim
        
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=num_layers,batch_first = True)
        
        self.linear = nn.Linear(self.hidden_size, self.output_dim)
        self.dropout = nn.Dropout(0.3)
        self.sig = nn.Sigmoid()
        
    def forward(self, x, hidden):
        batch_size = x.size(0)
        lstm_out, (hn, cn) = self.lstm(x, hidden)
        
        drop_out = self.dropout(lstm_out)
        re_drop_out = drop_out.reshape([-1, self.hidden_size])
        
        linear_out = self.linear(re_drop_out)
        
        sig_out = self.sig(linear_out)
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1]
        
        return sig_out, (hn, cn)
        
    def init_hidden(self, batch_size, device):
        h0 = torch.zeros((self.num_layers, batch_size, self.hidden_size), dtype=torch.float32).to(device)
        c0 = torch.zeros((self.num_layers, batch_size, self.hidden_size), dtype=torch.float32).to(device)
        
        return (h0, c0)

fast_model = fasttext.load_model("fasttext_jn.bin") # 모델 로드
print('fasttext model loaded')
dataset = CustomDataset('./morphologized_ratings.csv', num_word=32)
print('init dataset')

train_size = int(len(dataset) * 0.8)
valid_size = len(dataset) - train_size
batch_size = 16

train_data, valid_data = random_split(dataset, [train_size, valid_size])

train_dataloader = DataLoader(train_data, batch_size = batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_data, batch_size = batch_size, shuffle=True)

num_layers = 2
input_size = 100
hidden_size = 128
output_dim = 1

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(device)
lstm_model = SentimentLSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_dim=output_dim)

lstm_model.to(device)
print(lstm_model)

lr = 0.001
clip = 5

loss_func = nn.BCELoss().to(device)
optimizer = torch.optim.Adam(lstm_model.parameters(), lr = lr)

def acc(pred, label):
    pred = torch.round(pred.squeeze())
    return torch.sum(pred == label.squeeze()).item()

epochs = 5

epoch_tr_acc, epoch_tr_loss = [], []
epoch_vl_acc, epoch_vl_loss = [],[]

for epoch in range(epochs):
    train_losses = []
    train_acc = 0.0
    lstm_model.train()
    hidden = lstm_model.init_hidden(batch_size, device)

    for inputs, labels in tqdm(train_dataloader):
        print(inputs.size())
        inputs, labels = inputs.to(device), labels.to(device)
        lstm_model.zero_grad()
        pred, hidden= lstm_model(inputs, hidden)
        
        loss = loss_func(pred, labels)
        optimizer.zero_grad()
        loss.backward()
        train_losses.append(loss.item())
        
        accuracy = acc(pred, labels)

        train_acc += accuracy


    epoch_train_loss = np.mean(train_losses)
    epoch_train_acc = train_acc/len(train_dataloader.dataset)
    epoch_tr_loss.append(epoch_train_loss)
    epoch_tr_acc.append(epoch_train_acc)
    
    val_losses = []
    val_acc = 0.0
    lstm_model.eval()
    hidden = lstm_model.init_hidden(batch_size, device)

    for inputs, labels in test_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        pred = lstm_model(inputs, hidden)

        val_loss = loss_func(pred.squeeze(), labels.float())
        val_losses.append(val_loss.item())
        accuracy = acc(pred, labels)

        val_acc += accuracy

    epoch_val_loss = np.mean(val_losses)
    epoch_val_acc = val_acc/len(valid_dataloader.dataset)
    epoch_vl_loss.append(epoch_val_loss)
    epoch_vl_acc.append(epoch_val_acc)

    print(f'Epoch {epoch+1}') 
    print(f'train_loss : {epoch_train_loss} val_loss : {epoch_val_loss}')
    print(f'train_accuracy : {epoch_train_acc*100} val_accuracy : {epoch_val_acc*100}')
    print(25*'==')