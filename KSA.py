import fasttext
from khaiii import KhaiiiApi
from torch import nn
import torch
import numpy as np
import argparse
import hgtk

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
        
    def forward(self, x):
        batch_size = x.size(0)
        lstm_out, _ = self.lstm(x)
        
        drop_out = self.dropout(lstm_out)
        re_drop_out = drop_out.reshape([-1, self.hidden_size])
            
        linear_out = self.linear(re_drop_out)
        
        sig_out = self.sig(linear_out).reshape([batch_size, -1])[:, -1]
        
        return sig_out

def decompose(form):
    word = ''
    try:
        for s in form:
            if s == ' ':
                word += ''
            elif hgtk.checker.is_hangul(s):
                a, b, c = hgtk.letter.decompose(s)
                if not a:
                    a = '-'
                if not b:
                    b = '-'
                if not c:
                    c = '-'
                word = word + a + b + c
    except e:
        print(e)
        print(f'except: {form}')
    return word

def analyze_sent(sent, fast_model, lstm_model, khaiii, num_word):
    morphs = []
    try:
        for word in khaiii.analyze(sent):
            for m in word.morphs:
                morphs.append(m.lex)
    except:
        print('Can\'t analyze sentence')
        return -1
    
    
    if len(morphs) > num_word:
        morphs = morphs[:num_word]
        
    sent_vec = np.zeros((num_word, fast_model.get_dimension()), dtype=np.float32)
    
    for i, m in zip(range(num_word), morphs):
        word_vec = fast_model.get_word_vector(decompose(m)).astype(np.float32)
        sent_vec[-(i + 1)] = word_vec
    
    sent_tensor = torch.from_numpy(sent_vec)
    sent_tensor = sent_tensor.reshape([1, num_word, fast_model.get_dimension()])

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    
    if device == 'cuda':
        sent_tensor = sent_tensor.to(device)
    
    lstm_model.eval()
    
    pred = lstm_model(sent_tensor)
    
    return pred

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--lstm', dest = 'lstm', required=True)
    parser.add_argument('-f', '--fast', dest = 'fast', required=True)
    parser.add_argument('-n', '--nword', dest = 'nword', required=True)

    args = parser.parse_args()
    
    khaiii_api = KhaiiiApi()
    fast_model = fasttext.load_model(args.fast)
    lstm_model = torch.load(args.lstm)
    num_word = int(args.nword)

    if device == 'cuda':
        print('model to device')
        lstm_model.to(device)

    while True:
        sentence = input('문장 입력. 종료 입력 시 종료.\n')
        if sentence == '종료':
            break
        pred = analyze_sent(sentence, fast_model, lstm_model, khaiii_api, num_word)

        if round(pred.item()) == 1:
            print('긍정')
        else:
            print('부정')
