#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import time
import random
import matplotlib.pyplot as plt
import math
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence,pack_sequence,pad_packed_sequence
from data_processing import get_data, get_sources_targets, get_source_target, normalize, get_speed


# In[2]:


class Encoder(nn.Module):
    def __init__(self,
                 input_size = 2,
                 embedding_size = 128,
                 hidden_size = 256,
                 n_layers = 4,
                 dropout = 0.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.linear = nn.Linear(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, n_layers,
                           dropout = dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: input batch data, size: [sequence len, batch size, feature size]
        for the argoverse trajectory data, size(x) is [20, batch size, 2]
        """
        # embedded: [sequence len, batch size, embedding size]
        embedded = self.dropout(F.relu(self.linear(x)))
        # you can checkout https://pytorch.org/docs/stable/nn.html?highlight=lstm#torch.nn.LSTM
        # for details of the return tensor
        # briefly speaking, output coontains the output of last layer for each time step
        # hidden and cell contains the last time step hidden and cell state of each layer
        # we only use hidden and cell as context to feed into decoder
        output, (hidden, cell) = self.rnn(embedded)
        # hidden = [n layers * n directions, batch size, hidden size]
        # cell = [n layers * n directions, batch size, hidden size]
        # the n direction is 1 since we are not using bidirectional RNNs
        return hidden, cell


# In[3]:


class Decoder(nn.Module):
    def __init__(self,
                 output_size = 2,
                 embedding_size = 128,
                 hidden_size = 256,
                 n_layers = 4,
                 dropout = 0.5):
        super().__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Linear(output_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, n_layers, dropout = dropout)
        self.linear = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden, cell):
        """
        x : input batch data, size(x): [batch size, feature size]
        notice x only has two dimensions since the input is batchs
        of last coordinate of observed trajectory
        so the sequence length has been removed.
        """
        # add sequence dimension to x, to allow use of nn.LSTM
        # after this, size(x) will be [1, batch size, feature size]
        x = x.unsqueeze(0)

        # embedded = [1, batch size, embedding size]
        embedded = self.dropout(F.relu(self.embedding(x)))

        #output = [seq len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #seq len and n directions will always be 1 in the decoder, therefore:
        #output = [1, batch size, hidden size]
        #hidden = [n layers, batch size, hidden size]
        #cell = [n layers, batch size, hidden size]
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))

        # prediction = [batch size, output size]
        prediction = self.linear(output.squeeze(0))

        return prediction, hidden, cell


# In[45]:


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hidden_size == decoder.hidden_size,             "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers,             "Encoder and decoder must have equal number of layers!"

    def forward(self, x, y, teacher_forcing_ratio = 1.0):
        """
        x = [observed sequence len, batch size, feature size]
        y = [target sequence len, batch size, feature size]
        for our argoverse motion forecasting dataset
        observed sequence len is 20, target sequence len is 30
        feature size for now is just 2 (x and y)

        teacher_forcing_ratio is probability of using teacher forcing
        e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        """
        batch_size = x.shape[1]
        target_len = y.shape[0]
        
        # tensor to store decoder outputs of each time step
        outputs = torch.zeros(y.shape).to(self.device)
        
        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(x)

        # first input to decoder is last coordinates of x
        decoder_input = x[-1, :, :]
        
        for i in range(target_len):
            # run decode for one time step
            output, hidden, cell = self.decoder(decoder_input, hidden, cell)

            # place predictions in a tensor holding predictions for each time step
            outputs[i] = output

            # decide if we are going to use teacher forcing or not
            teacher_forcing = random.random() < teacher_forcing_ratio

            # output is the same shape as input, [batch_size, feature size]
            # so we can use output directly as input or use true lable depending on
            # teacher_forcing is true or not
            decoder_input = y[i] if teacher_forcing else output

        return outputs


# In[5]:


# produce dataset
class MyDataset(Dataset):
    def __init__(self,data):
        # --------------------------------------------
        # Initialize paths, transforms, and so on
        # --------------------------------------------
        super().__init__()
        self.data = data
        

    def __getitem__(self, index):
        # --------------------------------------------
        # 1. Read from file (using numpy.fromfile, PIL.Image.open)
        # 2. Preprocess the data (torchvision.Transform).
        # 3. Return the data (e.g. image and label)
        # --------------------------------------------
        
        return self.data[index]
        
    def __len__(self):
        # --------------------------------------------
        # Indicate the total size of the dataset
        # --------------------------------------------
        return len(self.data)

class WrappedDataLoader:
    def __init__(self, dataloader, func):
        self.dataloader = dataloader
        self.func = func
        
    def __len__(self):
        return len(self.dataloader)
    
    def __iter__(self):
        iter_dataloader = iter(self.dataloader)
        for batch in iter_dataloader:
            yield self.func(*batch)
            
def preprocess(x, y):
    # x and y is [batch size, seq len, feature size]
    # to make them work with default assumption of LSTM,
    # here we transpose the first and second dimension
    # return size = [seq len, batch size, feature size]
    return x.transpose(0, 1), y.transpose(0, 1)


# In[6]:


path = 'D:/VRE_project/nba-movement-data/data/csv/0021500001.csv'
df = pd.read_csv(path)
game_data, flag_index = get_data(df)
print(game_data[0], flag_index[:5])


# In[7]:


source_ordinal = 0
num_source = 1
target_ordinal = 0
num_target = 1
sources, targets = get_sources_targets(game_data, flag_index)
source, target = get_source_target(sources, targets, source_ordinal,num_source,target_ordinal,num_target)


# In[8]:


print(source[0].shape)


# In[9]:


# source, target = normalize(source, target)
source, target = get_speed(source, target)

if (num_source == 1):
    for i in range(len(source)):
        source[i] = source[i].squeeze()
if (num_target ==1):
    for i in range(len(target)): 
        target[i] = target[i].squeeze() 


# In[10]:


reduce_source =[]
for s in source:
    if len(s)>=100:
        reduce_source.append(s[-100:])
reduce_target = []
for t in target:
    if len(t)>=100:
        reduce_target.append(t[:100])
source =reduce_source
target =reduce_target


# In[11]:


total = []
for i in range(len(source)):
    temp = (source[i], target[i])
    total.append(temp)


# In[12]:


total_data = MyDataset(total)
train_size = int(len(total_data) * 0.7)
val_size = int(len(total_data)*0.2)
test_size = len(total_data) - train_size - val_size
train_data,val_data, test_data =random_split(total_data, [train_size, val_size, test_size])
BATCH_SIZE = 20
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last = True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, drop_last = True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, drop_last = True)
train_loader = WrappedDataLoader(train_loader, preprocess)
val_loader = WrappedDataLoader(val_loader, preprocess)
test_loader = WrappedDataLoader(test_loader, preprocess)


# In[19]:


INPUT_DIM = 4
OUTPUT_DIM = 4
ENC_EMB_DIM = 128
DEC_EMB_DIM = 128
HID_DIM = 256
N_LAYERS = 4
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Seq2Seq(enc, dec, dev).to(dev)


# In[20]:


def train(model, dataloader, optimizer, criterion):
    model.train()
    epoch_loss = 0
    for i, (x, y) in enumerate(dataloader):
        # put data into GPU
        
        x = x.to(dev)
        y = y.to(dev)
        
        # zero all param gradients
        optimizer.zero_grad()
        
        # run seq2seq to get predictions
        y_pred = model(x, y)
#         plt.plot(y_pred[:,0], y_pred[:, 1], color = 'tab:greem')

        
        # get loss and compute model trainable params gradients though backpropagation
        loss = criterion(y_pred, y)
        loss.backward()
        
        # update model params
        optimizer.step()
        
        # add batch loss, since loss is single item tensor
        # we can get its value by loss.item()
        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)

def evaluate(model, dataloader, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            x = x.to(dev)
            y = y.to(dev)
            
            # turn off teacher forcing
            y_pred = model(x, y, teacher_forcing_ratio = 0)
            loss = criterion(y_pred, y)
            epoch_loss += loss.item()
    return epoch_loss / len(dataloader)


# In[21]:


N_EPOCHES = 3
best_val_loss = float('inf')
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


# In[43]:


# load previous best model params if exists
model_dir = "saved_models/Seq2Seq"
saved_model_path = model_dir + "/best_1_1.pt"
if os.path.isfile(saved_model_path):
    model.load_state_dict(torch.load(saved_model_path))
    print("successfully load previous best model parameters")
    


# In[44]:


get_ipython().run_line_magic('load_ext', 'tensorboard')
writer = SummaryWriter()


# In[24]:


for epoch in range(N_EPOCHES):
    
    start_time = time.time()
    train_loss = train(model, train_loader, optimizer, criterion)
    val_loss = evaluate(model, val_loader, criterion)
    end_time = time.time()
    secs = end_time - start_time
    
    print(F'Epoch: {epoch+1:02} | Time:  {secs}s')
    print(F'\tTrain Loss: {train_loss:.3f}')
    print(F'\t Val Loss: {val_loss:.3f}')
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss,epoch)
   
   
   

    if val_loss < best_val_loss:
        os.makedirs(model_dir, exist_ok=True)
        torch.save(model.state_dict(), saved_model_path)
writer.close()


# In[25]:


test_loss = evaluate(model, test_loader, criterion)
print(f'| Test Loss: {test_loss:.3f}')
#  | Test PPL: {math.exp(round(test_loss, 3)):7.3f} |')


# In[34]:


print(iter(test_loader).__next__()[0][2][0])
print(iter(test_loader).__next__()[0].shape)


# In[42]:


scenes = 3
fig, axs = plt.subplots(scenes,sharex=True, sharey=True )
fig.suptitle('The position of the ball')
t = iter(test_loader)
p, q = next(t)
r = model(p, q).detach()
for i in range(scenes):
    
    
    p_ = p.permute(1, 0, 2)[i]
    q_ = q.permute(1, 0, 2)[i]
    r_= r.permute(1, 0, 2)[i]
    axs[i].plot(p_[:, 0], p_[:, 1], color = "red")
    axs[i].plot(q_[:, 0], q_[:, 1], color ="blue")
    axs[i].plot(r_[:, 0], r_[:, 1], color ="green")
   


# In[ ]:


# !tensorboard --logdir=runs


# In[ ]:




