#!/discover/nobackup/jframe/anaconda3/bin/python
from load_lstm_data import load_lstm_data
import glob
import math
import matplotlib
import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import tqdm
from tqdm import tqdm

# Initialize stuff
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)
hidden_state_size = 3
num_epochs = 2
batch_size = 64
learning_rate = 0.001

#import the test and training data as pandas dataframes for now.
data_dir = '/discover/nobackup/jframe/data/pals/full_data/'
site = 2
year_test = 2005
year_val = 2006

df = load_lstm_data(data_dir, site, year_test, year_val)

scaler = StandardScaler()
datas = ["forcing_train", "forcing_val", "forcing_test", "obs_train","obs_val","obs_test"] 
X_train = torch.Tensor(scaler.fit_transform(df["forcing_train"]))
#print(type(X_train))
#exit()
input_feature_size = X_train.shape[1]
#X_train = [torch.tensor(list(np.array(df["forcing_train"])[i,:])) for i in range(X_train.shape[0])]
X_val = scaler.transform(df["forcing_val"])
X_test = scaler.transform(df["forcing_test"])
y_train = torch.Tensor(scaler.fit_transform(df["obs_train"]))
#y_train = [torch.tensor(list(np.array(df["forcing_train"])[i,:])) for i in range(y_train.shape[0])]
y_val = scaler.transform(df["obs_val"])
y_test = scaler.transform(df["obs_test"])

class LSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Input gate (i_t)
        self.U_i = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.V_i = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_i = nn.Parameter(torch.Tensor(hidden_size))

        # Fotget gate (f_t)
        self.U_f = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.V_f = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_f = nn.Parameter(torch.Tensor(hidden_size))

        # candidate (c_t)
        self.U_c = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.V_c = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_c = nn.Parameter(torch.Tensor(hidden_size))

        # Output gate (o_t)
        self.U_o = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.V_o = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(hidden_size))

        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, init_states=None):
        """
        assumes x.shape represents (batch_size, sequence_size, input_size
        """
        batch_size, sequence_size, _ = x.size()
        hidden_seq = []

        if init_states is None:
            h_t, c_t = (torch.zeros(batch_size, self.hidden_size).to(x.device),
                        torch.zeros(batch_size, self.hidden_size).to(x.device))
        else:
            h_t, c_t = init_states

        for t in range(sequence_size):
            x_t = x[:,t,:]

            i_t = torch.sigmoid(x_t @ self.U_i + h_t @ self.V_i + self.b_i)
            f_t = torch.sigmoid(x_t @ self.U_f + h_t @ self.V_f + self.b_f)
            g_t = torch.tanh(x_t @ self.U_c + h_t @ self.V_c + self.b_c)
            o_t = torch.sigmoid(x_t @ self.U_o + h_t @ self.V_o + self.b_o)
            c_t = f_t * c_t + i_t * g*t
            h_t = o_t * torch.tanh(c_t)

            hidden_sequence.append(h_t.unsqueeze(0))

        # reshape hidden sequence p/ retornar
        hidden_sequence = torch.cat(hidden_sequence, dim=0)
        hidden_sequence = hidden_sequence.transpose(0,1).contiguous()
        return hidden_sequence, (h_t, c_t)
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(len(encoder.vocab)+1, 32)
        self.lstm = CustomLSTM(32,32)#nn.LSTM(32, 32, batch_first=True)
        self.fc1 = nn.Linear(32, 2)
                                                
    def forward(self, x):
        x_ = self.embedding(x)
        x_, (h_n, c_n) = self.lstm(x_)
        x_ = (x_[:, -1, :])
        x_ = self.fc1(x_)
        return x_
classifier = Net().to(device)

model = nn.LSTM(input_feature_size, hidden_state_size)

# set up the training data 
ds_train = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

epoch_bar = tqdm(range(num_epochs),desc="Training", position=0, total=2)
acc = 0
for epoch in epoch_bar:
    batch_bar = tqdm(enumerate(train_loader),
                     desc="Epoch: {}".format(str(epoch)),
                     position=1,
                     total=len(train_loader))

    for i, (datapoints, labels) in batch_bar:
        optimizer.zero_grad()
        predictions = classifier(datapoints.long().to(device))
        loss = criterion(predictions, labels.to(device))
        loss.backward()
        optimizer.step()
        if (i+1) % 50 == 0:
            acc=0
            with torch.no_grad():
                for i, (datapoints_, labels_) in enumerate(test_loader):
                    predictions = classifier(datapoints_.to(device))
                    acc += (predictions.argmax(dim=1) == labels_.to(device)).float().sum().cpu().item()
            acc /= len(X_test)
        batch_bar.set_postfix(loss=loss.cpu().item(),
                              accuracy="{:.2f}".format(acc),
                              epoch=epoch)
        batch_bar.update()
    epoch_bar.set_postfix(loss=loss.cpu().item(),
                          accuracy="{:.2f}".format(acc),
                          epoch=epoch)
    batch_bar.update()




# OLD OLD OLD OLD
##inputs = [torch.tensor(list(train_forc[0:hidden_state_size,i])) for i in range(train_forc.shape[1])]
#inputs = [torch.tensor(list(np.array(df["forcing_train"])[i,:])) for i in range(train_forc.shape[0])]
#
## initialize the hidden state.
#hidden = (torch.randn(1, 1, hidden_state_size),
#          torch.randn(1, 1, hidden_state_size))
#
#inputs = torch.cat(inputs).view(len(inputs), 1, -1)
#hidden = (torch.randn(1, 1, hidden_state_size), torch.randn(1, 1, hidden_state_size))  # clean out hidden state
#out, hidden = lstm(inputs, hidden)
#print('out')
#print(out)
#print('hidden')
#print(hidden)



