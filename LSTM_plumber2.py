#!/discover/nobackup/jframe/anaconda3/bin/python
from load_lstm_data import load_plumber_data
import glob
import math
import matplotlib.pyplot as plt
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
hidden_state_size = 37
num_epochs = 5
num_features = 1
batch_size = 64
learning_rate = 0.001

#import the test and training data as pandas dataframes for now.
data_dir = '/discover/nobackup/jframe/data/plumber-2-flux-txt/'
data_dir = '/discover/nobackup/jframe/data/plumber-2-met-txt/'
site = 2
year_test = 2005
year_val = 2006

df = load_plumber_data(data_dir, site, year_test, year_val)

scaler = StandardScaler()
X_train = torch.Tensor(scaler.fit_transform(df["forcing_train"]))
input_feature_size = X_train.shape[1]
X_val = torch.Tensor(scaler.transform(df["forcing_val"]))
X_test = torch.Tensor(scaler.transform(df["forcing_test"]))
y_train = torch.Tensor(scaler.fit_transform(df["obs_train"]))
y_val = torch.Tensor(scaler.transform(df["obs_val"]))
y_test = torch.Tensor(scaler.transform(df["obs_test"]))

class LSTM_jf(nn.Module):
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

        ###########################################################################################
        # Define fully connected layer from input size and output size
        # The output size will be the number of features to predict. more than 1 for multi-output.
        # Regression layer, predict output as linear reduction of hidden states.
        self.fc = nn.Linear(hidden_size, num_features) # N x hidden_size x features  

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, init_states=None):
        """
        assumes x.shape represents (batch_size, sequence_size, input_size
        """
        batch_size, sequence_size, _ = x.size()
        hidden_sequence = []

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
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)

            hidden_sequence.append(h_t.unsqueeze(0))

        # reshape hidden sequence p/ retornar
        hidden_sequence = torch.cat(hidden_sequence, dim=0)
        hidden_sequence = hidden_sequence.transpose(0,1).contiguous()

        # Get sequence from fully conected layer.
        prediction = self.fc(hidden_sequence)

        return prediction, (h_t, c_t)

model = LSTM_jf(input_feature_size, hidden_state_size).to(device)

# set up the training data 
ds_train = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True)
ds_test = torch.utils.data.TensorDataset(X_test, y_test)
test_loader = torch.utils.data.DataLoader(ds_test, batch_size=batch_size, shuffle=True)
ds_val = torch.utils.data.TensorDataset(X_val, y_val)
val_loader = torch.utils.data.DataLoader(ds_val, batch_size=batch_size, shuffle=True)

# Loss and optimizer
criterion = nn.MSELoss()  # Use costome loss function. 
# TODO:  Use NSE rather than MSE, precalculate and cary through the varience

# Set the optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Now run the model. Run it once for fun on the whole training set.
y_pred_test, states = model(X_test.unsqueeze(1))
loss = criterion(y_pred_test.squeeze(1),y_test)
optimizer.zero_grad()
loss.backward()
loss=loss.cpu().item()
RMSE=loss**(1/2)
print("Initial RMSE before batch training: {:.2f}".format(RMSE))
# Loop and run the batches.
epoch_bar = tqdm(range(num_epochs),desc="Training", position=0, total=2)
for epoch in epoch_bar:
    batch_bar = tqdm(enumerate(train_loader),
                     desc="Epoch: {}".format(str(epoch)),
                     position=1,
                     total=len(train_loader))

    for i, (data, targets) in batch_bar:
        optimizer.zero_grad()

        data = data.to(device=device).squeeze(1)
        targets = targets.to(device=device).unsqueeze(1)

        # Forward
        y_pred, states = model(data.unsqueeze(1)) 
        loss = criterion(y_pred,targets)

        #backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

        batch_bar.set_postfix(loss=loss.cpu().item(),
                              RMSE="{:.2f}".format(loss**(1/2)),
                              epoch=epoch)
        batch_bar.update()

    with torch.no_grad():
        rmse_list = []
        for i, (data_, targets_) in enumerate(test_loader):
            data_ = data_.to(device=device).squeeze(1)
            targets_ = targets_.to(device=device).unsqueeze(1)
            y_pred_, states_ = model(data_.unsqueeze(1))
            MSE_ = criterion(y_pred_,targets_)
            rmse_list.append(MSE_**(1/2))
    epoch_bar.set_postfix(loss=loss.cpu().item(),
                          RMSE="{:.2f}".format(np.mean(np.array(rmse_list))),
                          epoch=epoch)
    batch_bar.update()

y_pred_val, states = model(X_val.unsqueeze(1))
y_pred_val = y_pred_val.squeeze(1)
y_plot_val = (y_pred_val.cpu().detach().numpy() * np.mean(np.array(df["obs_val"]))) + np.mean(np.array(df["obs_val"]))
plt.plot(y_plot_val, label="LSTM prediction")
plt.plot(np.array(df["obs_val"]), label="observation")
plt.legend()
plt.show()
