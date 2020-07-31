#!/discover/nobackup/jframe/anaconda3/bin/python
from load_lstm_data import load_lstm_data
import glob
import matplotlib
import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Initialize stuff
torch.manual_seed(1)
hidden_state_size = 3

#import the test and training data as pandas dataframes for now.
data_dir = '/discover/nobackup/jframe/data/pals/full_data/'
site = 2
year_test = 2005
year_val = 2006

df = load_lstm_data(data_dir, site, year_test, year_val)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
datas = ["forcing_train", "forcing_val", "forcing_test", "obs_train","obs_val","obs_test"] 
train_forc = scaler.fit_transform(df["forcing_train"])
input_feature_size = train_forc.shape[1]
val_forc = scaler.transform(df["forcing_val"])
test_forc = scaler.transform(df["forcing_test"])
train_obs = scaler.fit_transform(df["obs_train"])
val_obs = scaler.transform(df["obs_val"])
test_obs = scaler.transform(df["obs_test"])


lstm = nn.LSTM(input_feature_size, hidden_state_size)
#inputs = [torch.tensor(list(train_forc[0:hidden_state_size,i])) for i in range(train_forc.shape[1])]
inputs = [torch.tensor(list(np.array(df["forcing_train"])[i,:])) for i in range(train_forc.shape[0])]

# initialize the hidden state.
hidden = (torch.randn(1, 1, hidden_state_size),
          torch.randn(1, 1, hidden_state_size))

#print('doing the sequance in a loop.')
#
#for i in inputs:
#    # Step through the sequence one element at a time.
#    # after each step, hidden contains the hidden state.
#    out, hidden = lstm(i.view(1, 1, -1), hidden)
#
#print('out')
#print(out)
#print('hidden')
#print(hidden)

#print('doing the entire sequence all at once.')

# alternatively, we can do the entire sequence all at once.
# the first value returned by LSTM is all of the hidden states throughout
# the sequence. the second is just the most recent hidden state
# (compare the last slice of "out" with "hidden" below, they are the same)
# The reason for this is that:
# "out" will give you access to all hidden states in the sequence
# "hidden" will allow you to continue the sequence and backpropagate,
# by passing it as an argument  to the lstm at a later time
# Add the extra 2nd dimension
inputs = torch.cat(inputs).view(len(inputs), 1, -1)
hidden = (torch.randn(1, 1, hidden_state_size), torch.randn(1, 1, hidden_state_size))  # clean out hidden state
out, hidden = lstm(inputs, hidden)
print('out')
print(out)
print('hidden')
print(hidden)



