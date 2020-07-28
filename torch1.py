import pandas as pd
import numpy as np
import pickle as pkl
from load_lstm_data import load_lstm_data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Initialize stuff
torch.manual_seed(1)
hidden_state_size = 4

#import the test and training data as pandas dataframes for now.
data_dir = '/discover/nobackup/jframe/data/pals/full_data/'
site = 2
year = 2006

data_dict = load_lstm_data(data_dir, site, year)

inputShape = data_dict['forcing_train'].shape
inputData = np.transpose(np.array(data_dict['forcing_train']))

lstm = nn.LSTM(hidden_state_size, hidden_state_size)
inputs = [torch.tensor(list(inputData[0:hidden_state_size,i])) for i in range(inputShape[0])]

# initialize the hidden state.
hidden = (torch.randn(1, 1, hidden_state_size),
          torch.randn(1, 1, hidden_state_size))

print('doing the sequance in a loop.')

for i in inputs:
    # Step through the sequence one element at a time.
    # after each step, hidden contains the hidden state.
    out, hidden = lstm(i.view(1, 1, -1), hidden)

print('out')
print(out)
print('hidden')
print(hidden)

print('doing the entire sequence all at once.')

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
