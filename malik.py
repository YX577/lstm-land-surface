
# FROM : https://stackabuse.com/time-series-prediction-using-lstm-with-pytorch-in-python/
# AND  : https://stackabuse.com/introduction-to-pytorch-for-classification/

import torchi
import torch.nn as nn

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

########## INITIALIZE BOOLEAN OPTIONS ##########
is_plots = False
is_debug = False
is_verbose = False
########## INITIALIZE BOOLEAN OPTIONS ##########

flight_data = sns.load_dataset("flights")

# Let's plot the frequency of the passengers traveling per month.
if is_plots:
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 15
    fig_size[1] = 5
    plt.rcParams["figure.figsize"] = fig_size
    plt.title('Month vs Passenger')
    plt.ylabel('Total Passengers')
    plt.xlabel('Months')
    plt.grid(True)
    plt.autoscale(axis='x',tight=True)
    plt.plot(flight_data['passengers'])


# The first preprocessing step is to change the type of the passengers column to float.
all_data = flight_data['passengers'].values.astype(float)
if is_debug:
  print(all_data)

# Set testing and training data.
test_data_size = 12
train_data = all_data[:-test_data_size]
test_data = all_data[-test_data_size:]

# normalizes our data using the min/max scaler 
# with minimum and maximum values of -1 and 1, respectively
scaler = MinMaxScaler(feature_range=(-1, 1))
train_data_normalized = scaler.fit_transform(train_data .reshape(-1, 1))

# convert our dataset into tensors since PyTorch models are trained using tensors.
train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)

train_window = 12

# accept the raw input data and will return a list of tuples. In each tuple, the first element will
# contain list of 12 items corresponding to the number of passengers traveling in 12 months, the second
# tuple element will contain one item i.e. the number of passengers in the 12+1st month.
def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq ,train_label))
    return inout_seq

# create sequences and corresponding labels for training:
train_inout_seq = create_inout_sequences(train_data_normalized, train_window)


############### CREATE THE LSTM MODEL #######################################
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq),1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq),-1))
        return predictions[-1]


