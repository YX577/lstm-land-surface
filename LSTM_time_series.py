#!/discover/nobackup/jframe/anaconda3/bin/python
from load_lstm_data import load_lstm_data
import glob
import matplotlib
import numpy as np
import pandas as pd
import sklearn
#import torch


# Initialize stuff
#torch.manual_seed(1)
hidden_state_size = 4

#import the test and training data as pandas dataframes for now.
data_dir = '/discover/nobackup/jframe/data/pals/full_data/'
site = 2
year = 2006

df = load_lstm_data(data_dir, site, year)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
datas = ["forcing_train", "forcing_val", "forcing_test", "obs_train","obs_val","obs_test"] 
train_forc = scaler.transform(df["forcing_train"])
val_forc = scaler.transform(df["forcing_val"])
test_forc = scaler.transform(df["forcing_test"])
train_obs = scaler.transform(df["obs_train"])
val_obs = scaler.transform(df["obs_val"])
test_obs = scaler.transform(df["obs_test"])

from torch.autograd import Variable
def transform_data(arr1, arr2, seq_len):
    x, y = [], []
    for i in range(len(arr1) - seq_len):
        x_i = arr1[i : i + seq_len]
        y_i = arr2[i + 1 : i + seq_len + 1]
        x.append(x_i)
        y.append(y_i)
    x_arr = np.array(x).reshape(-1, seq_len)
    y_arr = np.array(y).reshape(-1, seq_len)
    x_var = Variable(torch.from_numpy(x_arr).float())
    y_var = Variable(torch.from_numpy(y_arr).float())
    return x_var, y_var

seq_len = 100

x_train, y_train = transform_data(train_arr, seq_len)
x_val, y_val = transform_data(val_arr, seq_len)
x_test, y_test = transform_data(test_arr, seq_len)
