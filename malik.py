
# FROM : https://stackabuse.com/time-series-prediction-using-lstm-with-pytorch-in-python/

import torch
import torch.nn as nn

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

flight_data = sns.load_dataset("flights")

# Let's plot the frequency of the passengers traveling per month. 
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
