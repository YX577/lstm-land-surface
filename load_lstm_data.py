#!/discover/nobackup/jframe/anaconda3/bin/python
import pandas as pd
import numpy as np
import pickle as pkl

def load_lstm_data(data_dir, site, year):
    s = str(site)
    y = str(year)
    siteyear = s+'_'+y
    obs_f_path = data_dir+'obs_'+s+'.txt' 
    forcing_f_path = data_dir+'forcing_'+s+'.txt' 
    with open(obs_f_path, 'r') as f:
            obs = pd.read_csv(f, header=None, delimiter=r"\s+")
    with open(forcing_f_path, 'r') as f:
            forcing = pd.read_csv(f, header=None, delimiter=r"\s+")
    data_dict = {'obs':obs, 'forcing':forcing}
    return data_dict
