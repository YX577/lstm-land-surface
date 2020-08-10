#!/discover/nobackup/jframe/anaconda3/bin/python
import pandas as pd
import numpy as np
import pickle as pkl

def split_test_train_pals_sm(data_dict, year_test, year_val):
    obs_drop = ['year','day','hour', 'x1', 'x2', 'p', 'sm2']
    data_dict['obs_train'] = data_dict['obs'][~data_dict['obs']['year'].isin([year_test,year_val])]
    data_dict['obs_test'] = data_dict['obs'].loc[data_dict['obs']['year'] == year_test]
    data_dict['obs_val'] = data_dict['obs'].loc[data_dict['obs']['year'] == year_val]
    data_dict['obs_train'] = data_dict['obs_train'].drop(obs_drop, axis=1)
    data_dict['obs_test'] = data_dict['obs_test'].drop(obs_drop, axis=1)
    data_dict['obs_val'] = data_dict['obs_val'].drop(obs_drop, axis=1)

    forcing_drop = ['year','day','hour', 'dummy']
    data_dict['forcing_train'] = data_dict['forcing'][~data_dict['forcing']['year'].isin([year_test,year_val])]
    data_dict['forcing_test'] = data_dict['forcing'].loc[data_dict['forcing']['year'] == year_test]
    data_dict['forcing_val'] = data_dict['forcing'].loc[data_dict['forcing']['year'] == year_val]
    data_dict['forcing_train'] = data_dict['forcing_train'].drop(forcing_drop, axis=1)
    data_dict['forcing_test'] = data_dict['forcing_test'].drop(forcing_drop, axis=1)
    data_dict['forcing_val'] = data_dict['forcing_val'].drop(forcing_drop, axis=1)

    return data_dict

def load_pals_data(data_dir, site, year_test, year_val):
    s = str(site)
    obs_f_path = data_dir+'obs_'+s+'.txt' 
    forcing_f_path = data_dir+'forcing_'+s+'.txt' 
    with open(obs_f_path, 'r') as f:
        obs = pd.read_csv(f, header=None, delimiter=r"\s+")
        obs.columns=['year','day','hour', 'x1', 'x2', 'sm', 'sm2', 'p']
    with open(forcing_f_path, 'r') as f:
        forcing = pd.read_csv(f, header=None, delimiter=r"\s+")
        forcing.columns=['year','day','hour', 'wind', 'dummy', 'temp', 'humid', 'press', 'sw', 'lw', 'p']
    data_dict = {'obs':obs, 'forcing':forcing}
    data_dict = split_test_train_pals_sm(data_dict, year_test, year_val)
    return data_dict
