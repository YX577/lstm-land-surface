#!/discover/nobackup/jframe/anaconda3/bin/python
import pandas as pd
import numpy as np
import pickle as pkl

def site_directories():
    # Directories for the PLUMBER-2 data
    flux_dir = '/discover/nobackup/jframe/data/plumber-2-flux-txt/'
    met_dir = '/discover/nobackup/jframe/data/plumber-2-met-txt/'
    dirz = {'flux':flux_dir, 'met':met_dir}
    return dirz

def split_test_train(data_dict, year_test, year_val, obs_drop=[], forcing_drop=[]):
    data_dict['obs_train'] = data_dict['obs'][~data_dict['obs']['year'].isin([year_test,year_val])]
    data_dict['obs_test'] = data_dict['obs'].loc[data_dict['obs']['year'] == year_test]
    data_dict['obs_val'] = data_dict['obs'].loc[data_dict['obs']['year'] == year_val]
    data_dict['obs_train'] = data_dict['obs_train'].drop(obs_drop, axis=1)
    data_dict['obs_test'] = data_dict['obs_test'].drop(obs_drop, axis=1)
    data_dict['obs_val'] = data_dict['obs_val'].drop(obs_drop, axis=1)

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
    obs_drop = ['year','day','hour', 'x1', 'x2', 'p', 'sm2']
    with open(forcing_f_path, 'r') as f:
        forcing = pd.read_csv(f, header=None, delimiter=r"\s+")
        forcing.columns=['year','day','hour', 'wind', 'dummy', 'temp', 'humid', 'press', 'sw', 'lw', 'p']
    forcing_drop = ['year','day','hour', 'dummy']
    data_dict = {'obs':obs, 'forcing':forcing}
    data_dict = split_test_train(data_dict, year_test, year_val, obs_drop, forcing_drop)
    return data_dict

def load_plumber2_data(site, year_test, year_val):
    s = str(site)
    dirz = site_directories()
    obs_f_path = dirz['flux']+site+'_Flux.txt'
    forcing_f_path = dirz['met']+site+'_Met.txt'
    with open(obs_f_path, 'r') as f:
        obs = pd.read_csv(f, header=0)
        #obs.columns=['datetime', 'year', 'month', 'day', 'hour', 'minute', 'NEE', 'GPP', 'Qle', 'Qh']
    obs_drop = ['year', 'month', 'day', 'hour', 'minute', 'GPP', 'Qle', 'Qh']
    with open(forcing_f_path, 'r') as f:
        forcing = pd.read_csv(f, header=0)
        #forcing.columns=['datetime', 'year', 'month', 'day', 'hour', 'minute', 'Tair', 'SWdown', 'LWdown', 'VPD',
        #                 'Qair', 'Psurf', 'Precip', 'RH', 'CO2air', 'Wind',
        #                 'LAI_alternative','LAI',  'IGBP_veg_long']
    forcing_drop = ['year', 'day', 'minute']
    data_dict = {'obs':obs, 'forcing':forcing}
    data_dict = split_test_train(data_dict, year_test, year_val, obs_drop, forcing_drop)
    return data_dict
