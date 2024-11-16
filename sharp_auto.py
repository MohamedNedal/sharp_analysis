#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.filterwarnings('ignore')

import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pylab as plt
import matplotlib.dates as mdates
from datetime import datetime as dt_obj, timedelta
from scipy.stats import pearsonr, spearmanr

import drms
c = drms.Client()

dt_window = 12                   # ± dt time in hours
data_type = 'hmi.sharp_cea_720s'
si = c.info(data_type)           # Set a series

import pandas as pd
pd.set_option('display.max_rows', None)

plt.rcParams['figure.facecolor']  = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['figure.dpi']  = 100
plt.rcParams['savefig.dpi'] = 300

# ==============================================================

# import the final table of geomagnetic storms
path     = './data/csv'
filename = 'GSs_50nT.csv'
# filename = 'M-class-flares_SC24.csv'
# filename = 'X-class-flares_SC24.csv'
df_table = pd.read_csv(f'{path}/{filename}')

# ==============================================================

if filename == 'GSs_50nT.csv':
    def create_datetime(row):
        """
        Make a datetime column from a row in a dataframe.
        """
        yyyy = int(row['Flare_Year'])
        mm = int(row['Flare_Month'])
        dd = int(row['Flare_Day'])
        flare_onset = row['Flare_onset']
        time = dt_obj.strptime(flare_onset, '%H:%M')
        return dt_obj(yyyy, mm, dd, time.hour, time.minute)

else:
    def create_datetime(row):
        """
        Make a datetime column from a row in a dataframe.
        """
        yyyy = int(row['Year'])
        mm = int(row['Month'])
        dd = int(row['Day'])
        flare_onset = row['Flare_onset']
        time = dt_obj.strptime(flare_onset, '%H:%M')
        return dt_obj(yyyy, mm, dd, time.hour, time.minute)


def parse_tai_string(time_str, datetime_obj=True):
    """
    Convert the keyword `T_REC` into a datetime object.
    """
    year   = int(time_str[:4])
    month  = int(time_str[5:7])
    day    = int(time_str[8:10])
    hour   = int(time_str[11:13])
    minute = int(time_str[14:16])
    if datetime_obj:
        return dt_obj(year, month, day, hour, minute)
    else:
        return year, month, day, hour, minute

# ==============================================================

# add a datetime column for the flare onset time
df_table['datetime_flare_onset'] = df_table.apply(create_datetime, axis=1)


for event_index in range(len(df_table)):

    flare_onset_datetime = df_table['datetime_flare_onset'][event_index]
    end_year, end_month, end_day  = str(flare_onset_datetime.date()).split('-')
    end_hour, end_minute, end_sec = str(flare_onset_datetime.time()).split(':')

    start_datetime = flare_onset_datetime - timedelta(hours=dt_window)
    start_year, start_month, start_day  = str(start_datetime.date()).split('-')
    start_hour, start_minute, start_sec = str(start_datetime.time()).split(':')

    if filename == 'GSs_50nT.csv':
        noaa_ar =  df_table['AR_number'][event_index]   # NOAA AR No. (i.e., 11092)
    else:
        noaa_ar =  df_table['AR'][event_index]

    print(f'Doing event {event_index} ..\n')
    print(f'Start datetime: {start_datetime}')
    print(f'Start year: {start_year}')
    print(f'Start month: {start_month}')
    print(f'Start day: {start_day}')
    print(f'Start hour: {start_hour}')
    print(f'Start minute: {start_minute}\n')

    print(f'End datetime: {flare_onset_datetime}')
    print(f'End year: {end_year}')
    print(f'End month: {end_month}')
    print(f'End day: {end_day}')
    print(f'End hour: {end_hour}')
    print(f'End minute: {end_minute}\n')

    print(f'NOAA AR No.: {noaa_ar}')
    print(f'SHARP dataset: {data_type}\n')
    print('==============================================================')









