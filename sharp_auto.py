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

raise SystemExit('Stop mark reached!')

# ==============================================================

sharp_starttime = pd.to_datetime(f'{start_year}.{start_month}.{start_day} {start_hour}:{start_minute}')

# add a time offset to get full SHARP timeseries data that covers the flare end time
sharp_endtime   = pd.to_datetime(f'{end_year}.{end_month}.{end_day} {end_hour}:{end_minute}') + timedelta(hours=2)

print(f'Data type: {data_type}')
print(f'SHARP start time: {sharp_starttime}')
print(f'SHARP end time: {sharp_endtime}')
print(f'AR No.: {noaa_ar}')

keys = c.query(f'{data_type}[][{sharp_starttime} - {sharp_endtime}][? NOAA_ARS ~ "{noaa_ar}" ?]',
                      key='T_REC, HARPNUM, NOAA_ARS, USFLUX, MEANGAM, MEANGBT, MEANGBZ, MEANGBH, MEANJZD, TOTUSJZ,\
                        MEANALP, MEANJZH, TOTUSJH, ABSNJZH, SAVNCPP, MEANPOT, TOTPOT, MEANSHR, R_VALUE')
if len(keys) == 0:
    print('No SHARP data found!')
else:
    print(f'SHARP data found! --> length: {len(keys)}')

t_rec = np.array([parse_tai_string(keys.T_REC[i], datetime_obj=True) for i in range(keys.T_REC.size)])


# Plot the choosen SHARP parameters in a for loop
chosen_parameters = {
    'USFLUX': 'USFLUX\n[Mx]',
    'MEANGAM': 'MEANGAM\n[degrees]',
    'MEANGBT': 'MEANGBT\n[G/Mm]',
    'MEANALP': 'MEANALP\n[1/Mm]',
    'MEANPOT': 'MEANPOT\n[Ergs/cm$^3$]',
    'TOTPOT': 'TOTPOT\n[Ergs/cm$^3$]',
    'MEANSHR': 'MEANSHR\n[degrees]',
    'R_VALUE': 'R_VALUE\n[Mx]',
    'MEANGBZ': 'MEANGBZ\n[G/Mm]',
    'MEANGBH': 'MEANGBH\n[G/Mm]',
    'MEANJZD': 'MEANJZD\n[mA/m$^2$]',
    'TOTUSJZ': 'TOTUSJZ\n[A]',
    'MEANJZH': 'MEANJZH\n[G$^2$/m]',
    'TOTUSJH': 'TOTUSJH\n[G$^2$/m]',
    'ABSNJZH': 'ABSNJZH\n[G$^2$/m]',
    'SAVNCPP': 'SAVNCPP\n[A]'
}

flare_peak_moment = pd.to_datetime(f"{flare_onset_datetime.date()} {df_table['Flare_peak'][event_index]}")
flare_end_moment  = pd.to_datetime(f"{flare_onset_datetime.date()} {df_table['Flare_end'][event_index]}")

if flare_peak_moment < flare_onset_datetime:
    flare_peak_moment += timedelta(days=1)

if flare_end_moment < flare_peak_moment or flare_end_moment < flare_onset_datetime:
    flare_end_moment += timedelta(days=1)

print(f'Flare onset: {flare_onset_datetime}')
print(f'Flare peak: {flare_peak_moment}')
print(f'Flare end: {flare_end_moment}')

dt_rise = (flare_peak_moment - flare_onset_datetime).total_seconds()/60
dt_dec  = (flare_end_moment - flare_peak_moment).total_seconds()/60

print(f'\nFlare rise time: {dt_rise} min.')
print(f'Flare decend time: {dt_dec} min.')

fig = plt.figure(figsize=[15,30])

for i, (col, unit) in enumerate(chosen_parameters.items()):
    ax = fig.add_subplot(len(chosen_parameters), 1, i+1)
    ax.plot(t_rec, keys[col], 'o-')
    ax.axvline(x=flare_onset_datetime, color='g', linestyle='--', label='Flare onset')
    ax.axvline(x=flare_peak_moment, color='r', linestyle='--', label='Flare peak')
    ax.axvline(x=flare_end_moment, color='b', linestyle='--', label='Flare end')
    ax.set_ylabel(unit)
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d %H:%M'))
    ax.set_xlim(left=t_rec[0], right=t_rec[-1])
    
    if i == 0:
        ax.legend(loc='best', ncol=3)
        ax.set_xticklabels([])
    elif i != len(chosen_parameters) - 1:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel('Time (UT)')

fig.tight_layout()
fig.savefig(f'./data/pdf/{str(flare_onset_datetime.date())}.pdf', format='pdf', bbox_inches='tight')
fig.savefig(f'./data/png/{str(flare_onset_datetime.date())}.png', format='png', dpi=300, bbox_inches='tight')
plt.show()







