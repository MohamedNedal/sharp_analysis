#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pylab as plt
import matplotlib.dates as mdates
from matplotlib.ticker import AutoMinorLocator
from datetime import datetime as dt_obj, timedelta
from scipy.stats import pearsonr, spearmanr
import drms
c = drms.Client()

# work with confined flares or not
confined  = True
dt_window = 12                   # ± dt time in hours
data_type = 'hmi.sharp_cea_720s'
si = c.info(data_type)           # Set a series

import pandas as pd
pd.set_option('display.max_rows', None)

plt.rcParams['figure.facecolor']  = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
# plt.rcParams['figure.dpi']  = 100
# plt.rcParams['savefig.dpi'] = 300

# List of directories to make
directories = [
    './output/csv/',
    './output/png/'
]
for directory in directories:
    os.makedirs(directory, exist_ok=True)  # Make directories if they don't exist

# ==============================================================

# import the final table of geomagnetic storms
path     = './data/csv'
filename = 'M-class-flares_SC24.csv'
df_table = pd.read_csv(f'{path}/{filename}')

# ==============================================================

def create_datetime(row):
    """
    Make a datetime column from a row in a dataframe.
    """
    yyyy = int(row['year'])
    mm = int(row['month'])
    dd = int(row['day'])
    flare_onset = row['onset']
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

# Extract only the integer part
df_table['AR'] = df_table['AR'].str.extract('(\d+)')
# Remove any non-digit or non-colon characters
df_table['CME time'] = df_table['CME time'].str.replace(r'[^0-9:]', '', regex=True)

# Split the DataFrame based on "Confinement" column
df_table_yes = df_table[df_table['Confinement'] == 'yes']
df_table_no = df_table[df_table['Confinement'] == 'no']

print('\n========================================================')
print('Analyzing SHARP data automatically for a list of flares')
print('========================================================\n')
print(f'Total number of flares: {len(df_table)}')
print(f'Number of confined flares: {len(df_table_yes)}')
print(f'Number of unconfined flares: {len(df_table_no)}')
print(f'Number of confined & unconfined flares: {len(df_table_yes)+len(df_table_no)}')
print(f'Number of uncertain flares: {len(df_table)-(len(df_table_yes)+len(df_table_no))}\n')

if confined:
    df_flares = df_table_yes.copy()
    print('Working with Confined flares ..\n\n')
else:
    df_flares = df_table_no.copy()
    print('Working with Un-confined flares ..\n\n')

# Make empty dfs to store the correlations tables
col_names = ['HARPNUM', 'NOAA_ARS', 'USFLUX', 'MEANGAM', 'MEANGBT', 'MEANGBZ',
            'MEANGBH', 'MEANJZD', 'TOTUSJZ', 'MEANALP', 'MEANJZH', 'TOTUSJH',
            'ABSNJZH', 'SAVNCPP', 'MEANPOT', 'TOTPOT', 'MEANSHR', 'R_VALUE',
            'datetimes']
df_final_0    = pd.DataFrame(columns=col_names)
df_final_rise = pd.DataFrame(columns=col_names)
df_final_dec  = pd.DataFrame(columns=col_names)
df_final_all  = pd.DataFrame(columns=col_names)

event_indices_with_sharp_data    = []
event_indices_without_sharp_data = []

with tqdm(total=len(df_flares), desc=f'Loading events info ...') as pbar:
    for event_index, row in df_flares.iterrows():
        noaa_ar = row['AR']
        flare_onset_datetime = row['datetime_flare_onset']
        end_year, end_month, end_day  = str(flare_onset_datetime.date()).split('-')
        end_hour, end_minute, end_sec = str(flare_onset_datetime.time()).split(':')

        start_datetime = flare_onset_datetime - timedelta(hours=dt_window)
        start_year, start_month, start_day  = str(start_datetime.date()).split('-')
        start_hour, start_minute, start_sec = str(start_datetime.time()).split(':')

        # print(f'Doing event {event_index} ..')
        # print(f'NOAA AR No.: {noaa_ar}')
        # print(f'SHARP dataset: {data_type}\n')

        # print(f'Start datetime: {start_datetime}')
        # print(f'Start year: {start_year}')
        # print(f'Start month: {start_month}')
        # print(f'Start day: {start_day}')
        # print(f'Start hour: {start_hour}')
        # print(f'Start minute: {start_minute}\n')

        # print(f'End datetime: {flare_onset_datetime}')
        # print(f'End year: {end_year}')
        # print(f'End month: {end_month}')
        # print(f'End day: {end_day}')
        # print(f'End hour: {end_hour}')
        # print(f'End minute: {end_minute}\n')
        # print('==============================================================')

        # raise SystemExit('Stop mark reached!')

        # ==============================================================

        sharp_starttime = pd.to_datetime(f'{start_year}.{start_month}.{start_day} {start_hour}:{start_minute}')
        # add a time offset to get full SHARP timeseries data that covers the flare end time
        sharp_endtime   = pd.to_datetime(f'{end_year}.{end_month}.{end_day} {end_hour}:{end_minute}') + timedelta(hours=2)

        keys = c.query(f'{data_type}[][{sharp_starttime} - {sharp_endtime}][? NOAA_ARS ~ "{noaa_ar}" ?]',
                            key='T_REC, HARPNUM, NOAA_ARS, USFLUX, MEANGAM, MEANGBT, MEANGBZ, MEANGBH, MEANJZD, TOTUSJZ,\
                                MEANALP, MEANJZH, TOTUSJH, ABSNJZH, SAVNCPP, MEANPOT, TOTPOT, MEANSHR, R_VALUE')
        if len(keys) == 0:
            event_indices_without_sharp_data.append(event_index)
            pass
        else:
            event_indices_with_sharp_data.append(event_index)

            print('==============================================================')
            print(f'Doing event {event_index} ..')
            print(f'NOAA AR No.: {noaa_ar}')
            print(f'SHARP dataset: {data_type}\n')

            print(f'SHARP start time: {sharp_starttime}')
            print(f'SHARP end time: {sharp_endtime}\n')

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
            print(f'End minute: {end_minute}')
            print('==============================================================')

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

            flare_peak_moment = pd.to_datetime(f"{flare_onset_datetime.date()} {df_table['peak'][event_index]}")
            flare_end_moment  = pd.to_datetime(f"{flare_onset_datetime.date()} {df_table['end'][event_index]}")

            if flare_peak_moment < flare_onset_datetime:
                flare_peak_moment += timedelta(days=1)

            if flare_end_moment < flare_peak_moment or flare_end_moment < flare_onset_datetime:
                flare_end_moment += timedelta(days=1)

            print(f'Flare onset: {flare_onset_datetime}')
            print(f'Flare peak: {flare_peak_moment}')
            print(f'Flare end: {flare_end_moment}\n')

            dt_rise = (flare_peak_moment - flare_onset_datetime).total_seconds()/60
            dt_dec  = (flare_end_moment - flare_peak_moment).total_seconds()/60

            print(f'Flare rise time: {dt_rise} min.')
            print(f'Flare decend time: {dt_dec} min.\n')

            fig = plt.figure(figsize=[15,30])
            for i, (col, unit) in enumerate(chosen_parameters.items()):
                ax = fig.add_subplot(len(chosen_parameters), 1, i+1)
                ax.plot(t_rec, keys[col], 'o-')
                ax.axvline(x=flare_onset_datetime, color='g', linestyle='--', label='Flare onset')
                ax.axvline(x=flare_peak_moment, color='r', linestyle='--', label='Flare peak')
                ax.axvline(x=flare_end_moment, color='b', linestyle='--', label='Flare end')
                ax.set_ylabel(unit)
                ax.xaxis.set_minor_locator(AutoMinorLocator(n=4))
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
            fig.savefig(f'./output/png/{str(flare_onset_datetime.date())}.png', format='png', dpi=100, bbox_inches='tight')
            plt.show()

            # raise SystemExit('Stop mark reached!')

            # Prep the correlation tables
            # === MAKE THESE TABLES AND EXPORT THEM ===
            # corr_matrix_0    : first point (of each SHARP parameter) before the flare onset time.
            # corr_matrix_rise : avg value (of each SHARP parameter) from the flare onset to peak time.
            # corr_matrix_dec  : avg value (of each SHARP parameter) from the flare peak to end time.
            # corr_matrix_all  : avg value (of each SHARP parameter) from the flare onset to end time.

            # set the keys datetime as the index column
            keys.index = t_rec
            try:
                keys.drop('T_REC', axis=1, inplace=True)
            except:
                pass

            # Take only these columns from the given table of events
            df = df_flares.filter(['datetime_flare_onset', 'Dst', 'CME_onset', 'CME_speed',
                                'CME_AW', 'MPA', 'Flare_onset', 'Flare_peak', 'Flare_end', 'C-class', 'lat', 'long',
                                'AR number', 'AR location', 'AR same day', 'AR next day'])
            df.index = df['datetime_flare_onset']
            df.drop('datetime_flare_onset', axis=1, inplace=True)

            ### Get the first point (of each SHARP parameter) before the flare onset time ( `corr_matrix_0` )
            # Get the index position of the flare_onset_datetime

            # Check if the timestamp exists directly in the index
            if flare_onset_datetime in keys.index:
                # Exact match found, get the position
                position = keys.index.get_loc(flare_onset_datetime)
                # Get the previous row if position is greater than 0, to avoid IndexError
                if position > 0:
                    previous_row = keys.iloc[position - 1]
                else:
                    previous_row = None  # Handle if it's the first row
            else:
                # Use get_indexer to find the nearest timestamp before or equal to the target
                position = keys.index.get_indexer([flare_onset_datetime], method='pad')[0]
                # Check if the position is valid (it returns -1 if there’s no preceding match)
                if position != -1:
                    # If a preceding timestamp is found, position will be non-negative
                    if position > 0:
                        previous_row = keys.iloc[position]
                    else:
                        previous_row = None  # It's the first row
                else:
                    previous_row = None  # No timestamp before or equal to flare_onset_datetime

            # Combine the two rows from the two Dataframes
            series1        = df.loc[flare_onset_datetime]
            series2_0      = previous_row
            combined_row_0 = pd.concat([series1, series2_0])

            # Add the combined row to the final DataFrame
            df_final_0.loc[index] = combined_row_0
            # export the dataframe
            df_final_0.to_csv('./output/csv/df_final_0.csv')

            # ------------------------------------------------------------------

            ### Get the average value (of each SHARP parameter) from the flare onset to peak time ( `corr_matrix_rise` )
            series2_rise = keys.loc[flare_onset_datetime:flare_peak_moment].mean(skipna=True)

            # Combine the two rows from the two Dataframes
            combined_row_rise = pd.concat([series1, series2_rise])
            # Add the combined row to the final DataFrame
            df_final_rise.loc[index] = combined_row_rise
            # export the dataframe
            df_final_rise.to_csv('./output/csv/df_final_rise.csv')

            # ------------------------------------------------------------------

            ### Get the average value (of each SHARP parameter) from the flare peak to end time ( `corr_matrix_dec` )
            series2_dec = keys.loc[flare_peak_moment:flare_end_moment].mean(skipna=True)

            # Combine the two rows from the two Dataframes
            combined_row_dec = pd.concat([series1, series2_dec])
            # Add the combined row to the final DataFrame
            df_final_dec.loc[index] = combined_row_dec
            # export the dataframe
            df_final_dec.to_csv('./output/csv/df_final_dec.csv')

            # ------------------------------------------------------------------

            ### Get the average value (of each SHARP parameter) from the flare onset to end time ( `corr_matrix_all` )
            series2_all = keys.loc[flare_onset_datetime:flare_end_moment].mean(skipna=True)

            # Combine the two rows from the two Dataframes
            combined_row_all = pd.concat([series1, series2_all])
            # Add the combined row to the final DataFrame
            df_final_all.loc[index] = combined_row_all
            # export the dataframe
            df_final_all.to_csv('./output/csv/df_final_all.csv')

            # ==================================================================

            pbar.update(1)





raise SystemExit('Stop mark reached!')


### Plot the correlations

# Drop unnecessary columns
df_final_0.drop(['NOAA_ARS','lat','long','AR number','AR location','AR same day',
        'AR next day','datetimes','CME_onset','Flare_peak','Flare_end'], axis=1, inplace=True)
df_final_rise.drop(['NOAA_ARS','lat','long','AR number','AR location','AR same day',
        'AR next day','datetimes','CME_onset','Flare_peak','Flare_end'], axis=1, inplace=True)
df_final_dec.drop(['NOAA_ARS','lat','long','AR number','AR location','AR same day',
        'AR next day','datetimes','CME_onset','Flare_peak','Flare_end'], axis=1, inplace=True)
df_final_all.drop(['NOAA_ARS','lat','long','AR number','AR location','AR same day',
        'AR next day','datetimes','CME_onset','Flare_peak','Flare_end'], axis=1, inplace=True)




# Exporting events indices to a binary file
np.save('./output/events_with_sharp_data.npy', np.array(event_indices_with_sharp_data))
np.save('./output/events_without_sharp_data.npy', np.array(event_indices_without_sharp_data))
print('Arrays of indices exported')

# # Loading the binary file back
# loaded_data = np.load('./test.npy')
# print('Loaded data:', loaded_data)

