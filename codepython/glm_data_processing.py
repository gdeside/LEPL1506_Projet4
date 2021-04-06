import pandas as pd
from scipy import signal


def import_data(file_path):
    """Import data from a *.glm file and store it in a data frame"""

    # Columns to import
    ColNames = ['time (s)',
                'Fxal (N)',
                'Fyal (N)',
                'Fzal (N)',
                'Txal (Nm)',
                'Tyal (Nm)',
                'Tzal (Nm)',
                'Fxar (N)',
                'Fyar (N)',
                'Fzar (N)',
                'Txar (Nm)',
                'Tyar (Nm)',
                'Tzar (Nm)',
                'OPxal (m)',
                'OPyal (m)',
                'OPxar (m)',
                'OPyar (m)',
                'OPxgl (m)',
                'OPzgl (m)',
                'OPxgr (m)',
                'OPzgr (m)',
                'Fxgl (N)',
                'Fygl (N)',
                'Fzgl (N)',
                'Txgl (Nm)',
                'Tygl (Nm)',
                'Tzgl (Nm)',
                'Fxgr (N)',
                'Fygr (N)',
                'Fzgr (N)',
                'Txgr (Nm)',
                'Tygr (Nm)',
                'Tzgr (Nm)',
                'Fx (N)',
                'Fy (N)',
                'Fz (N)',
                'Tx (Nm)',
                'Ty (Nm)',
                'Tz (Nm)',
                'GF (N)',
                'LFv (N)',
                'LFh (N)',
                'LFt (N)',
                'LowAcc/X (g)',
                'LowAcc/Y (g)',
                'LowAcc/Z (g)',
                'HighAcc (g)',
                'Humidity_L (V)',
                'Humidity_R (V)',
                'RateGyro/X (V/deg.s-1)',
                'RateGyro/Y (V/deg.s-1)',
                'RateGyro/Z (V/deg.s-1)',
                'EnvAcc/X (g)',
                'EnvAcc/Y (g)',
                'EnvAcc/Z (g)',
                'Metronome_b0',
                'Metronome_b1',
                'Metronome_b2',
                'CODA_clock',
                'LED1',
                'LED2',
                'LED3',
                'LED4',
                'Switch_MEv2']

    # New column names (without unit and front slash)
    NewColNames = ['time',
                   'Fxal',
                   'Fyal',
                   'Fzal',
                   'Txal',
                   'Tyal',
                   'Tzal',
                   'Fxar',
                   'Fyar',
                   'Fzar',
                   'Txar',
                   'Tyar',
                   'Tzar',
                   'OPxal',
                   'OPyal',
                   'OPxar',
                   'OPyar',
                   'OPxgl',
                   'OPzgl',
                   'OPxgr',
                   'OPzgr',
                   'Fxgl',
                   'Fygl',
                   'Fzgl',
                   'Txgl',
                   'Tygl',
                   'Tzgl',
                   'Fxgr',
                   'Fygr',
                   'Fzgr',
                   'Txgr',
                   'Tygr',
                   'Tzgr',
                   'Fx',
                   'Fy',
                   'Fz',
                   'Tx',
                   'Ty',
                   'Tz',
                   'GF',
                   'LFv',
                   'LFh',
                   'LFt',
                   'LowAcc_X',
                   'LowAcc_Y',
                   'LowAcc_Z',
                   'HighAcc',
                   'Humidity_L',
                   'Humidity_R',
                   'RateGyro_X',
                   'RateGyro_Y',
                   'RateGyro_Z',
                   'EnvAcc_X',
                   'EnvAcc_Y',
                   'EnvAcc_Z',
                   'Metronome_b0',
                   'Metronome_b1',
                   'Metronome_b2',
                   'CODA_clock',
                   'LED1',
                   'LED2',
                   'LED3',
                   'LED4',
                   'Switch_MEv2']

    # Load data and store it in a data frame   
    df = pd.read_csv(file_path,
                     sep='\t',
                     header=0,
                     usecols=ColNames)

    # Rename columns
    df.columns = NewColNames

    return df


def filter_signal(y, fs=800, fc=40, N=4, type='low'):
    """Filter signal y by using a Butterworth filter of order N and a cut-off 
    frequency fc"""

    # Converts the cut-off frequency to [pi rad/s]
    Wn = fc / (fs / 2)

    # Create butterworth digital filter
    b, a = signal.butter(N, Wn, btype=type, analog=False)

    # Filter y with a zero-phase forward and reverse digital IIR
    ys = signal.filtfilt(b, a, y)

    return ys
