# -*- coding: utf-8 -*-
"""
Some tools to process signals: filtering, differentiating, etc.
    
Created on Thu Mar 18 08:06:14 2021

@author: opsomerl
"""
import numpy as np
from scipy import signal


def filter_signal(y, axis=1, fs=200, fc=10, N=4, type='low'):
    """Filters signal y by using a Butterworth filter of order N and a cut-off 
    frequency fc."""

    # Converts the cut-off frequency to [pi rad/s]
    Wn = fc / (fs / 2)

    # Create butterworth digital filter
    b, a = signal.butter(N, Wn, btype=type, analog=False)

    # Filter y with a zero-phase forward and reverse digital IIR
    ys = signal.filtfilt(b, a, y, axis=axis)

    return ys


def derive(sig, freq, axis=0):
    """Computes the derivative of the input signal.
    
     Syntax: out = derive(sig,freq)      
    
     Inputs:
       sig           input signal (1- or 2-d numpy array)
       freq          sampling frequency
       axis          axis along which the derivative is computed 
                     (default value is 0)
    
     Outputs:
       out           derivative of input signal"""

    # Check dimensions
    dims = np.shape(sig)
    nax = len(dims)
    if nax > 2:
        raise ValueError("Arrays with more than 2 dimensions are not allowed")
    if nax <= axis:
        raise ValueError("Axis is outside of signal dimensions")

    # Q vector
    qs = round(0.01 * freq)
    denom = 2 / freq * sum(np.square((np.linspace(1, qs, num=qs))))
    Q = np.linspace(-qs, qs, num=(2 * qs + 1))

    # Initialize output
    if axis == 0:
        sig = sig.transpose()

    out = np.zeros_like(sig)

    # Compute derivatives
    for i in range(dims[0]):
        out[i] = -np.convolve(sig[i], Q, 'same') / denom

    out[:, :qs] = np.nan
    out[:, -qs:] = np.nan

    if axis == 0:
        out = out.transpose()

    return out
