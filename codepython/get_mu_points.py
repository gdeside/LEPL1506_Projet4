# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 16:54:37 2020
GET_MU_POINTS returns the static coefficient of friction computed at slip onset
from the ratio TF/NF, during a friction task along the y-axis ("rubbing 
axis"). The method was developed by Barrea et al. (2016).

   [mu,slip_indexes,search_zones,directions,outlier] = get_mu_points(y_COP,tf,NF,...)

 Inputs:
   - y_COP [m]: position of the center of pressure along the "rubbing axis" 
               (N-by-1 vector) 
   - tf_h [N]: horizontal tangential force vector field . 
   - tf_v [N] vertical tangential force vector field (along the rubbing axis)
   - NF [N]: normal force (N-by-1 vector)
   - Parameter/value pairs: 
       - 'fs': Sampling frequency (Hz)
       - 'y_thresh': maximum acceptable displacement of y_COP from the 
         time of TF sign change to slip onset, expressed as a proportion 
         of the total displacement from the time of TF sign change to COP
         extremum). Def. is 0.5
       - 'nf_thresh': minimum acceptable NF
       - 'tf_thresh': minimum acceptable TF

 Outputs:
   - mu: static coefficients of friction computed at each slip point
   - slip_indexes: indices of the points of slip onset
   - iStart: indices delimiting the beginning of the search zones
   - iEnd: indices delimiting the end of the search zones
@author: fschiltz (adapted from Matlab, LO2018)
"""
from os import path

import glm_data_processing as glm
import derive as der
import numpy as np
import math
def get_mu_points(y_COP,tf_h,tf_v,NF,fs=1000,y_thresh=0.5,tf_thresh=0.25,nf_thresh=0.25):
    if len(y_COP)!=len(tf_h) or len(tf_h)!=len(tf_v) or len(tf_v)!=len(NF):
        raise NameError('Vectors must be the same length')
        
    # Filter signals
    fc = 40
    y_COP = glm.filter_signal(y_COP-np.nanmean(y_COP),fs,fc,4)
    dy = der.derive(y_COP,fs)
    tf_h =glm.filter_signal(tf_h,fs,fc)
    tf_v =glm.filter_signal(tf_v,fs,fc)
    NF = glm.filter_signal(NF,fs,fc)    
    
    # Compute the norm of the normal and tangential forces and TF/NF
    NF = abs(NF)
    TF = np.sqrt(np.multiply(tf_h,tf_h)+np.multiply(tf_v,tf_v))
    ratio = TF/NF
    
    # Find first index where NF>NF_thresh
    i0=np.nonzero(NF>nf_thresh)[0][0]
    
    # Find first index where NF>NF_thresh starting from the end
    if NF[-1]>nf_thresh:
        iend=len(NF)-1
    else:
        reversed_NF=NF[::-1]
        iend=np.nonzero(reversed_NF>nf_thresh)[0][0]
        iend=len(NF)-iend
        
    # Find where tangential force changes sign. This marks the beginnings of the
    # useful zones
    iRoots=i0+np.nonzero(np.diff(np.sign(tf_v[i0:iend])))[0]
    iStart = np.insert(iRoots,0,i0) #Add i0 as the beginning of the first usefull zone
    
    # Find the end of the search zones by looking at the COP displacement 
    # (end zone is reached when displacement is greater than y_thresh)
    iEnd = np.zeros(len(iStart),dtype=int)
    out = np.zeros(len(iStart),dtype=bool)
    out[0]=True
    out[-1]=True
    
    for i in range(0,len(iStart)-1):
        y_loc=y_COP[iStart[i]:iStart[i+1]]
        dy_loc=dy[iStart[i]:iStart[i+1]]
        #range(0,np.floor((iStart[i+1]-iStart[i])/2))
        #print('%d' %(np.floor((iStart[i+1]-iStart[i])/2)))
        slope=np.nanmean(dy_loc[0:np.floor((iStart[i+1]-iStart[i])/2).astype(np.int)])
        
        if slope <0:
            extrem=min(y_COP[iStart[i]:iStart[i+1]])
        else:
            extrem=max(y_COP[iStart[i]:iStart[i+1]])
    
        displacement_thresh=y_thresh*abs(extrem - y_loc[1])
        
        tf_loc=abs(np.mean(tf_v[iStart[i]:iStart[i+1]]))
        nf_loc=np.mean(NF[iStart[i]:iStart[i+1]])
        
        if displacement_thresh > 0.002 and tf_loc > tf_thresh and nf_loc > nf_thresh:
            i_end_loc = np.nonzero(abs(y_loc-y_loc[0])>=displacement_thresh)[0][0]
            if i_end_loc==0:
                out[i]=True
            else:
                iEnd[i]=(iStart[i]+i_end_loc-1).astype(np.int)
                
        else:
            out[i]=True
    
    iStart=iStart[np.logical_not(out)]
    iEnd=iEnd[np.logical_not(out)]
    
    check_indexes=iEnd-iStart
    if any(check_indexes<=0):
        raise NameError('Some start indexes are equal to - or bigger than - their corresponding stop indexes')
    
    # Search for slip points as the points where TF/NF is maximal. TF/NF then
    # theoritically corresponds to the static coefficient of friction (mu)
    nz=len(iStart)
    mu=np.zeros(nz)
    slip_indexes=np.zeros(nz,dtype=int)
    directions = np.zeros(nz,dtype=int)
    
    for i in range(0,nz):
        mu_loc=ratio[iStart[i]:iEnd[i]]
        imax=np.argmax(mu_loc)
        
        dy_loc=dy[iStart[i]:iEnd[i]]
        
        slip_indexes[i]=iStart[i]+imax-1
        mu[i]=mu_loc[imax]
        directions[i]=np.sign(np.nanmean(dy_loc))
        # Some things were removes from the Matlab version here
        
    # Remove slip points corresponding to abnormally low values of TF or NF
    discard= (abs(tf_v[slip_indexes])<tf_thresh).astype(bool) | (NF[slip_indexes]<nf_thresh).astype(bool)
    slip_indexes=slip_indexes[np.logical_not(discard)]
    mu=mu[np.logical_not(discard)]
    directions=directions[np.logical_not(discard)]
    
    # Remove outliers (code missing, can be added later if needed)
    
    # Return values
    
    return mu,slip_indexes,iStart,iEnd
    
    
    
    


names=["GD","LH","MH","PDs"]
pos=["pre","post"]
for name in names:
    for k in pos:
        for i in range(1,4):
            glm_path = "../data/CF_data/%s_%s_00%d.glm" % (name, k, i)
            if not path.exists(glm_path):
                continue
            glm_df = glm.import_data(glm_path)
            baseline = range(0, 400)
            # Normal Force exerted by the thumb
            NF_thumb = glm_df.loc[:, 'Fygl'] - np.nanmean(glm_df.loc[baseline, 'Fygl'])
            # Vertical Tangential Force exerted by the thumb
            TFx_thumb = glm_df.loc[:, 'Fxgl'] - np.nanmean(glm_df.loc[baseline, 'Fxgl'])
            # Horizontal Tangential Force exerted by the thumb
            TFz_thumb = glm_df.loc[:, 'Fzgl'] - np.nanmean(glm_df.loc[baseline, 'Fzgl'])
    
    
    
    
    
    