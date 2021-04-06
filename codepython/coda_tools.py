# -*- coding: utf-8 -*-
"""
Some tools to import and process data from the codas.
    
Created on Wed Mar 17 

@author: opsomerl
"""
import pandas as pd
import numpy as np
from scipy import signal


def import_data(file_path):
    """Imports data from a CODA *.txt file and stores it in a data frame"""


    # Import data and store it in a data frame   
    df = pd.read_csv(file_path, 
                      sep = '\t', 
                      header = None,
                      skiprows = 5)
    
    # Rename columns
    nvar = np.size(df,1)
    nmrk = int((nvar-1)/4)
    colnames = ['time']
    for i in range(nmrk):
        mrk = i+1
        Xname = ['Marker%d_X' % mrk]
        Yname = ['Marker%d_Y' % mrk]
        Zname = ['Marker%d_Z' % mrk]
        Vname = ['Marker%d_Visibility' % mrk]
        colnames = colnames + Xname + Yname + Zname + Vname
        
    df.columns = colnames
    
    # Set occluded samples to NaNs
    for i in range(nmrk):
        mrk = i+1
        Xname = 'Marker%d_X' % mrk
        Yname = 'Marker%d_Y' % mrk
        Zname = 'Marker%d_Z' % mrk
        Vname = 'Marker%d_Visibility' % mrk
        
        df.loc[df[Vname] == 0, Xname] = np.nan
        df.loc[df[Vname] == 0, Yname] = np.nan
        df.loc[df[Vname] == 0, Zname] = np.nan

    return df
 
    


def manipulandum_center(coda_df, markers_id=[1,2,3,4]):
    """Computes the position of the center of the manipulandum from the position
    of the four markers.     
        
    Args:
    -----
    coda_df: data_frame
             data frame containing the position of all markers a        
    markers_id: integer_list
                ids of markers 1, 2, 3, 4 with 1 being top left, 
                2 being top right, 3 being bottom left and 4 being 
                bottom right:
        
                        1-----2
                        |     |
                        | GLM |    (front view)
                        |     |
                        3-----4
                        
                                | X
                                |
                  FRAME         |
                                |
                       _________|
                       Y
        
    """
    
    # Store 3-d positions in matrices
    pos1x = coda_df['Marker%d_X' % markers_id[0]].to_numpy()
    pos1y = coda_df['Marker%d_Y' % markers_id[0]].to_numpy()
    pos1z = coda_df['Marker%d_Z' % markers_id[0]].to_numpy()
    pos1 = np.vstack((pos1x,pos1y,pos1z))
    
    pos2x = coda_df['Marker%d_X' % markers_id[1]].to_numpy()
    pos2y = coda_df['Marker%d_Y' % markers_id[1]].to_numpy()
    pos2z = coda_df['Marker%d_Z' % markers_id[1]].to_numpy()
    pos2 = np.vstack((pos2x,pos2y,pos2z)) 
    
    pos3x = coda_df['Marker%d_X' % markers_id[2]].to_numpy()
    pos3y = coda_df['Marker%d_Y' % markers_id[2]].to_numpy()
    pos3z = coda_df['Marker%d_Z' % markers_id[2]].to_numpy()
    pos3 = np.vstack((pos3x,pos3y,pos3z))
    
    pos4x = coda_df['Marker%d_X' % markers_id[3]].to_numpy()
    pos4y = coda_df['Marker%d_Y' % markers_id[3]].to_numpy()
    pos4z = coda_df['Marker%d_Z' % markers_id[3]].to_numpy()
    pos4 = np.vstack((pos4x,pos4y,pos4z))
      
    # Compute X-axis
    X1 = (pos1 - pos3)
    X1 = X1 / np.linalg.norm(X1,axis=0)
    X2 = (pos2 - pos4)
    X2 = X2 / np.linalg.norm(X2,axis=0)
    
    # Compute Y-axis
    Y1 = (pos1 - pos2)
    Y1 = Y1 / np.linalg.norm(Y1,axis=0)
    Y2 = (pos3 - pos4)
    Y2 = Y2 / np.linalg.norm(Y2,axis=0)

    # Compute the center of the manipulandum from the four triplets of markers
    
    # 124
    Z = np.cross(X2,Y1,axisa=0,axisb=0,axisc=0)
    Z = Z / np.linalg.norm(Z,axis=0)
    C1 = pos1 - 37*X2 - 10*Y1 - 20*Z
    
    # 243
    Z = np.cross(X2,Y2,axisa=0,axisb=0,axisc=0)
    Z = Z / np.linalg.norm(Z,axis=0)
    C2 = pos2 - 37*X2 + 10*Y2 - 20*Z
    
    # 431
    Z = np.cross(X1,Y2,axisa=0,axisb=0,axisc=0)
    Z = Z / np.linalg.norm(Z,axis=0)
    C3 = pos4 + 37*X1 + 10*Y2 - 20*Z
    
    # 312
    Z = np.cross(X1,Y1,axisa=0,axisb=0,axisc=0)
    Z = Z / np.linalg.norm(Z,axis=0)
    C4 = pos3 + 37*X2 - 10*Y2 - 20*Z
    
    # Center = average of the four centers
    C = np.nanmean(np.array((C1,C2,C3,C4)),axis=0)
    
    return C

        
        
        
       
        
        
        
        
        
        



