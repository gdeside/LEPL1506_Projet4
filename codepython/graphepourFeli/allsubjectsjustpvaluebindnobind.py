from os import path

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.stats.weightstats as sm2
from scipy import signal
from scipy import stats as sc

import coda_tools as coda
import processing_tools as tool

import glm_data_processing as glm

ntrials = [2, 3, 4, 5]  # /!\ changer noms de fichiers
positions = ['UR', 'SP', 'UD']
names = ['GD', 'PDs', 'LH', 'MH']
colors = ['plum', 'aquamarine', 'aquamarine', 'royalblue', 'royalblue']
sujet = {
    "GD": "Sujet 1",
    "LH": "Sujet 3",
    "PDs": "Sujet 2",
    "MH": "Sujet 4"
}
positionsdico = {
    "SP": "Supine",
    "UD": "UpsideDown",
    "UR": "UpRight"
}

sujetcolor = {
    "PDs": "deeppink",
    "MH": "black",
    "GD": "green",
    "LH": "blueviolet"
}
sujetmarker = {
    "GD": "d",
    "MH": "o",
    "LH": "s",
    "PDs": "*"
}
indexsubject = {
    "GD": [1, 2],
    "MH": [7, 8],
    "LH": [5, 6],
    "PDs": [3, 4]
}


def transformpvalue(p: float):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return 'ns'


###meanlucile
glm_path = "../../data/%s_%s_00%d.glm" % ('LH', 'UD', 5)
glm_df = glm.import_data(glm_path)
baseline = range(101, 400)
GF = glm_df.loc[:, 'GF'].to_numpy()
meanlucile = np.nanmean(GF[baseline])
##


file1 = open("statsBNB_delatz", "w")
for p in positions:
   
    a = -3
    openarray1 = []
    openarray2 = []
    for name in names:
        a += 1
        for n in ntrials:
            glm_path = "../../data/%s_%s_00%d.glm" % (name, p, n)
            if not path.exists(glm_path):
                continue

            glm_df = glm.import_data(glm_path)
            if name == 'LH' and p == 'UD':
                baseline = range(101, 400)
            else:
                baseline = range(0, 400)
            # Normal Force exerted by the thumb
            NF_thumb = glm_df.loc[:, 'Fygl'] - np.nanmean(glm_df.loc[baseline, 'Fygl'])
            # Vertical Tangential Force exerted by the thumb
            TFx_thumb = glm_df.loc[:, 'Fxgl'] - np.nanmean(glm_df.loc[baseline, 'Fxgl'])
            # Horizontal Tangential Force exerted by the thumb
            TFz_thumb = glm_df.loc[:, 'Fzgl'] - np.nanmean(glm_df.loc[baseline, 'Fzgl'])

            # Normal Force exerted by the index
            NF_index = -(glm_df.loc[:, 'Fygr'] - np.nanmean(glm_df.loc[baseline, 'Fygr']))
            # Vertical Tangential Force exerted by the index
            TFx_index = glm_df.loc[:, 'Fxgr'] - np.nanmean(glm_df.loc[baseline, 'Fxgr'])
            # Horizontal Tangential Force exerted by the index
            TFz_index = glm_df.loc[:, 'Fzgr'] - np.nanmean(glm_df.loc[baseline, 'Fzgr'])

            # %% Get acceleration, LF and GF
            time = glm_df.loc[:, 'time'].to_numpy()
            accX = glm_df.loc[:, 'LowAcc_X'].to_numpy() * (-9.81)
            accX = accX - np.nanmean(accX[baseline])
            if name == 'LH' and p == 'UD':
                GF = glm_df.loc[:, 'GF'].to_numpy()
                GF = GF - meanlucile
            else:
                GF = glm_df.loc[:, 'GF'].to_numpy()
                GF = GF - np.nanmean(GF[baseline])
            LFv = TFx_thumb + TFx_index
            LFh = TFz_thumb + TFz_index
            LF = np.hypot(LFv, LFh)

            # %%Filter data
            freqAcq = 800  # Frequence d'acquisition des donnees
            freqFiltAcc = 20  # Frequence de coupure de l'acceleration
            freqFiltForces = 20  # Frequence de coupure des forces

            accX = glm.filter_signal(accX, fs=freqAcq, fc=freqFiltAcc)
            GF = glm.filter_signal(GF, fs=freqAcq, fc=freqFiltForces)
            LF = glm.filter_signal(LF, fs=freqAcq, fc=freqFiltForces)
            LFv = glm.filter_signal(LFv, fs=freqAcq, fc=freqFiltForces)
            LFh = glm.filter_signal(LFh, fs=freqAcq, fc=freqFiltForces)

            file_path = "../../data/Groupe_1_codas/%s_%s_coda000%d.txt" % (name, p, n)
            if not path.exists(file_path):
                continue
            coda_df = coda.import_data(file_path)
            time1 = coda_df.time.to_numpy()
            markers_id = [6, 5, 8, 7]

            # Center position
            pos = coda.manipulandum_center(coda_df, markers_id)
            pos = pos / 1000

            # Filter position signal
            # pos = tool.filter_signal(pos, axis=1, fs=200, fc=10, N=4)

            # Derive position to get velocity
            vel = tool.derive(pos, 200, axis=1)

            # %% CUTTING THE TASK INTO SEGMENTS (your first task)
            pk = signal.find_peaks(vel[0], prominence=1, width=(100, 1000))
            ipk = pk[0]  # index
            cycle_starts = ipk[:-1]
            cycle_ends = ipk[1:] - 1

            GFmax = []
            indexGF=[]
            LFmax=[]
            indexLF=[]
            for i in range(len(cycle_starts)):
                id = np.where((time > time1[cycle_starts[i]]) & (time < time1[cycle_ends[i]]))
                idbis = np.nanargmax(GF[id])
                indexGF.append(time[id][idbis])
                GFmax.append(GF[id][idbis])
                # print(len(time[id]))
                if idbis > 500 and idbis + 500 < len(time[id]):
                    idarray = np.arange(idbis - 500, idbis + 500)
                elif idbis > 500 and idbis + 500 > len(time[id]):
                    idarray = np.arange(idbis - 500, len(time[id]))
                else:
                    idarray = np.arange(0, idbis + 250)
                idbis1 = np.nanargmax(LF[id][idarray])
                indexLF.append(time[id][idarray][idbis1])
                LFmax.append(LF[id][idarray][idbis1])
            plt.plot(time,GF,label='GF')
            plt.plot(time,LF,label='LF')
            plt.scatter(indexGF,GFmax,label='GF max')
            plt.scatter(indexLF,LFmax,label='LF max')
            plt.legend()
            plt.title("%s en position %s essais %d"%(sujet[name],positionsdico[p],n))
            plt.show()
            

