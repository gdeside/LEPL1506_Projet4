import numpy as np
import matplotlib.pyplot as plt
import math

from scipy import signal

import glm_data_processing as glm
import get_mu_points as gmp
import get_mu_fit as gmf
from os import path

import coda_tools as coda
import processing_tools as tool

ntrials = 5  # /!\ changer noms de fichiers
positions = ['UR', 'SP', 'UD']
names = ['PDS', 'GD', 'MH', 'LH']

for name in names:
    GFmean = []
    index = []
    GFstd = []
    a = -1
    for p in positions:
        a += 1
        fig = plt.figure(figsize=[15, 7])
        for n in range(1, ntrials + 1):
            glm_path = "../../data/%s_%s_00%d.glm" % (name, p, n)
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
            for i in range(len(cycle_starts)):
                id = np.where((time > time1[cycle_starts[i]]) & (time < time1[cycle_ends[i]]))
                GFmax.append(np.nanmax(GF[id]))
            index.append(n + a * 5)
            if name == "LH" and (n == 2 or n == 3) and p=="UD":
                GFmax += GFmean[0]
            GFmean.append(np.mean(GFmax))
            GFstd.append(np.std(GFmax))

    plt.errorbar(index, GFmean, yerr=GFstd,linestyle='dotted')
    plt.ylabel("GF[N]")
    plt.xlabel("essais")
    plt.title("Moyenne et ecart type: GFmax pour les diférrents essais pour {}".format(name))
    plt.axvspan(0.5, 1.50, facecolor='azure', alpha=0.5, label='UR train')
    plt.axvspan(1.50, 3.50, facecolor='skyblue', alpha=0.5, label='UR no bind')
    plt.axvspan(3.50, 5.50, facecolor='steelblue', alpha=0.5, label='UR bind')
    plt.axvspan(5.50, 6.50, facecolor='pink', alpha=0.5, label='SP train')
    plt.axvspan(6.50, 8.50, facecolor='hotpink', alpha=0.5, label='SP no bind')
    plt.axvspan(8.50, 10.50, facecolor='deeppink', alpha=0.5, label='SP bind')
    plt.axvspan(10.50, 11.5, facecolor='beige', alpha=0.5, label='UD train')
    plt.axvspan(11.50, 13.5, facecolor='khaki', alpha=0.5, label='UD no bind')
    plt.axvspan(13.50, 15.5, facecolor='yellow', alpha=0.5, label='UD bind')
    plt.legend()
    plt.savefig("meanGFmaxcolor_%s.png" % name)
