import numpy as np
import matplotlib.pyplot as plt
import math

from scipy import signal

import glm_data_processing as glm
import get_mu_points as gmp
import get_mu_fit as gmf
from os import path

# Fermeture des figures ouvertes
plt.close('all')

names = ["GD", "LH", "PDs", "MH"]
positions = ["UR", "SP", "UD"]

ntrials = 5  # /!\ changer noms de fichiers
meanlucile = 0
for n in names:
    for pos in positions:
        gf_train = []
        gf_nobind1 = []
        gf_nobind2 = []
        gf3_bind1 = []
        gf3_bind2 = []
        time = []
        for k in range(5, 1, -1):
            glm_path = "../../data/%s_%s_00%d.glm" % (n, pos, k)
            if not path.exists(glm_path):
                continue

            glm_df = glm.import_data(glm_path)
            baseline = range(101, 400)
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
            if n == 'LH' and pos == 'UD' and k == 5:
                GF = GF - np.nanmean(GF[baseline])
                meanlucile = np.nanmean(GF[baseline])
            elif n == 'LH' and pos == 'UD':
                GF = GF - meanlucile
            else:
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

            if k == 1:
                gf_train.append(LF)
            if k == 2:
                gf_nobind1.append(LF)
            if k == 3:
                gf_nobind2.append(LF)
            if k == 4:
                gf3_bind1.append(LF)
            if k == 5:
                gf3_bind2.append(LF)
        if n == "MH" and pos == "UD":
            fig = plt.figure(figsize=[15, 7])
            plt.title("figures pour %s : LF en fonction du temps en postion %s.png" % (n, pos))
            plt.plot(time, np.mean(gf3_bind1, axis=0), label="GF_bind1")
            plt.legend()
            plt.xlabel("time[s]")
            plt.ylabel("LF[N]")
            fig.savefig("%s_LF_%s.png" % (n, pos))
            continue
        fig = plt.figure(figsize=[15, 7])
        plt.title("figures pour %s : LF en fonction du temps en postion %s.png" % (n, pos))
        plt.plot(time, np.mean(gf_nobind1, axis=0), label="LF_noblind1")
        plt.plot(time, np.mean(gf_nobind2, axis=0), label="LF_noblind2")
        plt.plot(time, np.mean(gf3_bind1, axis=0), label="LF_blind1")
        plt.plot(time, np.mean(gf3_bind2, axis=0), label="LF_blind2")
        plt.legend()
        plt.xlabel("time[s]")
        plt.ylabel("LF[N]")
        fig.savefig("%s_LF_%s.png" % (n, pos))
