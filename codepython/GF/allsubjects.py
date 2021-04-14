from os import path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

import coda_tools as coda
import processing_tools as tool
import glm_data_processing as glm

ntrials = [1, 2, 3, 4, 5]  # /!\ changer noms de fichiers
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

sujetcolor={
    "PDs":"deeppink",
    "MH": "black",
    "GD":"green",
    "LH":"blueviolet"
}
sujetmarker={
    "GD":"d",
    "MH":"o",
    "LH":"s",
    "PDs":"*"
}

fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(7, 10))
tup = (ax1, ax2, ax3)
for p, ax in zip(positions, tup):
    for name in names:
        box = []
        box1 = []
        index = []
        for n in ntrials:
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
            else:
                coda_df = coda.import_data(file_path)
                time1 = coda_df.time.to_numpy()
                markers_id = [6, 5, 8, 7]

                pos = coda.manipulandum_center(coda_df, markers_id)
                pos = pos / 1000
                vel = tool.derive(pos, 200, axis=1)

                pk = signal.find_peaks(vel[0], prominence=1, width=(100, 1000))
                ipk = pk[0]  # index
                cycle_starts = ipk[:-1]
                cycle_ends = ipk[1:] - 1

                GFmax = []
                for i in range(len(cycle_starts)):
                    id = np.where((time > time1[cycle_starts[i]]) & (time < time1[cycle_ends[i]]))
                    GFmax.append(np.nanmax(GF[id]))
                index.append(n)
                if name == "LH" and (n == 2 or n == 3) and p == "UD":
                    GFmax += (np.nanmean(box) - np.nanmean(GFmax))
                box.append(np.nanmean(GFmax))
                box1.append(np.nanstd(GFmax))
        if ax==ax3:
            ax.errorbar(index, box, yerr=box1, linestyle='dotted',color=sujetcolor[name],marker=sujetmarker[name],label=sujet[name])
        else:
            ax.errorbar(index, box, yerr=box1, linestyle='dotted',color=sujetcolor[name],marker=sujetmarker[name])
    if ax3==ax:
        ax.axvspan(0.5, 1.50, facecolor='steelblue', alpha=0.2)
        ax.axvspan(1.5, 3.50, facecolor='steelblue', alpha=0.5)
        ax.axvspan(3.5, 5.50, facecolor='steelblue', alpha=0.8)
    else:
        ax.axvspan(0.5, 1.50, facecolor='steelblue', alpha=0.2, label='train')
        ax.axvspan(1.5, 3.50, facecolor='steelblue', alpha=0.5, label='no bind')
        ax.axvspan(3.5, 5.50, facecolor='steelblue', alpha=0.8, label='bind')
    #ax.set_ylim(0.20, 0.55)
    ax.set_xlim(0.9,5.1)
    ax.set_xticks([1,2,3,4,5])
    ax.set_title("%s" % positionsdico[p])
    ax.set_ylabel("GF[N]")
    if p == 'UR':
        ax.legend(loc="lower left")
    if p == 'UD':
        ax.legend(loc="lower left")
        ax.set_xlabel('blocs(#)')
fig.suptitle("GF moyenne")
plt.savefig("errorbar_GF.png")

