from os import path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import statsmodels.stats.weightstats as sm2
from scipy import stats as sc

import coda_tools as coda
import processing_tools as tool
import glm_data_processing as glm

ntrials = [4, 5]  # /!\ changer noms de fichiers
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
meanlucileGF = np.nanmean(GF[baseline])

fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(7, 10))
tup = (ax1, ax2, ax3)
file1 = open("statsblindall", "w")
for p, ax in zip(positions, tup):
    file1.write("########################%s######################\n" % positionsdico[p])
    arrayopenall = []
    arraycloseall = []
    for name in names:
        arrayopen = []
        arrayclose = []
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
                GF = GF - meanlucileGF
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

                ecart = []
                for i in range(len(cycle_starts)):
                    id = np.where((time > time1[cycle_starts[i]]) & (time < time1[cycle_ends[i]]))
                    if n == 2 or n == 4:
                        ecart.append(np.nanmax(GF[id]) / np.nanmax(LF[id]))
                    if n == 5 or n == 3:
                        ecart.append(np.nanmax(GF[id]) / np.nanmax(LF[id]))

                if n == 2 or n == 4:
                    arrayopen.append(np.nanmean(ecart))
                if n == 5 or n == 3:
                    arrayclose.append(np.nanmean(ecart))

        arraycloseall.append(np.nanmean(arrayclose))
        arrayopenall.append(np.nanmean(arrayopen))
    X1 = sm2.DescrStatsW(arrayopenall)
    X2 = sm2.DescrStatsW(arraycloseall)
    Ttest = sm2.CompareMeans(X1, X2)
    t2, p2 = sc.ttest_ind(arrayopenall, arraycloseall)
    Txbis, pvalbis = sc.bartlett(arrayopenall, arraycloseall)
    file1.write(Ttest.summary(usevar='pooled').as_text() + "\n")
    file1.write("les deux moyennes sont: %f et %f\n" % (np.nanmean(arrayopenall), np.nanmean(arraycloseall)))
    file1.write("p_value pour la variance %f \n" % pvalbis)
    file1.write("les deux variances sont %f et %f\n" % (np.nanstd(arrayopenall), np.nanstd(arraycloseall)))
    index = [1, 2]
    indexgraph1 = np.linspace(index[0] - 0.25, index[0] + 0.25, 50)
    plotarray1 = np.zeros(50) + np.nanmean(arrayopenall)
    indexgraph2 = np.linspace(index[1] - 0.25, index[1] + 0.25, 50)
    plotarray2 = np.zeros(50) + np.nanmean(arraycloseall)
    indexscatter1 = np.zeros(len(arrayopenall)) + index[0]
    indexscatter2 = np.zeros(len(arraycloseall)) + index[1]
    ax.plot(indexgraph1, plotarray1, linestyle='dotted')
    ax.plot(indexgraph2, plotarray2, linestyle='dotted')
    ax.scatter(indexscatter1, arrayopenall, alpha=0.5, s=20)
    ax.scatter(indexscatter2, arraycloseall, alpha=0.5, s=20)
    ax.text(index[0] + 0.35, 0.48, 'mean:%s' % transformpvalue(p2), fontsize=12)
    ax.text(index[0] + 0.375, 0.455, 'std:%s' % transformpvalue(pvalbis), fontsize=12)
    ax.set_ylim(0.20, 0.55)
    ax.set_xlim(0.8, 2.2)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['no blind', 'blind'])
    ax.set_xlim(0.70, 2.30)
    ax.set_title("%s" % positionsdico[p], fontweight='bold')
    ax.set_ylabel("Amplitude mvt en X[m]")
fig.suptitle("amplitude x Errorbar all subjects")
plt.savefig("34_en_x_for_all.png")
file1.close()
