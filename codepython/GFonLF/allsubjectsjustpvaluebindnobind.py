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

fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(7, 10))
tup = (ax1, ax2, ax3)
file1 = open("statschgtdecondition", "w")
for p, ax in zip(positions, tup):
    file1.write("########################%s######################\n" % positionsdico[p])
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
            for i in range(len(cycle_starts)):
                id = np.where((time > time1[cycle_starts[i]]) & (time < time1[cycle_ends[i]]))
                if n == 2 or n == 3:
                    openarray1.append(np.nanmax(GF[id]) / np.nanmax(LF[id]))
                if n == 5 or n == 4:
                    openarray2.append(np.nanmax(GF[id]) / np.nanmax(LF[id]))

        X1 = sm2.DescrStatsW(openarray1)
        X2 = sm2.DescrStatsW(openarray2)
        Ttest = sm2.CompareMeans(X1, X2)
        t2, p2 = sc.ttest_ind(openarray1, openarray2)
        Txbis, pvalbis = sc.bartlett(openarray1, openarray2)
        file1.write(Ttest.summary(usevar='pooled').as_text() + "\n")
        file1.write("les deux moyennes sont: %f et %f\n" % (np.nanmean(openarray1), np.nanmean(openarray2)))
        file1.write("p_value pour la variance %f \n" % pvalbis)
        file1.write("les deux variances sont %f et %f\n" % (np.nanstd(openarray1), np.nanstd(openarray2)))
        box = [np.nanmean(openarray1), np.nanmean(openarray2)]
        box1 = [np.nanstd(openarray1), np.nanstd(openarray2)]
        index = indexsubject[name]
        indexgraph1 = np.linspace(index[0] - 0.25, index[0] + 0.25, 50)
        plotarray1 = np.zeros(50) + box[0]
        indexgraph2 = np.linspace(index[1] - 0.25, index[1] + 0.25, 50)
        plotarray2 = np.zeros(50) + box[1]
        indexscatter1 = np.zeros(len(openarray1)) + index[0]
        indexscatter2 = np.zeros(len(openarray2)) + index[1]
        if ax == ax3:
            ax.plot(indexgraph1, plotarray1, linestyle='dotted', color=sujetcolor[name],
                    label=sujet[name])
            ax.scatter(indexscatter1, openarray1, color=sujetcolor[name], alpha=0.5, s=20)
            ax.plot(indexgraph2, plotarray2, linestyle='dotted', color=sujetcolor[name])
            ax.scatter(indexscatter2, openarray2, color=sujetcolor[name], alpha=0.5, s=20)
            ax.text(index[0] + 0.25, 10, 'mean:%s' % transformpvalue(p2), fontsize=13)
            ax.text(index[0] + 0.25, 8.5, 'std:%s' % transformpvalue(pvalbis), fontsize=13)
        else:
            ax.plot(indexgraph1, plotarray1, linestyle='dotted', color=sujetcolor[name])
            ax.scatter(indexscatter1, openarray1, color=sujetcolor[name], alpha=0.5, s=20)
            ax.plot(indexgraph2, plotarray2, linestyle='dotted', color=sujetcolor[name])
            ax.scatter(indexscatter2, openarray2, color=sujetcolor[name], alpha=0.5, s=20)
            ax.text(index[0] + 0.25, 10, 'mean:%s' % transformpvalue(p2), fontsize=13)
            ax.text(index[0] + 0.25, 8.5, 'std:%s' % transformpvalue(pvalbis), fontsize=13)
    ax.set_ylim(0.0, 11)
    ax.set_xlim(0.85, 2.15)
    ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8])
    ax.set_xlim(0.5, 8.5)
    ax.set_xticklabels(["no blind", 'blind', "no blind", 'blind', "no blind", 'blind', "no blind", 'blind'])
    ax.set_title("%s" % positionsdico[p], fontweight='bold')
    ax.set_ylabel("GF/LF")
    if p == 'UD':
        ax.legend(loc="center left", prop={'size': 8})
        # ax.set_xlabel('blocs(#)')
fig.suptitle("Comparaison De la moyenne de la GFmax/LFmax entre les deux conditions")
plt.savefig("errorbar_en_GFonLF_for_allpvaluebindnobind.png")
file1.close()
