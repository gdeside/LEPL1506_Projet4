from os import path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import statsmodels.stats.weightstats as sm2
from scipy import stats as sc

import coda_tools as coda
import processing_tools as tool

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


def transformpvalue(p: float):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return 'ns'


fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(7, 10))
tup = (ax1, ax2, ax3)
file1 = open("stats34_delatx", "w")
for p, ax in zip(positions, tup):
    file1.write("########################%s######################\n" % positionsdico[p])
    arrayopenall = []
    arraycloseall = []
    for name in names:
        arrayopen = []
        arrayclose = []
        for n in ntrials:
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
                for k in range(len(cycle_starts)):
                    if not np.isnan(abs(np.nanmax(pos[0][cycle_starts[k]:cycle_ends[k]]) - np.nanmin(
                            pos[0][cycle_starts[k]:cycle_ends[k]]))):
                        ecart.append(abs(np.nanmax(pos[0][cycle_starts[k]:cycle_ends[k]]) - np.nanmin(
                            pos[0][cycle_starts[k]:cycle_ends[k]])))

                if n == 2 or n == 3:
                    arrayopen.append(np.nanmean(ecart))
                if n == 4 or n == 5:
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
    ax.plot(indexgraph1,plotarray1, linestyle='dotted')
    ax.plot(indexgraph2, plotarray2, linestyle='dotted')
    ax.scatter(indexscatter1,arrayopenall , alpha=0.5, s=20)
    ax.scatter(indexscatter2, arraycloseall, alpha=0.5, s=20)
    ax.text(index[0] + 0.5, 0.48, '%s' % transformpvalue(p2), fontsize=13)
    ax.set_ylim(0.20, 0.55)
    ax.set_xlim(0.8, 2.2)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['no blind', 'blind'])
    ax.set_xlim(0.70, 2.30)
    ax.set_title("%s" % positionsdico[p])
    ax.set_ylabel("Amplitude mvt en X[m]")
fig.suptitle("amplitude x Errorbar all subjects")
plt.savefig("34_en_x_for_all.png")
file1.close()
