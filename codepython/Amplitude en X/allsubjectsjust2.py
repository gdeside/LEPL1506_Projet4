from os import path

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.stats.weightstats as sm2
from scipy import signal
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
file1 = open("statsBNB_delatx", "w")
for p, ax in zip(positions, tup):
    a = -3
    file1.write("########################%s######################" % positionsdico[p])
    for name in names:
        openarray = []
        closearray = []
        a += 1
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
                        if n == 2 or n == 3:
                            openarray.append(abs(np.nanmax(pos[0][cycle_starts[k]:cycle_ends[k]]) - np.nanmin(
                                pos[0][cycle_starts[k]:cycle_ends[k]])))
                        if n == 4 or n == 5:
                            closearray.append(abs(np.nanmax(pos[0][cycle_starts[k]:cycle_ends[k]]) - np.nanmin(
                                pos[0][cycle_starts[k]:cycle_ends[k]])))
        X1 = sm2.DescrStatsW(openarray)
        X2 = sm2.DescrStatsW(closearray)
        Ttest = sm2.CompareMeans(X1, X2)
        t2, p2 = sc.ttest_ind(openarray, closearray)
        file1.write(Ttest.summary(usevar='pooled').as_text() + "\n")
        file1.write("les deux moyennes sont: %f et %f\n" % (np.nanmean(openarray), np.nanmean(closearray)))
        Txbis, pvalbis = sc.bartlett(openarray, closearray)
        file1.write("p_value pour la variance %f \n" % pvalbis)
        file1.write("les deux variances sont %f et %f\n" % (np.nanstd(openarray), np.nanstd(closearray)))
        file1.write('****transforme de pvalue =%s\n'%transformpvalue(p2))

        box = [np.nanmean(openarray), np.nanmean(closearray)]
        box1 = [np.nanstd(openarray), np.nanstd(closearray)]
        index = [1 + a * 0.05, 2 + a * 0.05]
        if ax == ax3:
            ax.errorbar(index, box, yerr=box1, linestyle='dotted', color=sujetcolor[name], marker=sujetmarker[name],
                        label=sujet[name])
        else:
            ax.errorbar(index, box, yerr=box1, linestyle='dotted', color=sujetcolor[name], marker=sujetmarker[name])
    ax.set_ylim(0.20, 0.55)
    ax.set_xlim(0.85, 2.15)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["no blind", 'blind'])
    ax.set_title("%s" % positionsdico[p])
    ax.set_ylabel("Amplitude mvt [m]")
    if p == 'UD':
        ax.legend(loc="lower left")
        #ax.set_xlabel('blocs(#)')
fig.suptitle("amplitude x Errorbar all subjects")
plt.savefig("errorbar_en_x_for_all23.png")
file1.close()