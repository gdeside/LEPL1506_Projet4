from os import path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy import stats as sc
import statsmodels.stats.weightstats as sm2
import coda_tools as coda
import processing_tools as tool

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

meanopen = []
meanclose = []
for p in positions:
    for name in names:
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
                    ecart.append(abs(np.nanmax(pos[1][cycle_starts[k]:cycle_ends[k]]) - np.nanmin(
                        pos[1][cycle_starts[k]:cycle_ends[k]])))
                if n == 2 or n == 3:
                    if not np.isnan(np.nanmean(ecart)):
                        meanopen.append(np.nanmean(ecart))
                if n == 4 or n == 5:
                    if not np.isnan(np.nanmean(ecart)):
                        meanclose.append(np.nanmean(ecart))

result = []
print(meanclose)
print(meanopen)
X1 = sm2.DescrStatsW(meanopen)
X2 = sm2.DescrStatsW(meanclose)
Ttest = sm2.CompareMeans(X1, X2)
test1=Ttest.ttest_ind()
print(Ttest.summary())
result.append(np.nanmean(meanopen))
result.append(np.nanmean(meanclose))
fig, ax = plt.subplots(figsize=(5,5))
label = ['no bind', 'bind']
ind = np.arange(2)
hbars = ax.bar(ind, result, width=0.35, align='center')
ax.set_xticks(ind)
ax.set_xticklabels(label)
ax.text(-0.05, 0.04, 'pvalue is for mean: %f'% test1[1], style='italic',
        bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})

fig.suptitle("Amplitude moyenne en Y pour yeux ouverts et fermés")
plt.savefig("bar_y.png")
