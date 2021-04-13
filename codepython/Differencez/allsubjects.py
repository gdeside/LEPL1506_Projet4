from os import path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

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
                    ecart.append(abs(np.nanmax(pos[2][cycle_starts[k]:cycle_ends[k]]) - np.nanmin(
                        pos[2][cycle_starts[k]:cycle_ends[k]])))
                box.append(np.nanmean(ecart))
                box1.append(np.nanstd(ecart))
                index.append(n)
        if ax==ax2:
            ax.errorbar(index, box, yerr=box1, linestyle='dotted',color=sujetcolor[name],marker=sujetmarker[name],label=sujet[name])
        else:
            ax.errorbar(index, box, yerr=box1, linestyle='dotted',color=sujetcolor[name],marker=sujetmarker[name])
    if ax2==ax:
        ax.axvspan(0.5, 1.50, facecolor='steelblue', alpha=0.2)
        ax.axvspan(1.5, 3.50, facecolor='steelblue', alpha=0.5)
        ax.axvspan(3.5, 5.50, facecolor='steelblue', alpha=0.8)
    else:
        ax.axvspan(0.5, 1.50, facecolor='steelblue', alpha=0.2, label='train')
        ax.axvspan(1.5, 3.50, facecolor='steelblue', alpha=0.5, label='no bind')
        ax.axvspan(3.5, 5.50, facecolor='steelblue', alpha=0.8, label='bind')
    ax.set_ylim(0.0, 0.15)
    ax.set_xlim(0.9,5.1)
    ax.set_xticks([1,2,3,4,5])
    ax.set_title("%s" % positionsdico[p])
    ax.set_ylabel("Amplitude mvt [m]")
    if p == 'UR':
        ax.legend(loc="upper left")
    if p=="SP":
        ax.legend(loc="upper left")
    if p == 'UD':
        ax.set_xlabel('blocs(#)')
fig.suptitle("amplitude z Errorbar all subjects")
plt.savefig("errorbar_en_z_for_all.png")

