import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn import preprocessing
from scipy import signal

import glm_data_processing as glm
import get_mu_points as gmp
import get_mu_fit as gmf
from os import path

import coda_tools as coda
import processing_tools as tool

ntrials = [1, 2, 3, 4, 5]  # /!\ changer noms de fichiers
positions = ['UR', 'SP', 'UD']
names = ['LH', 'GD', 'PDs', 'MH']
sujet = {
  "GD": "Sujet 1",
  "LH": "Sujet 3",
  "PDs": "Sujet 2",
    "MH": "Sujet 4"
}
positionsdico={
    "SP": "Supine",
    "UD":"UpsideDowm",
    "UR":"UpRight"
}


nb_magique = 576
min = np.Infinity
colors = ["lightpink","paleturquoise","paleturquoise","moccasin","moccasin"]

def trim_axs(axs, N):
    """
    Reduce *axs* to *N* Axes. All further Axes are removed from the figure.
    """
    axs = axs.flat
    for ax in axs[N:]:
        ax.remove()
    return axs[:N]


for name in names:
    axs = plt.figure(figsize=(22, 15), constrained_layout=True).subplots(3, 5)
    axs = trim_axs(axs, 15)
    a = 0
    for n in ntrials:
        for p in positions:
            file_path = "../../data/Groupe_1_codas/%s_%s_coda000%d.txt" % (name, p, n)
            if not path.exists(file_path):
                continue
            coda_df = coda.import_data(file_path)
            time1 = coda_df.time.to_numpy()
            markers_id = [6, 5, 8, 7]

            pos = coda.manipulandum_center(coda_df, markers_id)
            pos = pos / 1000
            vel = tool.derive(pos, 200,axis=1)

            # %% CUTTING THE TASK INTO SEGMENTS (your first task)
            pk = signal.find_peaks(vel[0], prominence=1, width=(100, 1000))
            ipk = pk[0]  # index
            cycle_starts = ipk[:-1]
            cycle_ends = ipk[1:] - 1
            if len(cycle_starts) == 0:
                continue
            allx = []
            ally = []
            allz = []
            index = (np.arange(0, nb_magique) / nb_magique) * 100
            for i in range(len(cycle_starts)):
                xbis = pos[0][cycle_starts[i]:cycle_ends[i]]
                xbis=xbis[0:nb_magique]
                allx.append(xbis/np.linalg.norm(xbis))
                ybis = pos[1][cycle_starts[i]:cycle_ends[i]]
                ybis=ybis[0:nb_magique]
                ally.append(ybis/np.linalg.norm(ybis))
                zbis = pos[2][cycle_starts[i]:cycle_ends[i]]
                zbis = zbis[0:nb_magique]
                allz.append(zbis / np.linalg.norm(zbis))

            xmean = np.nanmean(allx, axis=0)
            ymean = np.nanmean(ally, axis=0)
            zmean = np.nanmean(allz, axis=0)
            xstd = np.nanstd(allx, axis=0)
            ystd = np.nanstd(ally, axis=0)
            zstd =np.nanstd(allz, axis=0)
            axs[a].plot(index, xmean)
            axs[a].fill_between(index, xmean + xstd, xmean - xstd, alpha=0.3, label=positionsdico[p])
            axs[a].set_facecolor(colors[n-1])
            axs[a].legend()
            axs[a].set_ylim(-0.060,-0.020)
            axs[a + 5].plot(index, ymean)
            axs[a + 5].fill_between(index, ymean + ystd, ymean - ystd, alpha=0.3, label=positionsdico[p])
            axs[a+5].set_ylim(-0.050, -0.035)
            axs[a+5].set_facecolor(colors[n-1])
            axs[a + 10].plot(index, zmean)
            axs[a + 10].fill_between(index, zmean + zstd, zmean - zstd, alpha=0.3, label=positionsdico[p])
            axs[a + 10].set_ylim(0.034, 0.048)
            axs[a + 10].set_facecolor(colors[n - 1])
            axs[a + 10].set_xlabel('Cycle(%)')
            if a==0:
                axs[a].set_ylabel('x')
                axs[a+5].set_ylabel('y')
                axs[a+10].set_ylabel('z')

        a += 1
    plt.suptitle("%s x, y et z"%sujet[name], fontsize=24)
    plt.subplots_adjust(top=0.95)
    plt.savefig("%s_fillbetween.png"%name)
