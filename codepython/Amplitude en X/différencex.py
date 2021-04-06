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

ntrials = [1, 2, 3, 4, 5]  # /!\ changer noms de fichiers
positions = ['UR', 'SP', 'UD']
names = ['LH', 'GD', 'PDs', 'MH']


def trim_axs(axs, N):
    """
    Reduce *axs* to *N* Axes. All further Axes are removed from the figure.
    """
    axs = axs.flat
    for ax in axs[N:]:
        ax.remove()
    return axs[:N]


for name in names:
    axs = plt.figure(figsize=(15, 10), constrained_layout=True).subplots(5, 3)
    axs = trim_axs(axs, 15)
    a = 0
    for n in ntrials:
        for p in positions:
            file_path = "../../data/Groupe_1_codas/%s_%s_coda000%d.txt" % (name, p, n)
            if not path.exists(file_path):
                a+=1
                continue
            coda_df = coda.import_data(file_path)
            time1 = coda_df.time.to_numpy()
            markers_id = [6, 5, 8, 7]

            pos = coda.manipulandum_center(coda_df, markers_id)
            pos = pos / 1000
            vel = tool.derive(pos, 200, axis=1)

            # %% CUTTING THE TASK INTO SEGMENTS (your first task)
            pk = signal.find_peaks(vel[0], prominence=1, width=(100, 1000))
            ipk = pk[0]  # index
            cycle_starts = ipk[:-1]
            cycle_ends = ipk[1:] - 1
            if len(cycle_starts) == 0:
                a+=1
                continue
            index=[]
            ecart=[]
            for k in range(len(cycle_starts)):
                index.append(k)
                ecart.append(abs(np.nanmax(pos[0][cycle_starts[k]:cycle_ends[k]])-np.nanmin(pos[0][cycle_starts[k]:cycle_ends[k]])))
            axs[a].plot(index, ecart)
            axs[a].set_title("%s %s %d" % (name, p, n))
            axs[a].set_ylabel("amplitude[m]")
            axs[a].set_ylim(0.20,0.55)
            a += 1
    plt.suptitle("difference en x pour %s"%name)
    plt.savefig("diffrence_en_x_for_%s.png" % name)
