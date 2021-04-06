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
positions = ['UD', 'SP', 'UR']
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
                continue
            coda_df = coda.import_data(file_path)
            time1 = coda_df.time.to_numpy()
            markers_id = [6, 5, 8, 7]

            pos = coda.manipulandum_center(coda_df, markers_id)
            pos = pos / 1000
            axs[a].plot(time1, pos[0])
            axs[a].set_title("%s %s %d" % (name, p, n))
            a += 1
    plt.savefig("%s.png" % name)
