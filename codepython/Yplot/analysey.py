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
colors = ['plum', 'aquamarine', 'aquamarine', 'royalblue', 'royalblue']

for name in names:
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(15, 10))
    tup = (ax1, ax2, ax3)
    for p, ax in zip(positions, tup):
        box = []
        for n in ntrials:
            file_path = "../../data/Groupe_1_codas/%s_%s_coda000%d.txt" % (name, p, n)
            if not path.exists(file_path):
                box.append([])
            else:
                coda_df = coda.import_data(file_path)
                time1 = coda_df.time.to_numpy()
                markers_id = [6, 5, 8, 7]

                pos = coda.manipulandum_center(coda_df, markers_id)
                pos = pos / 1000
                baseline = range(0, 400)
                pos[1]-=np.nanmean(pos[1][baseline])
                posybis = pos[1][~np.isnan(pos[1])]
                vel = tool.derive(pos, 200, axis=1)
                box.append(posybis/np.linalg.norm(posybis))
        boxplot=ax.boxplot(box, notch=True,vert=True,medianprops={'color':'purple'}, patch_artist=True, showfliers=False)
        for patch, color in zip(boxplot['boxes'], colors):
            patch.set_facecolor(color)
        ax.legend([boxplot["boxes"][0],boxplot["boxes"][1],boxplot["boxes"][3]], ['train', 'no blind','blind'], loc='upper right')
        ax.set_title("%s"%p)
    fig.suptitle("Y boxplot %s" % name)
    plt.savefig("y_box_for_%s.png" % name)
