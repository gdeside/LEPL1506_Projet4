from os import path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

import coda_tools as coda
import processing_tools as tool

sujet = {
    "GD": "Sujet 1",
    "LH": "Sujet 3",
    "PDs": "Sujet 2",
    "MH": "Sujet 4"
}
positionsdico = {
    "SP": "Supine",
    "UD": "UpsidDowm",
    "UR": "UpRight"
}
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
                vel = tool.derive(pos, 200, axis=1)

                pk = signal.find_peaks(vel[0], prominence=1, width=(100, 1000))
                ipk = pk[0]  # index
                cycle_starts = ipk[:-1]
                cycle_ends = ipk[1:] - 1

                angle = []
                for k in range(len(cycle_starts)):
                    ecartx = (abs(np.nanmax(pos[0][cycle_starts[k]:cycle_ends[k]]) - np.nanmin(
                        pos[0][cycle_starts[k]:cycle_ends[k]])))
                    ecarty = (abs(np.nanmax(pos[1][cycle_starts[k]:cycle_ends[k]]) - np.nanmin(
                        pos[1][cycle_starts[k]:cycle_ends[k]])))
                    angle.append(90 - np.degrees(np.arctan(ecarty / ecartx)))

                box.append(angle)
        boxplot = ax.boxplot(box, notch=False, vert=True, medianprops={'color': 'purple'}, patch_artist=True,
                             showfliers=False)
        for patch, color in zip(boxplot['boxes'], colors):
            patch.set_facecolor(color)
        ax.legend([boxplot["boxes"][0], boxplot["boxes"][1], boxplot["boxes"][3]], ['train', 'no blind', 'blind'],
                  loc='upper right')
        ax.set_title("%s" % positionsdico[p])
        ax.set_ylabel('angle[degrees]')
        ax.set_ylim(70, 90)
        if p == 'UD':
            ax.set_xlabel('essais')
    fig.suptitle("evolution angle %s" % sujet[name])
    plt.savefig("box_angle_for_%s.png" % name)
