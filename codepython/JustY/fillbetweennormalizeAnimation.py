import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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
names = ['GD', 'PDs', 'LH', 'MH']
sujet = {
    "GD": "Sujet 1",
    "LH": "Sujet 3",
    "PDs": "Sujet 2",
    "MH": "Sujet 4"
}

positionsdico = {
    "SP": "Supine",
    "UD": "UpsideDowm",
    "UR": "UpRight"
}

poscolor = {
    "UD": "green",
    "UR": "blue",
    "SP": "orange"
}

conditioncolor = {
    2: "blue",
    3: "blue",
    4: "green",
    5: "green"
}

nb_magique = 576
min = np.Infinity
colors = ["lightpink", "paleturquoise", "paleturquoise", "moccasin", "moccasin"]


def trim_axs(axs, N):
    """
    Reduce *axs* to *N* Axes. All further Axes are removed from the figure.
    """
    axs = axs.flat
    for ax in axs[N:]:
        ax.remove()
    return axs[:N]


meanxnobind = []
meanznobind = []
meanynobind = []

for name in names:
    for p in positions:
        xbisnobind = []
        xbisbind = []
        ybisnobind = []
        ybisbind = []
        zbisnobind = []
        zbisbind = []

        for n in ntrials:
            file_path = "../../data/Groupe_1_codas/%s_%s_coda000%d.txt" % (name, p, n)
            if not path.exists(file_path):
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
                continue
            index = (np.arange(0, nb_magique) / nb_magique) * 100
            for i in range(len(cycle_starts)):
                xbis = pos[0][cycle_starts[i]:cycle_ends[i]]
                ybis = pos[1][cycle_starts[i]:cycle_ends[i]]
                xbis = xbis[0:nb_magique]
                ybis = ybis[0:nb_magique]
                zbis = pos[2][cycle_starts[i]:cycle_ends[i]]
                zbis = zbis[0:nb_magique]
                if n == 2 or n == 3:
                    xbisnobind.append(xbis / np.linalg.norm(xbis))
                    ybisnobind.append(ybis / np.linalg.norm(ybis))
                    zbisnobind.append(zbis / np.linalg.norm(zbis))
                if n == 4 or n == 5:
                    xbisbind.append(xbis / np.linalg.norm(xbis))
                    ybisbind.append(ybis / np.linalg.norm(ybis))
                    zbisbind.append(zbis / np.linalg.norm(zbis))
        xmeannobind = np.nanmean(xbisnobind, axis=0)
        xmeanbind = np.nanmean(xbisbind, axis=0)
        ymeannobind = np.nanmean(ybisnobind, axis=0)
        ymeanbind = np.nanmean(ybisbind, axis=0)
        zmeannobind = np.nanmean(zbisnobind, axis=0)
        zmeanbind = np.nanmean(zbisbind, axis=0)
        xstdnobind = np.nanstd(xbisnobind, axis=0)
        xstdbind = np.nanstd(xbisbind, axis=0)
        ystdnobind = np.nanstd(ybisnobind, axis=0)
        ystdbind = np.nanstd(ybisbind, axis=0)
        zstdnobind = np.nanstd(zbisnobind, axis=0)
        zstdbind = np.nanstd(zbisbind, axis=0)
        index = (np.arange(0, nb_magique) / nb_magique) * 100
        meanxnobind.append((xmeannobind, index))
        meanynobind.append(ymeannobind)
        meanznobind.append(zmeannobind)


def init():
    ax.set_xlim(0,100)
    ax.set_ylim(-0.07,-0.02)
    del xdata[:]
    del ydata[:]
    line.set_data(xdata, ydata)
    return line,


fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)
ax.grid()
xdata, ydata = [], []


def run(data):
    # update the data
    t, y = meanxnobind[0][1][data],meanxnobind[0][0][data]
    xdata.append(t)
    ydata.append(y)
    xmin, xmax =  ax.get_xlim()

    if t >= xmax:
        ax.set_xlim(xmin, 2 * xmax)
        ax.figure.canvas.draw()
    line.set_data(xdata, ydata)

    return line,


anim = animation.FuncAnimation(fig, run, frames=500,interval=20, init_func=init)
plt.show()
