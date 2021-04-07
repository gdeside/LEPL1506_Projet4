import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import signal
from mpl_toolkits.mplot3d import Axes3D
import glm_data_processing as glm
import get_mu_points as gmp
import get_mu_fit as gmf
from os import path
import coda_tools as coda
import processing_tools as tool


ntrials = 5  # /!\ changer noms de fichiers
positions = ["UR", 'SP', 'UD']
names = ['GD','PDs','LH','MH']

for name in names:
    for p in positions:
        for n in range(1, ntrials + 1):
            file_path = "../../data/Groupe_1_codas/%s_%s_coda000%d.txt" % (name, p, n)
            if not path.exists(file_path):
                continue
            coda_df = coda.import_data(file_path)
            time1 = coda_df.time.to_numpy()
            markers_id = [6, 5, 8, 7]

            # Center position
            pos = coda.manipulandum_center(coda_df, markers_id)
            pos = pos / 1000
            vel = tool.derive(pos, 200, axis=1)

            # %% CUTTING THE TASK INTO SEGMENTS (your first task)
            pk = signal.find_peaks(vel[0], prominence=1, width=(100, 1000))
            ipk = pk[0]  # index
            cycle_starts = ipk[:-1]
            cycle_ends = ipk[1:] - 1
            indexmax=np.nanargmax(pos[0])
            indexmin = np.nanargmin(pos[0])
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(pos[0][indexmax],pos[1][indexmax],pos[2][indexmax],label="max")
            ax.scatter(pos[0][indexmin], pos[1][indexmin], pos[2][indexmin], label="min")
            ax.legend()
            ax.set_xlabel('time[s]')
            ax.set_ylabel('y[m]')
            ax.set_zlabel('x[m]')
            plt.title("3D_%s_%s_%d"%(name,p,n))
            plt.show()