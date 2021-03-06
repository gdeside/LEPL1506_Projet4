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

ntrials = 5  # /!\ changer noms de fichiers
positions = ["UR", 'SP', 'UD']
names = ['GD','PDS','MH','LH']

for name in names:
    for p in positions:
        for n in range(2, ntrials + 1):
            glm_path = "../../data/%s_%s_00%d.glm" % (name, p, n)
            if not path.exists(glm_path):
                continue

            glm_df = glm.import_data(glm_path)
            baseline = range(0, 400)
            # Normal Force exerted by the thumb
            NF_thumb = glm_df.loc[:, 'Fygl'] - np.nanmean(glm_df.loc[baseline, 'Fygl'])
            # Vertical Tangential Force exerted by the thumb
            TFx_thumb = glm_df.loc[:, 'Fxgl'] - np.nanmean(glm_df.loc[baseline, 'Fxgl'])
            # Horizontal Tangential Force exerted by the thumb
            TFz_thumb = glm_df.loc[:, 'Fzgl'] - np.nanmean(glm_df.loc[baseline, 'Fzgl'])

            # Normal Force exerted by the index
            NF_index = -(glm_df.loc[:, 'Fygr'] - np.nanmean(glm_df.loc[baseline, 'Fygr']))
            # Vertical Tangential Force exerted by the index
            TFx_index = glm_df.loc[:, 'Fxgr'] - np.nanmean(glm_df.loc[baseline, 'Fxgr'])
            # Horizontal Tangential Force exerted by the index
            TFz_index = glm_df.loc[:, 'Fzgr'] - np.nanmean(glm_df.loc[baseline, 'Fzgr'])

            # %% Get acceleration, LF and GF
            time = glm_df.loc[:, 'time'].to_numpy()
            accX = glm_df.loc[:, 'LowAcc_X'].to_numpy() * (-9.81)
            accX = accX - np.nanmean(accX[baseline])
            GF = glm_df.loc[:, 'GF'].to_numpy()
            GF = GF - np.nanmean(GF[baseline])
            LFv = TFx_thumb + TFx_index
            LFh = TFz_thumb + TFz_index
            LF = np.hypot(LFv, LFh)

            # %%Filter data
            freqAcq = 800  # Frequence d'acquisition des donnees
            freqFiltAcc = 20  # Frequence de coupure de l'acceleration
            freqFiltForces = 20  # Frequence de coupure des forces

            accX = glm.filter_signal(accX, fs=freqAcq, fc=freqFiltAcc)
            GF = glm.filter_signal(GF, fs=freqAcq, fc=freqFiltForces)
            LF = glm.filter_signal(LF, fs=freqAcq, fc=freqFiltForces)
            LFv = glm.filter_signal(LFv, fs=freqAcq, fc=freqFiltForces)
            LFh = glm.filter_signal(LFh, fs=freqAcq, fc=freqFiltForces)

            file_path = "../../data/Groupe_1_codas/%s_%s_coda000%d.txt" % (name, p, n)
            if not path.exists(file_path):
                continue
            coda_df = coda.import_data(file_path)
            time1 = coda_df.time.to_numpy()
            markers_id = [6, 5, 8, 7]

            # Center position
            pos = coda.manipulandum_center(coda_df, markers_id)
            pos = pos / 1000

            # Filter position signal
            # pos = tool.filter_signal(pos, axis=1, fs=200, fc=10, N=4)

            # Derive position to get velocity
            vel = tool.derive(pos, 200, axis=1)

            # %% CUTTING THE TASK INTO SEGMENTS (your first task)
            pk = signal.find_peaks(vel[0], prominence=1, width=(100, 1000))
            ipk = pk[0]  # index
            cycle_starts = ipk[:-1]
            cycle_ends = ipk[1:] - 1
            if len(cycle_starts) == 0:
                continue

            fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, figsize=(15, 15))
            ax1.plot(time1, pos[0])
            ax1.set_xlim(time1[cycle_starts[3]], time1[cycle_ends[3]])
            ax1.set_ylabel("x[m]")
            ax1.set_title('position x')
            plt.setp(ax1.get_xticklabels(), visible=False)

            ax2.plot(time1, pos[1])
            ax2.set_xlim(time1[cycle_starts[3]],time1[cycle_ends[3]])

            # ax1.set_ylim(-0.80, -0.20)
            ax2.set_title('positions y')
            ax2.set_ylabel("y[m]")
            plt.setp(ax2.get_xticklabels(), visible=False)
            ax3.plot(time1, vel[0])
            ax3.set_title('velocity')
            ax3.set_ylabel("v[m/s]")
            ax3.set_xlim(time1[cycle_starts[3]], time1[cycle_ends[3]])
            plt.setp(ax3.get_xticklabels(), visible=False)
            ax4.plot(time, accX)
            plt.setp(ax4.get_xticklabels(), visible=False)
            ax4.set_title('accelaration')
            ax4.set_ylabel("a[m/s]")

            ax4.set_xlim(time1[cycle_starts[3]], time1[cycle_ends[3]])
            ax5.plot(time, GF, label="GF")
            ax5.set_title('GF and LF')
            ax5.plot(time, LF, label="LF")
            ax5.set_ylabel("GF et LF [N]")
            ax5.set_xlim(time1[cycle_starts[3]], time1[cycle_ends[3]])
            ax5.legend()
            ax5.set_xlabel('time[s]')

            namefile = "%s_1mvt_%s_%d" % (name, p, n)
            fig.suptitle(namefile)
            plt.savefig("%s.png" % namefile)
