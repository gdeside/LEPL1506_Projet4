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
names = ['PDS','GD','LH','MH']

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
            print(len(cycle_starts))
            if len(cycle_starts) == 0:
                continue

            fig, (ax1, ax2, ax3, ax4, ax5,ax6) = plt.subplots(6, figsize=(20, 20))
            ax1.plot(time1, pos[0])
            ax1.set_ylabel("x[m]")
            ax1.set_title('positions x')
            for i in range(0, len(cycle_starts)):
                rect0 = plt.Rectangle((time1[cycle_starts[i]], ax1.get_ylim()[0]),
                                      time1[cycle_ends[i] - cycle_starts[i]], ax1.get_ylim()[1] - ax1.get_ylim()[0], color='k', alpha=0.3)
                ax1.add_patch(rect0)
            plt.setp(ax1.get_xticklabels(), visible=False)

            ax2.plot(time1, pos[1])


            # ax1.set_ylim(-0.80, -0.20)
            ax2.set_title('positions y')
            ax2.set_ylabel("y[m]")
            for i in range(0, len(cycle_starts)):
                rect0 = plt.Rectangle((time1[cycle_starts[i]], ax2.get_ylim()[0]),
                                      time1[cycle_ends[i] - cycle_starts[i]], ax2.get_ylim()[1] - ax2.get_ylim()[0], color='k', alpha=0.3)
                ax2.add_patch(rect0)
            plt.setp(ax2.get_xticklabels(), visible=False)
            ax3.plot(time1, pos[2])
            ax3.set_title('positions z')
            ax3.set_ylabel("z[m]")
            for i in range(0, len(cycle_starts)):
                rect0 = plt.Rectangle((time1[cycle_starts[i]], ax3.get_ylim()[0]),
                                      time1[cycle_ends[i] - cycle_starts[i]], ax3.get_ylim()[1] - ax3.get_ylim()[0], color='k', alpha=0.3)
                ax3.add_patch(rect0)
            plt.setp(ax3.get_xticklabels(), visible=False)
            ax4.plot(time1, vel[0])
            ax4.set_title('velocity')
            ax4.set_ylabel("v[m/s]")
            plt.setp(ax3.get_xticklabels(), visible=False)
            ax5.plot(time, accX)
            ax5.set_title('accelaration')
            ax5.set_ylabel("a[m/s²]")
            plt.setp(ax4.get_xticklabels(), visible=False)
            ax6.plot(time, GF, label="GF")
            ax6.set_title('GF and LF')
            ax6.plot(time, LF, label="LF")
            ax6.legend()
            ax6.set_xlabel("time[s]")
            # ax4.set_ylim(0, 20)
            namefile = "%s_%s_%d" % (name, p, n)
            fig.suptitle(namefile)
            plt.savefig("%s.png" % namefile)
