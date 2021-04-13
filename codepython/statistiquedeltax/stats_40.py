import numpy as np
import matplotlib.pyplot as plt
import math
import statsmodels.stats.weightstats as sm2
from scipy import signal
from scipy import stats as sc

import glm_data_processing as glm
import get_mu_points as gmp
import get_mu_fit as gmf
from os import path

import coda_tools as coda
import processing_tools as tool

# %% Comparaison variance
markers_id = [6, 5, 8, 7]
ntrials = [2, 3, 4,5]  # /!\ changer noms de fichiers
positions = ['UR', 'SP', 'UD']
names = ['LH', 'GD', 'PDs', 'MH']
positionsdico = {
    "SP": "Supine",
    "UD": "UpsidDowm",
    "UR": "UpRight"
}
for name in names:
    file1 = open("stats40_%s" % name, "w")
    for p in positions:
        file1.write("######################## Position {} ###############################################\n".format(
            positionsdico[p]))
        for n in ntrials:
            file1.write("Stats DeltaX pour %s en position %s et l'essais %d\n" % (name, p, n))
            file_path1 = "../../data/Groupe_1_codas/%s_%s_coda000%d.txt" % (name, p, n)
            if not path.exists(file_path1):
                continue
            coda_df1 = coda.import_data(file_path1)
            pos1 = coda.manipulandum_center(coda_df1, markers_id)
            pos1 = pos1 / 1000
            vel1 = tool.derive(pos1, 200, axis=1)
            # posxbis1 = pos1[0][~np.isnan(pos1[0])]

            pk = signal.find_peaks(vel1[0], prominence=1, width=(100, 1000))
            ipk = pk[0]  # index
            cycle_starts = ipk[:-1]
            cycle_ends = ipk[1:] - 1

            ecart1 = []
            for k in range(len(cycle_starts)):
                ecart1.append(abs(np.nanmax(pos1[0][cycle_starts[k]:cycle_ends[k]]) - np.nanmin(
                    pos1[0][cycle_starts[k]:cycle_ends[k]])))

            result = sc.ttest_1samp(ecart1, 0.40)
            print(result)
            file1.write("p_value is %f" % result[1] + "\n")
            file1.write("la moyenne est: %f \n" % (np.nanmean(ecart1)))
            file1.write('\n')
            file1.write('\n')
    file1.close()
