from os import path

import numpy as np
import statsmodels.stats.weightstats as sm2
from scipy import signal
from scipy import stats as sc

import coda_tools as coda
import processing_tools as tool

# %% Comparaison variance
markers_id = [6, 5, 8, 7]
ntrials = [2, 3, 4]  # /!\ changer noms de fichiers
positions = ['UR', 'SP', 'UD']
names = ['LH', 'GD', 'PDs', 'MH']
positionsdico = {
    "SP": "Supine",
    "UD": "UpsidDowm",
    "UR": "UpRight"
}
for name in names:
    file1 = open("statsdeltay_%s" % name, "w")
    for p in positions:
        file1.write("######################## Position {} ###############################################\n".format(positionsdico[p]))
        for n in ntrials:
            file1.write("Stats DeltaY pour %s en position %s et les essais %d et %d\n"%(name,p,n,n+1))
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
                ecart1.append(abs(np.nanmax(pos1[1][cycle_starts[k]:cycle_ends[k]]) - np.nanmin(
                    pos1[1][cycle_starts[k]:cycle_ends[k]])))

            # seconde file
            file_path2 = "../../data/Groupe_1_codas/%s_%s_coda000%d.txt" % (name, p, n + 1)
            if not path.exists(file_path2):
                continue
            coda_df2 = coda.import_data(file_path2)
            pos2 = coda.manipulandum_center(coda_df2, markers_id)
            pos2 = pos2 / 1000
            vel2 = tool.derive(pos2, 200, axis=1)
            # posxbis2 = pos2[0][~np.isnan(pos2[0])]
            pk = signal.find_peaks(vel2[0], prominence=1, width=(100, 1000))
            ipk = pk[0]  # index
            cycle_starts = ipk[:-1]
            cycle_ends = ipk[1:] - 1

            ecart2 = []
            for k in range(len(cycle_starts)):
                ecart2.append(abs(np.nanmax(pos2[1][cycle_starts[k]:cycle_ends[k]]) - np.nanmin(
                    pos2[1][cycle_starts[k]:cycle_ends[k]])))

            Txbis, pvalbis = sc.bartlett(ecart1, ecart2)
            file1.write("p_value pour la variance %f \n" % pvalbis)
            file1.write("les deux variances sont %f et %f\n" %(np.nanstd(ecart1),np.nanstd(ecart2)))
            #print(sc.ttest_ind(ecart1, ecart2, equal_var=True))

            X1 = sm2.DescrStatsW(ecart1)
            X2 = sm2.DescrStatsW(ecart2)

            Ttest = sm2.CompareMeans(X1, X2)

            # Ttest.summary() gives us the p-value, the statistics and the critical values
            file1.write(Ttest.summary(usevar='pooled').as_text()+"\n")
            file1.write("les deux moyennes sont: %f et %f\n"%(np.nanmean(ecart1),np.nanmean(ecart2)))
            file1.write('\n')
            file1.write('\n')
    file1.close()
