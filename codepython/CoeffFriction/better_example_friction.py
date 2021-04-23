"""
Created on Wed Marcj 18
Exemple de script permettant de calculer la valeur du coefficient de friction 
d'un sujet en fonction de la force normale qu'il applique. Concretement, on
calcule les valeurs de k et n dans la formule:
    
    mu=k*NF^(n-1)

Ceci permet de calculer la slip force et la marge de securite

Dans ce script, on genere aussi les graphes montrant les differents moments du
glissement detectes ainsi que les graphes montrant mu en fonction de NF.

@author: fschiltz
"""
# Importation des differentes libraries necessaires
import numpy as np
import matplotlib.pyplot as plt
import math

import glm_data_processing as glm
import get_mu_points as gmp
import get_mu_fit as gmf

# Fermeture des figures ouvertes
plt.close('all')

# Chemins d'acces aux fichiers (A MODIFIER)
filepathss = [
    ["../../data/CF_data/GD_pre_001.glm", "../../data/CF_data/GD_pre_002.glm", "../../data/CF_data/GD_pre_003.glm"],
    ["../../data/CF_data/GD_post_001.glm", "../../data/CF_data/GD_post_002.glm", "../../data/CF_data/GD_post_003.glm"]]

color1 = "orange"
color2 = "blue"
values = np.zeros((2, 4))
index=[[],[]]
tup = [('pre', 'GD'), ('post', 'GD')]
# Initialisation des structures servant à stocker les vakeurs de TF/NF et NF
# aux moments des glissements
for filepaths, tpl in zip(filepathss, tup):
    all_mu_points_thumb = []
    all_NF_thumb = []
    all_mu_points_index = []
    all_NF_index = []

    # %% Filtrage des donnees
    freqAcq = 800  # Frequence d'acquisition des donnees
    freqFiltForces = 20;  # Frequence de coupure du filtrage des forces (peut etre modifié)

    for filepath in filepaths:
        glm_df = glm.import_data(filepath)
        for i in range(21, 43):
            glm_df.iloc[:, i] = glm.filter_signal(glm_df.iloc[:, i], freqAcq, freqFiltForces)

        # %% Extraction des signaux et mise a zero des forces en soustrayant la valeur
        # moyenne des 500 premieres ms
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

        # %%Compute COP manually
        Fal = -np.array([glm_df.loc[:, 'Fxal'], glm_df.loc[:, 'Fyal'], glm_df.loc[:, 'Fzal']])
        Far = -np.array([glm_df.loc[:, 'Fxar'], glm_df.loc[:, 'Fyar'], glm_df.loc[:, 'Fzar']])
        Tal = -np.array([glm_df.loc[:, 'Txal'], glm_df.loc[:, 'Tyal'], glm_df.loc[:, 'Tzal']])
        Tar = -np.array([glm_df.loc[:, 'Txar'], glm_df.loc[:, 'Tyar'], glm_df.loc[:, 'Tzar']])

        baseline = range(0, 400)
        Fal = np.subtract(Fal, np.nanmean(Fal[:, baseline], 1).reshape((3, 1)))
        Far = np.subtract(Far, np.nanmean(Far[:, baseline], 1).reshape((3, 1)))
        Tal = np.subtract(Tal, np.nanmean(Tal[:, baseline], 1).reshape((3, 1)))
        Tar = np.subtract(Tar, np.nanmean(Tar[:, baseline], 1).reshape((3, 1)))

        z0 = 0.00155;
        COPthumb = -np.array([(Tal[1, :] + Fal[0, :] * z0) / Fal[2, :], -(Tal[0, :] - Fal[1, :] * z0) / Fal[2, :]])
        COPindex = -np.array([(Tar[1, :] + Far[0, :] * z0) / Far[2, :], -(Tar[0, :] - Far[1, :] * z0) / Far[2, :]])

        COPthumb_g = -COPthumb[0, :] * math.sin(0.523599) - COPthumb[1, :] * math.cos(0.523599)
        COPindex_g = COPindex[0, :] * math.sin(0.523599) + COPindex[1, :] * math.cos(0.523599)

        # %% Calcul du coefficient de friction au moment du glissement. Voir fonction
        # get_mu_points et papier "Barrea et al. 2016" pour plus de détails
        mu_thumb, slip_indexes_thumb, start_search_zones_thumb, end_search_zones_thumb = \
            gmp.get_mu_points(COPthumb_g, TFz_thumb, TFx_thumb, NF_thumb)

        all_mu_points_thumb = [*all_mu_points_thumb, *mu_thumb]
        all_NF_thumb = [*all_NF_thumb, *NF_thumb[slip_indexes_thumb]]

        mu_index, slip_indexes_index, start_search_zones_index, end_search_zones_index = \
            gmp.get_mu_points(COPindex_g, TFz_index, TFx_index, NF_index)

        all_mu_points_index = [*all_mu_points_index, *mu_index]
        all_NF_index = [*all_NF_index, *NF_index[slip_indexes_index]]
        if tpl[0]=='pre':
            index[0].append(all_NF_thumb)
            index[0].append(all_mu_points_thumb)
            index[0].append(all_NF_index)
            index[0].append(all_mu_points_index)
        else:
            index[1].append(all_NF_thumb)
            index[1].append(all_mu_points_thumb)
            index[1].append(all_NF_index)
            index[1].append(all_mu_points_index)


        # %% Graphe de détection du glissement pour l'index
        siz = len(start_search_zones_index)
        time = glm_df.loc[:, 'time'].to_numpy()
        fig = plt.figure(figsize=[15, 7])
        ax = fig.subplots(3, 1)
        ax[0].set_title("Index", fontsize=14, fontweight="bold")
        ax[0].plot(time, COPindex_g * 1000)
        ax[0].set_ylabel("COP [mm]", fontsize=13)
        ax[0].set_ylim([-25, 25])

        ax[1].plot(time, TFx_index, label="TFv")
        ax[1].plot(time, NF_index, label="NF")
        ax[1].legend(fontsize=12)
        ax[1].set_ylabel("Forces [N]", fontsize=13)

        ax[2].plot(time, np.divide(np.hypot(TFx_index, TFz_index), NF_index))
        ax[2].plot(time[slip_indexes_index], mu_index, marker='.', linestyle='')
        ax[2].set_ylim([0, max(mu_index) + 0.2])
        ax[2].set_ylabel("TF/NF [-]", fontsize=13)
        ax[2].set_xlabel("Time [s]", fontsize=13)

        for i in range(0, siz):
            rect0 = plt.Rectangle((time[start_search_zones_index[i]], ax[0].get_ylim()[0]),
                                  time[end_search_zones_index[i] - start_search_zones_index[i]],
                                  ax[0].get_ylim()[1] - ax[0].get_ylim()[0], color='k', alpha=0.3)
            rect1 = plt.Rectangle((time[start_search_zones_index[i]], ax[1].get_ylim()[0]), \
                                  time[end_search_zones_index[i] - start_search_zones_index[i]], \
                                  ax[1].get_ylim()[1] - ax[1].get_ylim()[0], color='k', alpha=0.3)
            rect2 = plt.Rectangle((time[start_search_zones_index[i]], ax[2].get_ylim()[0]), \
                                  time[end_search_zones_index[i] - start_search_zones_index[i]], \
                                  ax[2].get_ylim()[1] - ax[2].get_ylim()[0], color='k', alpha=0.3)
            ax[0].add_patch(rect0)
            ax[1].add_patch(rect1)
            ax[2].add_patch(rect2)

        # %% Graphe de détection du moment du glissement pour l'essai "thumb strong"
        siz = len(start_search_zones_thumb)
        time = glm_df.loc[:, 'time'].to_numpy()
        fig = plt.figure(figsize=[15, 7])
        ax = fig.subplots(3, 1)
        ax[0].set_title("Thumb", fontsize=14, fontweight="bold")
        ax[0].plot(time, COPthumb_g * 1000)
        ax[0].set_ylabel("COP [mm]", fontsize=13)
        ax[0].set_ylim([-25, 25])

        ax[1].plot(time, TFx_thumb, label="TFv")
        ax[1].plot(time, NF_thumb, label="NF")
        ax[1].legend(fontsize=12)
        ax[1].set_ylabel("Forces [N]", fontsize=13)

        ax[2].plot(time, np.divide(np.hypot(TFx_thumb, TFz_thumb), NF_thumb))
        ax[2].plot(time[slip_indexes_thumb], mu_thumb, marker='.', linestyle='')
        ax[2].set_ylim([0, max(mu_thumb) + 0.2])
        ax[2].set_ylabel("TF/NF [-]", fontsize=13)
        ax[2].set_xlabel("Time [s]", fontsize=13)

        for i in range(0, siz):
            rect0 = plt.Rectangle((time[start_search_zones_thumb[i]], ax[0].get_ylim()[0]),
                                  time[end_search_zones_thumb[i] - start_search_zones_thumb[i]],
                                  ax[0].get_ylim()[1] - ax[0].get_ylim()[0], color='k', alpha=0.3)
            rect1 = plt.Rectangle((time[start_search_zones_thumb[i]], ax[1].get_ylim()[0]), \
                                  time[end_search_zones_thumb[i] - start_search_zones_thumb[i]], \
                                  ax[1].get_ylim()[1] - ax[1].get_ylim()[0], color='k', alpha=0.3)
            rect2 = plt.Rectangle((time[start_search_zones_thumb[i]], ax[2].get_ylim()[0]), \
                                  time[end_search_zones_thumb[i] - start_search_zones_thumb[i]], \
                                  ax[2].get_ylim()[1] - ax[2].get_ylim()[0], color='k', alpha=0.3)
            ax[0].add_patch(rect0)
            ax[1].add_patch(rect1)
            ax[2].add_patch(rect2)

    # %% Calcul des valeurs de k et n pour l'index
    k_index, n_index = gmf.get_mu_fit(all_mu_points_index, all_NF_index)

    # %% Calcul des valeurs de k et n pour le pouce
    k_thumb, n_thumb = gmf.get_mu_fit(all_mu_points_thumb, all_NF_thumb)
    if tpl[0]=='pre':
        values[0,:]=np.array([k_index, n_index, k_thumb, n_thumb])
    else:
        values[1,:] = np.array([k_index, n_index, k_thumb, n_thumb])
    # %% Impression de la valeur des variables dans la console
    print("Index: the value of k is %f and n is %f" % (k_index, n_index))
    print("Thumb: the value of k is %f and n is %f" % (k_thumb, n_thumb))

    # %% Figure finale pour l'index
x = np.arange(0.2, 30, 0.02).tolist()
fig = plt.figure(figsize=[15, 7])
plt.plot(x, values[0][0] * (x ** (values[0][1] - 1)),color=color1,label='index pre')
plt.plot(index[0][2], index[0][3], linestyle='', marker='.',color=color1)
plt.plot(x, values[0][2] * (x ** (values[0][3] - 1)),color=color1,linestyle='dashed',label='thumb pre')
plt.plot(index[0][0], index[0][1], linestyle='', marker='*',color=color1)
plt.plot(x, values[1][0] * (x ** (values[1][1] - 1)),color=color2,label='index post')
plt.plot(index[1][2], index[1][3], linestyle='', marker='.',color=color2)
plt.plot(x, values[1][2] * (x ** (values[1][3] - 1)),color=color2,linestyle='dashed',label='thumb post')
plt.plot(index[1][0], index[1][1], linestyle='', marker='*',color=color2)
plt.ylim([0, 3])
plt.xlim([0, 30])
plt.title('Coefficient of friction index')
plt.xlabel('Normal Force [N]')
plt.ylabel('Static Friction [-]')
plt.legend()
plt.show()


