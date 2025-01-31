import os
import sys
import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score
)

sys.path.append('/home/onyxia/work/libsigma')
import read_and_write as rw
import classification as cla
import plots

sys.path.append('/home/onyxia/work/Depot_Git/scripts')
from my_function import (
    sel_classif_pixel,
    report_from_dict_to_df,
    supprimer_dossier_non_vide,
    rasterization,
    id_construction,
    stratified_grouped_validation,
    save_classif
)

# Création du dossier de sauvegarde temporaire et paramètres
racine = "/home/onyxia/work"
output_dir = os.path.join(racine, "output_classif")
os.makedirs(output_dir, exist_ok=True)

path_sample = os.path.join(racine, "results/data/sample/Sample_BD_foret_T31TCJ.shp")
path_sample_px = os.path.join(output_dir, "sample_classif_px.shp")

sample_rasterized = os.path.join(output_dir, "rasterized_sample.tif")
path_image_3b = os.path.join(racine, "results/data/img_pretraitees/Serie_temp_S2_3_bands.tif")
path_image_10b = os.path.join(racine, "results/data/img_pretraitees/Serie_temp_S2_10_bands.tif")
path_image_allbands = os.path.join(racine, "results/data/img_pretraitees/Serie_temp_S2_allbands.tif")

path_sample_px_centroid = os.path.join(output_dir, "sample_px_centroid.shp")
path_sample_px_id = os.path.join(output_dir, "sample_px_id.shp")
path_rasterized_sample_id = os.path.join(output_dir, "rasterized_sample_id.tif")

out_classif = os.path.join(racine, "results/data/classif/carte_essences_echelle_pixel.tif")

# Sauvegarde d'un vecteur échantillons avec seulement les données pour classification pixel
sample = gpd.read_file(path_sample)
sample_px = sel_classif_pixel(sample[['Code', "geometry"]])
sample_px.to_file(path_sample_px)

# Rasterisation des échantillons pour la classification pixel
rasterization(
    in_vector=path_sample_px,
    out_image=sample_rasterized,
    field_name='Code',
    ref_image=path_image_allbands,
    dtype='Byte'
)

# Construction du raster id pour la méthode stratifiée groupée
id_construction(sample_px, path_sample_px_id)
rasterization(
    in_vector=path_sample_px_id,
    out_image=path_rasterized_sample_id,
    field_name="id",
    ref_image=path_image_allbands
)

# Entrainement du modèle et validation stratifiée groupée
id_filename = path_rasterized_sample_id
nb_iter = 30
nb_folds = 5
rfc, list_cm, list_accuracy, list_report, Y_predict = stratified_grouped_validation(
    nb_iter=nb_iter,
    nb_folds=nb_folds,
    sample_filename=sample_rasterized,
    image_filename=path_image_allbands,
    id_filename=id_filename
)

# Stratégie d'évitement : suppression des tableaux n'ayant pas toutes les classes
list_report_2 = [report for report in list_report if len(report.keys()) == 9]
print(len(list_report_2))

list_cm_2 = [cm for cm in list_cm if len(cm) == 9]
print(len(list_cm_2))

# Plots
suffix = '_CV{}folds_stratified_group_x{}times'.format(nb_folds, nb_iter)
out_figs_dir = os.path.join(racine, "Depot_Git/results/figure")
out_matrix = os.path.join(out_figs_dir, 'matrice{}.png'.format(suffix))
out_qualite = os.path.join(out_figs_dir, 'qualites{}.png'.format(suffix))

# Calcul de la moyenne de la matrice de confusion
array_cm = np.array(list_cm_2)
mean_cm = array_cm.mean(axis=0)

# Calcul de la moyenne et de l'écart-type de l'accuracy
array_accuracy = np.array(list_accuracy)
mean_accuracy = array_accuracy.mean()
std_accuracy = array_accuracy.std()

# Calcul de la moyenne et de l'écart-type du rapport de classification
array_report = np.array(list_report_2)
mean_report = array_report.mean(axis=0)
std_report = array_report.std(axis=0)
a_report = list_report_2[0]
mean_df_report = pd.DataFrame(mean_report, index=a_report.index, columns=a_report.columns)
std_df_report = pd.DataFrame(std_report, index=a_report.index, columns=a_report.columns)

# Affichage de la matrice de confusion
plots.plot_cm(mean_cm, np.unique(Y_predict))
plt.savefig(out_matrix, bbox_inches='tight')

# Affichage des métriques de classification
fig, ax = plt.subplots(figsize=(10, 7))
ax = mean_df_report.T.plot.bar(ax=ax, yerr=std_df_report.T, zorder=2)
ax.set_ylim(0.5, 1)
ax.text(1.5, 0.95, 'OA : {:.2f} +- {:.2f}'.format(mean_accuracy, std_accuracy), fontsize=14)
ax.set_title('Class quality estimation')

# Personnalisation du graphique
ax.set_facecolor('ivory')
ax.set_xlabel(ax.get_xlabel(), fontdict={'fontname': 'Sawasdee'}, fontsize=14)
ax.set_ylabel(ax.get_ylabel(), fontdict={'fontname': 'Sawasdee'}, fontsize=14)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.tick_params(axis='x', colors='darkslategrey', labelsize=14)
ax.tick_params(axis='y', colors='darkslategrey', labelsize=14)
ax.yaxis.grid(which='major', color='darkgoldenrod', linestyle='--', linewidth=0.5, zorder=1)
ax.yaxis.grid(which='minor', color='darkgoldenrod', linestyle='-.', linewidth=0.3, zorder=1)
plt.savefig(out_qualite, bbox_inches='tight')

save_classif(
    image_filename=path_image_allbands,
    model=rfc,
    out_classif=out_classif
)
