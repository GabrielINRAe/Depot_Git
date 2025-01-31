# Import des bibliothèques nécessaires
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import geopandas as gpd
import pandas as pd
import sys
import os
from osgeo import gdal, gdal_array
from shapely.geometry import Point
import seaborn as sns
# Ajout du chemin vers les fonctions personnalisées
sys.path.append('/home/onyxia/work/Depot_Git/scripts')
from my_function import main, get_samples_from_roi, calcul_distance
sys.path.append('/home/onyxia/work/libsigma')
import classification as cla

# Définition des paramètres 
my_folder = '/home/onyxia/work/results/data'
in_vector = os.path.join(
    my_folder,
    'sample/Sample_BD_foret_T31TCJ.shp')
raster_name = os.path.join(
    my_folder,
    'img_pretraitees/Serie_temp_S2_allbands.tif')
out_image = os.path.splitext(in_vector)[0] + '_v2.tif'
field_name = 'Code' 
violin_plot_path = os.path.join(
    my_folder,
    "../figure/violin_plot_dist_centroide_by_poly_by_class.png")
baton_plot_path = os.path.join(
    my_folder,
    "../figure/diag_baton_dist_centroide_classe.png")

# for those parameters, you know how to get theses information if you had to
sptial_resolution = 10
xmin = 501127.9696999999
ymin = 6240654.023599998
xmax = 609757.9696999999
ymax = 6314464.023599998

# Lecture du shp
gdf = gpd.read_file(in_vector)

# Création d'un dictionnaire pour assigner une valeur d'ID
unique_ids = gdf['ID'].unique()
id_to_int = {id_: idx for idx, id_ in enumerate(unique_ids)}
int_to_id = {v: k for k, v in id_to_int.items()}

# Ajout ID au shp
gdf['ID_num'] = gdf['ID'].map(id_to_int)

# Save du shp temporaire
temp_vector = os.path.splitext(in_vector)[0] + '_temp.shp'
gdf.to_file(temp_vector)

# Champs a rasteriser
field_class = 'Code'  # Classe
field_id = 'ID_num'  # ID des polygones

# Définitions des noms pour la sauvegarde
out_image_class = os.path.splitext(in_vector)[0] + '_class.tif'
out_image_id = os.path.splitext(in_vector)[0] + '_id.tif'

# Rasterisation des classes
cmd_class = (
    "gdal_rasterize -a {field_class} "
    "-tr {sptial_resolution} {sptial_resolution} "
    "-te {xmin} {ymin} {xmax} {ymax} -ot Byte -of GTiff "
    "{in_vector} {out_image_class}"
).format(
    in_vector=in_vector, field_class=field_class,
    sptial_resolution=sptial_resolution,
    xmin=xmin, ymin=ymin, xmax=xmax, 
    ymax=ymax, out_image_class=out_image_class
)

# Rasterisation des ID
cmd_id = (
    "gdal_rasterize -a {field_id} "
    "-tr {sptial_resolution} {sptial_resolution} "
    "-te {xmin} {ymin} {xmax} {ymax} -ot Int32 -of GTiff "
    "{temp_vector} {out_image_id}"
).format(
    temp_vector=temp_vector, field_id=field_id,
    sptial_resolution=sptial_resolution,
    xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, out_image_id=out_image_id
)

# Execution des codes
os.system(cmd_class)
os.system(cmd_id)

# Chargement des images
gdf_filtered = main(in_vector, raster_name, out_image_class, out_image_id)

band_columns = [f'band_{i}' for i in range(1, 61)]
# Calcul du centroïde spectral
centroide_spectral = gdf_filtered.groupby(['class', 'polygon_id'])\
    [band_columns].mean().reset_index()
# Merge entre gdf_filtré et les centroides
gdf_merged = gdf_filtered.merge(
    centroide_spectral,
    on=['class', 'polygon_id'],
    suffixes=('', '_centroid'))

# Distance entre chaque bande et son centroide
for band in band_columns:
    gdf_merged[f'diference_{band}'] = gdf_merged[band] -\
        gdf_merged[f'{band}_centroid']

# Reset de l'indice
gdf_merged = gdf_merged.reset_index(drop=True)
# Groupe par classe et id de polygon
gdf_merged = gdf_merged.groupby(['class', 'polygon_id'], as_index=False).\
    apply(lambda group: calcul_distance(group, band_columns))
# Calcule la distance moyenne par classe et polygon_id
distance_moyenne = gdf_merged.\
    groupby(['class', 'polygon_id'])["distance_euclidienne"].mean()

distance_moyenne_par_classe = gdf_merged.groupby('class')\
    ['distance_euclidienne'].mean().reset_index()

## Création des graphiques

# Calcul de la distance moyenne au centroïde par classe
distance_moyenne_par_classe = gdf_merged.groupby('class')\
    ['distance_euclidienne'].mean().reset_index()
# Définir une palette de couleurs pour chaque classe
palette = sns.color_palette("husl", len(distance_moyenne_par_classe))
# Création du bar plot
plt.figure(figsize=(12, 6))
sns.barplot(
    x='class',
    y='distance_euclidienne',
    data=distance_moyenne_par_classe,
    palette=palette)
# Ajout des étiquettes et du titre
plt.xlabel('Classe', fontsize=12)
plt.ylabel('Distance Moyenne au Centroïde', fontsize=12)
plt.title('Distance Moyenne des Pixels au Centroïde par Classe', fontsize=14)
# Création de la légende
handles = [plt.Rectangle((0,0),1 ,1 , color=palette[i])\
    for i in range(len(distance_moyenne_par_classe))]
labels = distance_moyenne_par_classe['class'].astype(str).tolist()
plt.legend(
    handles,
    labels,
    title="Classes",
    bbox_to_anchor=(1.05, 1),
    loc='upper left')
plt.savefig(baton_plot_path, dpi=300)
plt.close()

# Conversion du resultat de distance moyenne en dataframe
distance_moyenne_df = distance_moyenne.reset_index()
# Création d'un graphe violin_plot
plt.figure(figsize=(12, 8))
sns.violinplot(
    x='class',
    y='distance_euclidienne',
    data=distance_moyenne_df,
    inner='quart',
    palette="Set2")
# Modification des titres et étiquettes
plt.title('Distance Moyenne des Pixels au Centroïde par Classe', fontsize=16)
plt.xlabel('Clase', fontsize=12)
plt.ylabel('Distance Moyenne au Centroïde', fontsize=12)
plt.savefig(violin_plot_path, dpi=300)
plt.close()

print(f"Graphiques enregistrés :\n - \
{violin_plot_path}\n - {baton_plot_path}")