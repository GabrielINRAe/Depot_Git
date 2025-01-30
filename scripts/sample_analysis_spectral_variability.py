import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from osgeo import ogr, gdal, gdal_array

# Ajout du chemin vers les fonctions personnalisées
sys.path.append('/home/onyxia/work/libsigma')
import classification as cla
sys.path.append('/home/onyxia/work/Depot_Git/scripts')
from my_function import main, get_samples_from_roi, calcul_distance

# Définition des paramètres
my_folder = '/home/onyxia/work/results'
in_vector = os.path.join(my_folder, 'data/sample/Sample_BD_foret_T31TCJ.shp')
raster_name = os.path.join(my_folder, 'data/img_pretraitees/Serie_temp_S2_allbands.tif')
out_image = os.path.splitext(in_vector)[0] + '_v2.tif'
field_name = 'Code'
violin_plot_path = os.path.join(my_folder, "figure/violin_plot_dist_centroide_by_poly_by_class.png")
baton_plot_path = os.path.join(my_folder, "figure/diag_baton_dist_centroide_classe.png")

sptial_resolution = 10
xmin, ymin, xmax, ymax = 501127.9697, 6240654.0236, 609757.9697, 6314464.0236

# Lecture du shapefile
gdf = gpd.read_file(in_vector)

# Création des ID numériques uniques
unique_ids = gdf['ID'].unique()
id_to_int = {id_: idx for idx, id_ in enumerate(unique_ids)}
gdf['ID_num'] = gdf['ID'].map(id_to_int)

# Sauvegarde temporaire du shapefile
temp_vector = os.path.splitext(in_vector)[0] + '_temp.shp'
gdf.to_file(temp_vector)

# Définition des fichiers de sortie
out_image_class = os.path.splitext(in_vector)[0] + '_class.tif'
out_image_id = os.path.splitext(in_vector)[0] + '_id.tif'

# Commandes de rasterisation
cmd_class = (
    f"gdal_rasterize -a {field_name} "
    f"-tr {sptial_resolution} {sptial_resolution} "
    f"-te {xmin} {ymin} {xmax} {ymax} -ot Byte -of GTiff "
    f"{in_vector} {out_image_class}"
)
cmd_id = (
    f"gdal_rasterize -a ID_num "
    f"-tr {sptial_resolution} {sptial_resolution} "
    f"-te {xmin} {ymin} {xmax} {ymax} -ot Int32 -of GTiff "
    f"{temp_vector} {out_image_id}"
)

# Exécution des commandes
os.system(cmd_class)
os.system(cmd_id)

# Chargement des images
gdf_filtered = main(in_vector, raster_name, out_image_class, out_image_id)

band_columns = [f'band_{i}' for i in range(1, 61)]
centroides_espectrales = gdf_filtered.groupby(['class', 'polygon_id'])[band_columns].mean().reset_index()
gdf_merged = gdf_filtered.merge(centroides_espectrales, on=['class', 'polygon_id'], suffixes=('', '_centroid'))

# Calcul des différences par bande
for band in band_columns:
    gdf_merged[f'diferencia_{band}'] = gdf_merged[band] - gdf_merged[f'{band}_centroid']

gdf_merged = gdf_merged.reset_index(drop=True)
gdf_merged = gdf_merged.groupby(['class', 'polygon_id'], as_index=False).apply(
    lambda group: calcul_distance(group, band_columns)
)

distance_moyenne_par_classe = gdf_merged.groupby('class')['distance_euclidienne'].mean().reset_index()

# Création des graphiques
plt.figure(figsize=(12, 8))
sns.violinplot(x='class', y='distance_euclidienne', data=distance_moyenne_par_classe, inner='quart', palette="Set2")
plt.title('Distribution de la Distance Euclidienne par Classe et Polygone')
plt.xlabel('Classe')
plt.ylabel('Distance Euclidienne')
plt.tight_layout()
plt.savefig(violin_plot_path, dpi=300)
plt.close()

plt.figure(figsize=(12, 6))
sns.barplot(x='class', y='distance_euclidienne', data=distance_moyenne_par_classe, palette="husl")
plt.xlabel('Classe')
plt.ylabel('Distance Moyenne au Centroïde')
plt.title('Distance Moyenne des Pixels au Centroïde par Classe')
plt.tight_layout()
plt.savefig(baton_plot_path, dpi=300)
plt.close()

print(f"Graphiques enregistrés :\n - {violin_plot_path}\n - {baton_plot_path}")