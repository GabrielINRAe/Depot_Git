# Import des bibliothèques nécessaires
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
import geopandas as gpd
import pandas as pd
import sys
import seaborn as sns
from osgeo import ogr, gdal, gdal_array
import geopandas as gpd
from shapely.geometry import Point
# Ajout du chemin vers les fonctions personnalisées
sys.path.append('/home/onyxia/work/libsigma')
import classification as cla
sys.path.append('/home/onyxia/work/Depot_Git/scripts')
from my_function import main, get_samples_from_roi, calcul_distance

# Définition des paramètres 
my_folder = '/home/onyxia/work/results/data'
in_vector = os.path.join(my_folder, 'sample/Sample_BD_foret_T31TCJ.shp')
#raster_name = os.path.join(my_folder, 'img_pretraitees/Serie_temp_S2_allbands.tif')
out_image = os.path.splitext(in_vector)[0] + '_v2.tif'
field_name = 'Code'  # field containing the numeric label of the classes
violin_plot_path = os.path.join(my_folder, "figure/violin_plot_dist_centroide_by_poly_by_class.png")
baton_plot_path = os.path.join(my_folder, "figure/diag_baton_dist_centroide_classe.png")
# for those parameters, you know how to get theses information if you had to
sptial_resolution = 10
xmin = 501127.9696999999
ymin = 6240654.023599998
xmax = 609757.9696999999
ymax = 6314464.023599998

# Leer el shapefile
in_vector = os.path.join(my_folder, 'sample/Sample_BD_foret_T31TCJ.shp')
gdf = gpd.read_file(in_vector)

# Crear un diccionario que asigne un valor numérico único a cada ID
unique_ids = gdf['ID'].unique()
id_to_int = {id_: idx for idx, id_ in enumerate(unique_ids)}
int_to_id = {v: k for k, v in id_to_int.items()}

# Actualizar el shapefile temporalmente con los valores numéricos
gdf['ID_num'] = gdf['ID'].map(id_to_int)

# Guardar el shapefile temporal
temp_vector = os.path.splitext(in_vector)[0] + '_temp.shp'
gdf.to_file(temp_vector)

# Define los campos que vas a rasterizar
field_class = 'Code'  # campo que contiene la clase
field_id = 'ID_num'  # campo que contiene el ID numérico del polígono

# Define los nombres de los archivos de salida
out_image_class = os.path.splitext(in_vector)[0] + '_class.tif'
out_image_id = os.path.splitext(in_vector)[0] + '_id.tif'

# Comando para rasterizar el campo de las clases
cmd_class = (
    "gdal_rasterize -a {field_class} "
    "-tr {sptial_resolution} {sptial_resolution} "
    "-te {xmin} {ymin} {xmax} {ymax} -ot Byte -of GTiff "
    "{in_vector} {out_image_class}"
).format(
    in_vector=in_vector, field_class=field_class, sptial_resolution=sptial_resolution,
    xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, out_image_class=out_image_class
)

# Comando para rasterizar el campo de los IDs numéricos de los polígonos
cmd_id = (
    "gdal_rasterize -a {field_id} "
    "-tr {sptial_resolution} {sptial_resolution} "
    "-te {xmin} {ymin} {xmax} {ymax} -ot Int32 -of GTiff "
    "{temp_vector} {out_image_id}"
).format(
    temp_vector=temp_vector, field_id=field_id, sptial_resolution=sptial_resolution,
    xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, out_image_id=out_image_id
)

# Ejecuta los comandos en la terminal
os.system(cmd_class)
os.system(cmd_id)

image_filename = os.path.join(my_folder, 'img_pretraitees/Serie_temp_S2_allbands.tif')
sample_filename = os.path.join(my_folder, 'sample/Sample_BD_foret_T31TCJ_class.tif')
id_image_filename = os.path.join(my_folder, 'sample/Sample_BD_foret_T31TCJ_id.tif')

gdf_filtered = main(in_vector, image_filename, sample_filename, id_image_filename)

print(gdf_filtered.head())

band_columns = [f'band_{i}' for i in range(1, 61)]
# Asegúrate de que los centroides estén calculados correctamente
centroides_espectrales = gdf_filtered.groupby(['class', 'polygon_id'])[band_columns].mean().reset_index()

# Realizar el merge entre gdf_filtered y centroides_espectrales
gdf_merged = gdf_filtered.merge(centroides_espectrales, on=['class', 'polygon_id'], suffixes=('', '_centroid'))


# Calcular la diferencia entre cada banda y su respectivo centroide
for band in band_columns:
    gdf_merged[f'diferencia_{band}'] = gdf_merged[band] - gdf_merged[f'{band}_centroid']

# Primero, reseteamos el índice para evitar que 'class' y 'polygon_id' se usen como índice
gdf_merged = gdf_merged.reset_index(drop=True)

# Agrupar por 'class' y 'polygon_id' y aplicar la función
gdf_merged = gdf_merged.groupby(['class', 'polygon_id'], as_index=False).apply(lambda group: calcular_distancia(group, band_columns))

# Calcular la distancia promedio por class y polygon_id
distancia_promedio = gdf_merged.groupby(['class', 'polygon_id'])['distancia_euclidiana'].mean()

# Mostrar el resultado
print(distancia_promedio)

# Convertir el resultado de distancia_promedio a un DataFrame
distancia_promedio_df = distancia_promedio.reset_index()

# Crear un gráfico de violín
plt.figure(figsize=(12, 8))
sns.violinplot(x='class', y='distancia_euclidiana', data=distancia_promedio_df, inner='quart', palette="Set2")

# Ajustar etiquetas y título
plt.title('Distribución de la Distancia Euclidiana Promedio por Clase y Polígono', fontsize=16)
plt.xlabel('Clase', fontsize=12)
plt.ylabel('Distancia Euclidiana Promedio', fontsize=12)

# Mostrar el gráfico
plt.tight_layout()
plt.show()

# Sauvegarde du graphique
plt.savefig(violin_plot_path, dpi=300)
plt.close()

# Calcul de la distance moyenne au centroïde par classe
distance_moyenne_par_classe = gdf_merged.groupby('class')['distancia_euclidiana'].mean().reset_index()

# Définir une palette de couleurs pour chaque classe
palette = sns.color_palette("husl", len(distance_moyenne_par_classe))

# Création du bar plot
plt.figure(figsize=(12, 6))
sns.barplot(x='class', y='distancia_euclidiana', data=distance_moyenne_par_classe, palette=palette)

# Ajout des étiquettes et du titre
plt.xlabel('Classe', fontsize=12)
plt.ylabel('Distance Moyenne au Centroïde', fontsize=12)
plt.title('Distance Moyenne des Pixels au Centroïde par Classe', fontsize=14)

# Création de la légende
handles = [plt.Rectangle((0, 0), 1, 1, color=palette[i]) for i in range(len(distance_moyenne_par_classe))]
labels = distance_moyenne_par_classe['class'].astype(str).tolist()
plt.legend(handles, labels, title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')

# Affichage du graphique
plt.tight_layout()
plt.show()

# Sauvegarde du graphique
plt.savefig(baton_plot_path, dpi=300)
plt.close()

print(f"Graphiques enregistrés :\n - {violin_plot_path}\n - {baton_plot_path}")