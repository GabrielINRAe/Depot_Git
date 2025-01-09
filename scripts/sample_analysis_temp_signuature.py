# Importation des bibliothèques nécessaires
import os
from osgeo import gdal, ogr
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from my_function import read_ndvi_with_gdal, calculate_ndvi_stats, plot_ndvi_signatures

# Définition des paramètres
my_folder = '/home/onyxia/work'
ndvi_file = os.path.join(my_folder, 'Depot_Git/results/data/img_pretraitees/Serie_temp_S2_ndvi.tif')
in_vector = os.path.join(my_folder, 'Depot_Git/results/data/sample/Sample_BD_foret_T31TCJ.shp')
output_file = os.path.join(my_folder, 'Depot_Git/results/figure/temp_mean_ndvi.png')

# Chargement des couches des échantillons sélectionnés
samples = gpd.read_file(in_vector)

# Liste des classes d'intérêt
classes = [
    "Chêne", "Robinier", "Peupleraie", "Douglas", "Pin laricio ou pin noir", "Pin maritime"
]

# Filtrage des échantillons ayant une classe incluse dans la liste des classes d'intérêt
filtered_samples = samples[samples['Nom'].isin(classes)]
print(filtered_samples)

# Chargement du raster de NDVI avec GDAL
ndvi_data, dates = read_ndvi_with_gdal(ndvi_file)

spatial_resolution = 10

# Calcul des statistiques NDVI par classe
ndvi_stats = calculate_ndvi_stats(
    ndvi_data, filtered_samples, class_column='Code', geometry_column='geometry',
    spatial_resolution=spatial_resolution, reference_raster=ndvi_file
)

# Tracer les courbes des signatures temporelles
plot_ndvi_signatures(ndvi_stats, output_file)