# Pour charger vous images dans le workspace tapez la commande en changeant votre nom d'utilisateur:
# mc cp -r s3/nom_utilisateur/diffusion/images /home/onyxia/work/data

import geopandas as gpd
import rasterio
from osgeo import gdal, ogr, osr
import numpy as np
import pandas as pd
from rasterio.features import rasterize
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import os


def filter_classes(dataframe, valid_classes):
    """
    Filtre les classes de la BD Forêt 
    """
    return dataframe[dataframe['TFV'].isin(valid_classes)]


def count_polygons_by_class(dataframe, class_column='classif_objet'):
    """
    Compte le nombre de polygones par classe.
    """
    return dataframe.groupby(class_column).size().reset_index(name='count')


def count_pixels_by_class(dataframe, raster_path, class_column='classif_pixel'):
    """
    Compte le nombre de pixels par classe dans un raster donné.
    """
    with rasterio.open(raster_path) as src:
        data = src.read(1)
        unique, counts = np.unique(data, return_counts=True)
    return pd.DataFrame({'class': unique, 'count': counts})


def compute_ndvi(red_band, nir_band):
    """
    Calcule le NDVI à partir des bandes rouge et proche infrarouge.
    """
    nir_band = nir_band.astype('float32')
    red_band = red_band.astype('float32')
    ndvi = (nir_band - red_band) / (nir_band + red_band)
    return ndvi


def calculate_spectral_variability(samples, raster_path):
    """
    Calcule la distance moyenne au centroïde par classe pour un raster donné.
    """
    with rasterio.open(raster_path) as src:
        data = src.read()
    results = {}
    for class_label in np.unique(samples['class']):
        pixels = data[samples['class'] == class_label]
        centroid = np.mean(pixels, axis=0)
        distances = np.sqrt(np.sum((pixels - centroid) ** 2, axis=1))
        results[class_label] = np.mean(distances)
    return results


def train_random_forest(samples, features, target):
    """
    Entraîne un modèle Random Forest sur les échantillons.
    """
    model = RandomForestClassifier(max_depth=50, oob_score=True, max_samples=0.75, class_weight = 'balanced')
    model.fit(features, target)
    return model


def save_classification(model, features, output_file):
    """
    Applique un modèle de classification et sauvegarde la carte en raster.
    """
    predictions = model.predict(features)
    with rasterio.open(output_file, 'w', **features.meta) as dst:
        dst.write(predictions, 1)


def plot_violin(data, output_file):
    """
    Produit un violin plot pour visualiser les distributions.
    """
    plt.violinplot(data, showmeans=True)
    plt.savefig(output_file)

def plot_bar(data, title, xlabel, ylabel, output_path):
    """
    Génère un diagramme en bâtons
    """
    plt.figure(figsize=(10, 6))
    data.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def clip_raster_with_shapefile(raster_path, shapefile_path, output_path):
    """
    Découpe un raster selon l'emprise d'un shapefile en utilisant GDAL
    """
    gdal.UseExceptions()

    # Ouverture du raster 
    raster = gdal.Open(raster_path)
    shapefile = ogr.Open(shapefile_path)
    layer = shapefile.GetLayer()

    # Découpage de raster en prenant une couche vecteur comme masque 
    options = gdal.WarpOptions(cutlineDSName=shapefile_path,
                               cropToCutline=True,
                               dstNodata=0,
                               outputBoundsSRS="EPSG:2154")
    gdal.Warp(output_path, raster, options=options)

def reproject_and_resample(input_path, output_path, resolution=10):
    """
    Reprojette et rééchantillonne un raster en Lambert 93 à une résolution de 10 m
    """
    gdal.UseExceptions()

    # Ouverture du raster
    raster = gdal.Open(input_path)
    options = gdal.WarpOptions(
        xRes=resolution,
        yRes=resolution,
        dstSRS="EPSG:2154",
        resampleAlg="bilinear",
        dstNodata=0
    )
    gdal.Warp(output_path, raster, options=options)

def save_raster(data, ref_raster_path, output_path, dtype, nodata):
    """
    Sauvegarde d'une image raster en utilisant GDAL
    """
    ref = gdal.Open(ref_raster_path)
    driver = gdal.GetDriverByName('GTiff')