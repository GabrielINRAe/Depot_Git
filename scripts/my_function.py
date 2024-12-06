# Pour charger vous images dans le workspace tapez la commande en changeant votre nom d'utilisateur:
# mc cp -r s3/nom_utilisateur/diffusion/images /home/onyxia/work/data

import geopandas as gpd
import rasterio
import numpy as np
import pandas as pd
from rasterio.features import rasterize
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt


def filter_classes(dataframe, valid_classes):
    """
    Filtre les classes de la BD Forêt 
    """
    return dataframe[dataframe['TFV'].isin(valid_classes)]


def count_polygons_by_class(dataframe, class_column='classif_pixel'):
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
    return (nir_band - red_band) / (nir_band + red_band)


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
    model = RandomForestClassifier(max_depth=50, oob_score=True, max_samples=0.75, class_weight='balanced')
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
