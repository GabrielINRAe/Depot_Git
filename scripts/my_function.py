# Pour charger vos images dans le workspace tapez la commande en changeant votre nom d'utilisateur :
# mc cp -r s3/gabgab/diffusion/images /home/onyxia/work/data

import os
import seaborn as sns
import geopandas as gpd
import rasterio
from osgeo import gdal, ogr
import numpy as np
import pandas as pd
from rasterio.features import rasterize
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

def filter_classes(dataframe, valid_classes):
    """
    Filtre les classes de la BD Forêt.
    """
    return dataframe[dataframe['TFV'].isin(valid_classes)]

def sel_classif_pixel(dataframe):
    """
    Sélectionne seulement les classes pour la classification à l'échelle des pixels
    """
    codes = [11,12,13,14,21,22,23,24,25]
    return dataframe[dataframe['Code'].isin(codes)]

def count_polygons_by_class(dataframe, class_column='classif_objet'):
    """
    Compte le nombre de polygones par classe.
    """
    return dataframe.groupby(class_column).size().reset_index(name='count')


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
    model = RandomForestClassifier(
        max_depth=50, oob_score=True, max_samples=0.75, class_weight='balanced'
    )
    model.fit(features, target)
    return model


def save_classification(model, features, output_file):
    """
    Applique un modèle de classification et sauvegarde la carte en raster.
    """
    predictions = model.predict(features)
    with rasterio.open(output_file, 'w', **features.meta) as dst:
        dst.write(predictions, 1)


def plot_bar(data, title, xlabel, ylabel, output_path):
    """
    Génère un diagramme en bâtons.
    """
    plt.figure(figsize=(10, 6))
    data.plot(kind='bar', color='lightcoral', edgecolor='black')
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def violin_plot(
    df, x_col, y_col, output_file, title="", xlabel="", ylabel="", palette="muted", figsize=(12, 8)
):
    """
    Crée un graphique de type violin plot pour visualiser la distribution des données autour d'une valeur moyenne.

    Parameters:
        df (pd.DataFrame): DataFrame contenant les données à tracer.
        output_file (str): Chemin et nom du fichier pour enregistrer le graphique.
        title (str, optional): Titre du graphique. Par défaut "".
        xlabel (str, optional): Étiquette de l'axe X. Par défaut "".
        ylabel (str, optional): Étiquette de l'axe Y. Par défaut "".
        palette (str, optional): Palette de couleurs pour le graphique. Par défaut "muted".
        figsize (tuple, optional): Taille de la figure. Par défaut (12, 8).

    Returns:
        None
    """
    plt.figure(figsize=figsize)
    sns.violinplot(data=df, x=x_col, y=y_col, palette=palette)
    plt.xlabel(xlabel if xlabel else x_col, fontsize=12)
    plt.ylabel(ylabel if ylabel else y_col, fontsize=12)
    plt.title(title, fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()


def clip_raster_with_shapefile(raster_path, shapefile_path, output_path):
    """
    Découpe un raster selon l'emprise d'un shapefile en utilisant GDAL.
    """
    gdal.UseExceptions()
    raster = gdal.Open(raster_path)
    shapefile = ogr.Open(shapefile_path)
    layer = shapefile.GetLayer()
    options = gdal.WarpOptions(
        cutlineDSName=shapefile_path,
        cropToCutline=True,
        dstNodata=0,
        outputBoundsSRS="EPSG:2154"
    )
    gdal.Warp(output_path, raster, options=options)


def reproject_and_resample(input_path, output_path, resolution=10):
    """
    Reprojette et rééchantillonne un raster en Lambert 93 à une résolution de 10 m.
    """
    gdal.UseExceptions()
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
    Sauvegarde d'une image raster en utilisant GDAL.
    """
    ref = gdal.Open(ref_raster_path)
    driver = gdal.GetDriverByName('GTiff')


def supprimer_dossier_non_vide(dossier):
    '''
    Permet de supprimer un dossier contenant des fichiers
    '''
    # Parcourir tout le contenu du dossier
    for element in os.listdir(dossier):
        chemin_element = os.path.join(dossier, element)
        # Vérifier si c'est un fichier
        if os.path.isfile(chemin_element) or os.path.islink(chemin_element):
            os.remove(chemin_element)  # Supprimer le fichier ou le lien
        elif os.path.isdir(chemin_element):
            supprimer_dossier_non_vide(chemin_element)  # Appel récursif pour les sous-dossiers
    # Supprimer le dossier une fois qu'il est vide
    os.rmdir(dossier)


def read_ndvi_with_gdal(ndvi_file):
    """Lit le fichier NDVI multi-bandes avec GDAL et renvoie les données NDVI et les noms des bandes."""
    dataset = gdal.Open(ndvi_file)
    bands = [dataset.GetRasterBand(i + 1).ReadAsArray() for i in range(dataset.RasterCount)]
    dates = [dataset.GetRasterBand(i + 1).GetDescription() for i in range(dataset.RasterCount)]
    return np.array(bands), dates

def rasterize_vector(geometry, output_file, class_column, spatial_resolution, bounds):
    """
    Rasterise un vecteur en utilisant GDAL.
    
    Paramètres :
    - geometry : GeoSeries contenant les géométries à rasteriser.
    - output_file : Chemin du fichier de sortie rasterisé.
    - class_column : Nom de la colonne des attributs à utiliser pour la rasterisation.
    - spatial_resolution : Résolution spatiale (en mètres/pixels).
    - bounds : Tuple (xmin, ymin, xmax, ymax) représentant l'étendue spatiale.
    
    Lève une exception en cas d'échec de la rasterisation.
    """
    # Créer un fichier vecteur temporaire (au format GeoJSON)
    temp_vector = "temp_vector.geojson"
    try:
        # Enregistrer les géométries dans un fichier temporaire au format GeoJSON
        geometry.to_file(temp_vector, driver="GeoJSON")
        
        # Extraire les limites spatiales (bounds)
        xmin, ymin, xmax, ymax = bounds
        
        # Construire la commande GDAL
        cmd = (
            f"gdal_rasterize -a {class_column} "
            f"-tr {spatial_resolution} {spatial_resolution} "
            f"-te {xmin} {ymin} {xmax} {ymax} -ot Byte -of GTiff "
            f"{temp_vector} {output_file}"
        )
        
        print(f"Exécution de la commande : {cmd}")
        status = os.system(cmd)  # Exécuter la commande

        # Vérifier si la rasterisation a échoué
        if status != 0:
            raise RuntimeError(f"La commande GDAL a échoué : {cmd}")
        else:
            print(f"Rasterisation terminée pour {temp_vector}. Raster enregistré à : {output_file}")
    
    finally:
        # Supprimer le fichier vecteur temporaire
        if os.path.exists(temp_vector):
            os.remove(temp_vector)

def calculate_ndvi_stats(ndvi_data, samples, class_column, geometry_column, spatial_resolution, reference_raster):
    """Calcule les statistiques NDVI pour chaque classe des échantillons."""
    ndvi_stats = []

    # Obtenir les limites du raster de référence
    with gdal.Open(reference_raster) as ref:
        bounds = ref.GetGeoTransform()
        xmin, pixel_width, _, ymax, _, pixel_height = bounds
        xmax = xmin + (pixel_width * ref.RasterXSize)
        ymin = ymax + (pixel_height * ref.RasterYSize)
        bounds = (xmin, ymin, xmax, ymax)

    # Liste pour garder une trace des fichiers temporaires créés
    temp_rasters = []

    try:
        for cls in samples[class_column].unique():
            # Définir les fichiers de sortie pour la rasterisation des classes
            class_mask_raster = f"temp_class_{cls}_mask.tif"
            temp_rasters.append(class_mask_raster)  # Ajouter à la liste des fichiers temporaires

            # Sélectionner les échantillons pour cette classe
            class_samples = samples[samples[class_column] == cls]
            
            # Rasteriser les géométries de la classe
            # Appel de la fonction rasterize_vector définie précédemment
            rasterize_vector(class_samples, class_mask_raster, class_column, spatial_resolution, bounds)
            
            # Charger le masque rasterisé pour cette classe
            class_mask = gdal.Open(class_mask_raster).ReadAsArray()

            for i in range(ndvi_data.shape[0]):  # Parcourir les bandes NDVI (temps)
                # Calculer les statistiques NDVI pour cette classe et bande
                class_pixels = ndvi_data[i][class_mask == 1]
                if class_pixels.size > 0:
                    mean_ndvi = np.mean(class_pixels)
                    std_ndvi = np.std(class_pixels)
                    ndvi_stats.append({'classe': cls, 'date': i, 'mean': mean_ndvi, 'std': std_ndvi})

    finally:
        # Supprimer les fichiers temporaires créés
        for temp_file in temp_rasters:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                print(f"Fichier temporaire supprimé : {temp_file}")

    return ndvi_stats


def plot_ndvi_signatures(ndvi_stats, output_file):
    """Trace les courbes des signatures temporelles de NDVI."""
    plt.figure(figsize=(12, 6))

    for cls in set(stat['classe'] for stat in ndvi_stats):
        cls_stats = [x for x in ndvi_stats if x['classe'] == cls]
        dates = [x['date'] for x in cls_stats]
        means = [x['mean'] for x in cls_stats]
        stds = [x['std'] for x in cls_stats]

        plt.plot(dates, means, label=f"Classe {cls}")
        plt.fill_between(dates, np.array(means) - np.array(stds), np.array(means) + np.array(stds), alpha=0.2)

    plt.title("Signature temporelle de NDVI par classe")
    plt.xlabel("Date")
    plt.ylabel("NDVI")
    plt.legend()
    plt.savefig(output_file)
    plt.close()