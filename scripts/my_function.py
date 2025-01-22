# Pour charger vos images dans le workspace tapez la commande en changeant votre nom d'utilisateur :
# mc cp -r s3/gabgab/diffusion/images /home/onyxia/work/data

import os
import seaborn as sns
import geopandas as gpd
from osgeo import gdal, ogr
import numpy as np
import pandas as pd
import matplotlib
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
    A REFAIRE PAR (GAB)
    """
    nir_band = nir_band.astype('float32')
    red_band = red_band.astype('float32')
    ndvi = (nir_band - red_band) / (nir_band + red_band)
    return ndvi

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

def report_from_dict_to_df(dict_report):
    '''
    Permet de convertir en DataFrame un dictionnaire retourné par la fonction classification_report
    '''
    # convert report into dataframe
    report_df = pd.DataFrame.from_dict(dict_report)

    # drop unnecessary rows and columns
    try :
        report_df = report_df.drop(['accuracy', 'macro avg', 'weighted avg'], axis=1)
    except KeyError:
        print(dict_report)
        report_df = report_df.drop(['micro avg', 'macro avg', 'weighted avg'], axis=1)

    report_df = report_df.drop(['support'], axis=0)

    return report_df


def compute_centroid(samples):
    """
    Calcule le centroïde d'un ensemble de points.
    :param samples: array, shape (n_samples, n_features)
    :return: array, shape (n_features,)
    """
    return np.mean(samples, axis=0)

def compute_avg_distance_to_centroid(samples, centroid):
    """
    Calcule la distance moyenne au centroïde.
    :param samples: array, shape (n_samples, n_features)
    :param centroid: array, shape (n_features,)
    :return: float
    """
    distances = np.sqrt(np.sum((samples - centroid) ** 2, axis=1))
    return np.mean(distances)

def create_bar_plot(data, output_path):
    """
    Crée un graphique en bâton pour les distances moyennes au centroïde.
    :param data: dict {class: avg_distance}
    :param output_path: str, chemin du fichier de sortie
    """
    classes = list(data.keys())
    distances = list(data.values())

    plt.figure(figsize=(10, 6))
    plt.bar(classes, distances, color='skyblue')
    plt.xlabel('Classes')
    plt.ylabel('Distance moyenne au centroïde')
    plt.title('Distance moyenne au centroïde par classe')
    plt.savefig(output_path)
    plt.close()

def create_violin_plot(polygon_distances, violin_plot_dist_centroide_by_poly_by_class_path):
    """
    Crée un graphique en violon pour visualiser les distances moyennes au centroïde par classe.

    Parameters:
    - polygon_distances (dict): Dictionnaire où les clés sont les noms des classes et les valeurs
      sont des listes de distances moyennes des polygones de chaque classe.
    - violin_plot_dist_centroide_by_poly_by_class_path (str): Chemin complet pour sauvegarder le graphique.
    """

    # Créer les données pour le graphique
    class_names = list(polygon_distances.keys())
    distances = [polygon_distances[cls] for cls in class_names]

    # Créer le graphique en violon
    plt.figure(figsize=(12, 8))
    
    # Vérifiez la version de Matplotlib et appliquez le paramètre approprié
    if matplotlib.__version__ >= "3.4.0":
        plt.violinplot(distances, showmeans=True, showextrema=True, showmedians=True)
    else:
        plt.violinplot(distances, showmeans=True, showextrema=True, showmedians=True)

    # Ajouter des labels et un titre
    plt.xticks(ticks=range(1, len(class_names) + 1), labels=class_names, rotation=45, fontsize=10)
    plt.xlabel("Classes", fontsize=12)
    plt.ylabel("Distances moyennes au centroïde", fontsize=12)
    plt.title("Distribution des distances moyennes au centroïde par polygone et par classe", fontsize=14)

    # Sauvegarder le graphique
    plt.tight_layout()
    plt.savefig(violin_plot_dist_centroide_by_poly_by_class_path, dpi=300)
    plt.close()


def get_polygon_labels(Y):
    """
    Fonction pour extraire les étiquettes des polygones associés aux pixels.

    Arguments :
    Y -- ndarray (numpy array) contenant les étiquettes des classes (ou autre identifiant, comme les polygones).

    Retourne :
    labels -- ndarray avec les étiquettes des polygones pour chaque pixel.
    """
    if Y.ndim > 1:
        Y = Y.flatten()  # Aplatir si nécessaire
    return Y  # Retourne directement Y s'il contient les étiquettes des polygones


def masque_shp(path_input, path_output):
    """
    Permet la création du masque en format shp à partir d'un fichier formation végétale shp.

    Parameters:
        path_input (str): Chemin du fichier pour accéder au fichier formation végétale
        path_output (str) : Chemin du fichier pour enregistrer le masque

    Returns:
        None
    """
    f_vege = gpd.read_file(path_input)    # Mettre le path en paramètre
    L_mask = ['Lande',
          'Formation herbacée',
          'Forêt ouverte de conifères purs',
          'Forêt ouverte de feuillus purs',
          'Forêt ouverte sans couvert arboré',
          'Forêt ouverte à mélange de feuillus et conifères',
          'Forêt fermée sans couvert arboré']   # Liste des classes à masquer
    ones = np.ones((24041,1),dtype=int)      # Création d'un vecteur de 1
    f_vege.loc[:,'value'] = ones             # Ajout de la colonne value remplis de 1
    # Valeur 0 pour les classes à masquer
    for i,j in zip(f_vege['TFV'],range(len(f_vege['value']))):
        if i in L_mask:
            f_vege.loc[j,'value'] = 0
    # Ajout de la colonne Classe
    for i in range(len(f_vege['value'])):
        if f_vege['value'][i] == 1:
            f_vege.loc[i,'Classe'] = 'Zone de forêt'
        else:
            f_vege.loc[i,'Classe'] = 'Zone hors forêt'

    Masque = f_vege[['ID','Classe','value','geometry']]    # Sélections des colonnes d'intérêt
    Masque.loc[:,'value'] = Masque['value'].astype('uint8')   # Conversion de la colonne value en uint8

    Masque.to_file(path_output)  # Enregistrement du masque
    return None


def rasterization (
    in_vector,
    out_image,
    field_name,
    sp_resol,
    emprise = None):
    """
    Rasterise un fichier vectoriel.

    Parameters:
        in_vector (str): Chemin du fichier vectoriel à rasteriser.
        out_image (str): Chemin du fichier raster en sortie.
        field_name (str): Nom de la colonne du vecteur à rasteriser.
        sp_resol (str): Résolution spatiale du fichier à rasteriser.
        emprise (str, optional): Chemin du fichier emprise sur lequel rasteriser.

    Returns:
        None
    """
    if emprise is not None :
        xmin,ymin,xmax,ymax=emprise.total_bounds
    
    # Créer le répertoire de sortie si nécessaire
    out_dir = os.path.dirname(out_image)
    os.makedirs(out_dir, exist_ok=True)  # Crée les répertoires manquants

    # define command pattern to fill with parameters
    cmd_pattern = ("gdal_rasterize -a {field_name} "
                "-tr {sp_resol} {sp_resol} "
                "-te {xmin} {ymin} {xmax} {ymax} -ot Byte -of GTiff "
                "{in_vector} {out_image}")

    # fill the string with the parameter thanks to format function
    cmd = cmd_pattern.format(in_vector=in_vector, xmin=xmin, ymin=ymin, xmax=xmax,
                            ymax=ymax, out_image=out_image, field_name=field_name,
                            sp_resol=sp_resol)

    # execute the command in the terminal
    os.system(cmd)
    return None