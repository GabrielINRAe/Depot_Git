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
import sys
sys.path.append('/home/onyxia/work/libsigma')
import read_and_write as rw
import classification as cla
from rasterstats import zonal_stats
from sklearn.metrics import (confusion_matrix, classification_report,
    accuracy_score, precision_recall_fscore_support)
from sklearn.model_selection import StratifiedGroupKFold
from collections import defaultdict

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
    sp_resol = None,
    emprise = None,
    ref_image = None,
    dtype = None):
    """
    Rasterise un fichier vectoriel.

    Parameters:
        in_vector (str): Chemin du fichier vectoriel à rasteriser.
        out_image (str): Chemin du fichier raster en sortie.
        field_name (str): Nom de la colonne du vecteur à rasteriser.
        sp_resol (str,optional): Résolution spatiale du fichier à rasteriser.
        emprise (str, optional): Chemin du fichier emprise sur lequel rasteriser.
        ref_image (str, optional): Chemin du fichier image référence pour la rasterisation.
        dtype (str, optional) : Type de données en sortie.

    Returns:
        None
    """
    if emprise is not None :
        xmin,ymin,xmax,ymax=emprise.total_bounds
    else :
        ref_image_open = rw.open_image(ref_image)
        if sp_resol is None :
            sp_resol = rw.get_pixel_size(ref_image_open)[0]
        if dtype is None :
            band = ref_image_open.GetRasterBand(1)
            dtype = gdal.GetDataTypeName(band.DataType)
        xmin,ymax = rw.get_origin_coordinates(ref_image_open)
        y,x = rw.get_image_dimension(ref_image_open)[0:2]
        xmax,ymin = xmin+x*10,ymax-y*10
    
    # Créer le répertoire de sortie si nécessaire
    out_dir = os.path.dirname(out_image)
    os.makedirs(out_dir, exist_ok=True)  # Crée les répertoires manquants

    # define command pattern to fill with parameters
    cmd_pattern = ("gdal_rasterize -a {field_name} "
                "-tr {sp_resol} {sp_resol} "
                "-te {xmin} {ymin} {xmax} {ymax} -ot {dtype} -of GTiff "
                "{in_vector} {out_image}")

    # fill the string with the parameter thanks to format function
    cmd = cmd_pattern.format(in_vector=in_vector, xmin=xmin, ymin=ymin, xmax=xmax,
                            ymax=ymax, out_image=out_image, field_name=field_name,
                            sp_resol=sp_resol, dtype = dtype)

    # execute the command in the terminal
    os.system(cmd)
    return None

def apply_decision_rules(class_percentages, samples_path):
    
    """
    Applique des règles de décision pour déterminer la classe prédominante de chaque polygone.

    Arguments :
    - class_percentages : DataFrame contenant une colonne `class_percentages` avec des dictionnaires.
    - samples_path : Chemin vers le fichier des échantillons.

    Retourne :
    - Une liste `code_predit` avec les codes prédits pour chaque polygone.
    """
    code_predit = []  # Liste pour stocker les classes prédites
    samples = gpd.read_file(samples_path)  # Charger les données des échantillons
    samples["Surface"] = samples.geometry.area  # Calculer la surface des polygones

    for index, row in class_percentages.iterrows():
        # Récupérer le dictionnaire des pourcentages pour ce polygone
        class_dict = row["class_percentages"]

        # Surface du polygone
        surface = samples.loc[index, "Surface"] if index in samples.index else 0

        # Identifier la classe dominante et son pourcentage
        if class_dict:  # Vérifier que le dictionnaire n'est pas vide
            dominant_class_name = max(class_dict, key=class_dict.get)  # Classe avec le plus grand pourcentage
            dominant_class_percentage = class_dict[dominant_class_name]  # Pourcentage de cette classe
        else:  # Si le dictionnaire est vide
            dominant_class_name = None
            dominant_class_percentage = 0

    for index, row in class_percentages.iterrows():

        # Calcul des proportions
        sum_feuillus = row.get("11", 0) + row.get("16", 0) + row.get("15", 0)+row.get("12", 0)+row.get("14", 0)+row.get("13", 0)
        sum_coniferes = row.get("21", 0) + row.get("27", 0) + row.get("26", 0)+ row.get("23", 0)+ row.get("25", 0)+ row.get("24", 0)+ row.get("22", 0)

        # Décisions
        if surface < 20000:  # Cas surface < 2 ha
            if sum_feuillus > 75 and sum_coniferes < sum_feuillus: 
                code_predit.append("Feuillus_en_ilots")
            elif sum_coniferes > 75 and sum_coniferes > sum_feuillus: 
                code_predit.append("coniferes_en_ilots")
            elif sum_coniferes > sum_feuillus: 
                code_predit.append("Melange_de_coniferes_preponderants_et_feuillus")
            else:
                code_predit.append("Melange_de_feuillus_preponderants_et_coniferes")
        else:  # Cas surface >= 2 ha
            if dominant_class_percentage > 75:
                code_predit.append(dominant_class_name)
            elif sum_feuillus > 75 and sum_coniferes < 75: 
                code_predit.append("Melange_feuillus")
            elif sum_coniferes > 75 and sum_feuillus < 75: 
                code_predit.append("Melange_coniferes")
            elif sum_coniferes > sum_feuillus:
                code_predit.append("Melange_de_coniferes_preponderants_et_feuillus")
            else:
                code_predit.append("Melange_de_feuillus_preponderants_et_coniferes")
    return code_predit


def compute_confusion_matrix_with_plots(polygons, label_col, prediction_col):
    """
    Calcule la matrice de confusion, affiche les métriques et génère les graphiques demandés.
    :param polygons: GeoDataFrame ou DataFrame contenant les labels et prédictions.
    :param label_col: Nom de la colonne pour les labels vrais.
    :param prediction_col: Nom de la colonne pour les prédictions.
    :param output_dir: Répertoire où sauvegarder les graphiques.
    """
    # Vérification des colonnes
    if label_col not in polygons.columns or prediction_col not in polygons.columns:
        raise ValueError(f"Les colonnes {label_col} et/ou {prediction_col} sont introuvables dans les données.")

    # Suppression des lignes avec des valeurs manquantes dans les colonnes d'intérêt
    polygons = polygons.dropna(subset=[label_col, prediction_col])

    # Récupération des labels vrais et prédits
    y_true = polygons[label_col].astype(str)  # Conversion en chaîne pour éviter les comparaisons avec None
    y_pred = polygons[prediction_col].astype(str)
    print(polygons[[label_col, prediction_col]].head(10))
    # Calcul de la matrice de confusion
    labels = np.unique(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Classification report
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    
    # Normalisation pour les pourcentages
    cm_sum = cm.sum(axis=1)
    cm_sum[cm_sum == 0] = 1  # Évite la division par zéro
    cm_normalized = cm.astype('float') / cm_sum[:, np.newaxis]

    # ---- Création de la heatmap de la matrice de confusion ----
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Greens", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title("Confusion Matrix with Normalized Values")
    plt.tight_layout()
    plt.show()

    # ---- Création du graphique des métriques (précision, rappel, F1) ----
    metrics = np.array([precision, recall, f1_score])
    metric_names = ["Precision", "Recall", "F1 Score"]

    plt.figure(figsize=(10, 8))
    bar_width = 0.25
    x = np.arange(len(labels))

    for i, metric in enumerate(metrics):
        plt.bar(x + i * bar_width, metric * 100, width=bar_width, label=metric_names[i])

    # Personnalisation des axes
    plt.xlabel("Classes")
    plt.ylabel("Score (%)")
    plt.title("Class quality estimation")
    plt.xticks(x + bar_width, labels, rotation=45, ha="right")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Retour des métriques au besoin
    return {
        "confusion_matrix": cm,
        "classification_report": classification_report(y_true, y_pred, labels=labels, zero_division=0),
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }



def pre_traitement_img(
    p_emprise,
    l_images,
    input_raster_dir,
    output_dir):
    """
    Rasterise un fichier vectoriel.

    Parameters:
        p_emprise (str): Chemin du fichier vectoriel pour l'emprise du clip.
        l_images (list): Liste des images à traiter.
        input_raster_dir (str): Dossier où les images brutes sont stockées.
        output_dir (str): Chemin du dossier temporaire des output pré-traités.

    Returns:
        None.
    """
    # Charger le vecteur avec Geopandas
    emprise = gpd.read_file(p_emprise).to_crs("EPSG:2154")
    # Extraire le GeoJSON sous forme de string
    print("Chargement du geojson en str")
    geojson_str = emprise.to_json()
    print("Chargement du geojson en str ok!")
    print("Traitements des images")
    for i,img in enumerate(l_images) :
        date = img[11:19]
        bande = img[53:]
        ds_img = rw.open_image(os.path.join(input_raster_dir,img))
        name_file = f"traitement_{date}_{bande}"
        output_file = os.path.join(output_dir,name_file)
        # Appliquer le clip avec GDAL
        resolution = 10  # Résolution (10 m)
        output_raster_test = gdal.Warp(
            output_file, # Chemin de fichier, car on utilise GTiff
            # "",  # Pas de chemin de fichier, car on utilise MEM
            ds_img,  # Fichier en entrée (chemin ou objet gdal)
            format = "GTiff", # Utiliser GTiff comme format
            # format = "MEM",  # Utiliser MEM comme format
            cutlineDSName = geojson_str,  # Passer directement le GeoJSON
            cropToCutline = True,
            outputType = gdal.GDT_UInt16, #UInt16
            dstSRS = "EPSG:2154",  # Reprojection
            xRes = resolution,  # Résolution X
            yRes = resolution,  # Résolution Y
            dstNodata = 0  # Valeur NoData
        )
        print(f"Image {i+1}/{len(l_images)} traitée")
    ds_img = None
    emprise = None
    geojson_str = None
    return None


def id_construction(sample_px, path_sample_px_id):
    """
    Construis un fichier shp avec une colonne "id" sur les polygones.

    Parameters:
        sample_px (str): Chemin du fichier où ajouter les id.
        path_sample_px_id (str): Chemin du fichier id en sortie.

    Returns:
        None
    """
    l_id = [i+1 for i in range(sample_px.shape[0])]
    sample_px_id = sample_px.copy()
    sample_px_id['id'] = l_id
    sample_px_id = sample_px_id[['id','geometry']]
    sample_px_id.to_file(path_sample_px_id)
    return None


def stratified_grouped_validation(
    nb_iter,
    nb_folds,
    sample_filename,
    image_filename,
    id_filename
):
    """
    Réalise l'entrainement et la validation du modèle.

    Parameters:
        nb_iter (int): Nombre d'itération.
        nb_folds (int): Nombre de folds.
        sample_filename (str): Chemin vers le raster des échantillons.
        image_filename (str): Chemin vers l'image sur laquelle entrainer le modèle.
        id_filename (str): Chemin vers le raster des id des polygones.

    Returns:
        rfc: Retourne le modèle.
        list_cm: Liste des matrices de confusions.
        list_accuracy: Liste des scores de précisions.
        list_report: Liste des rapports de classifications.
        Y_predict: Utilisé pour les labels pour le plot.
    """
    # Extraction des échantillons
    X, Y, t = cla.get_samples_from_roi(image_filename, sample_filename)
    _, groups, _ = cla.get_samples_from_roi(image_filename, id_filename)
    list_cm = []   # Stockage des matrices de confusions
    list_accuracy = []    # Stockage des OA
    list_report = []    # Stockage des rapports de classifications
    groups = np.squeeze(groups)
    # Iter on stratified K fold
    for i in range(nb_iter):
        print (f"Début de la {i+1} itération")
        kf = StratifiedGroupKFold(n_splits=nb_folds, shuffle=True)
        for train, test in kf.split(X, Y, groups=groups):
            X_train, X_test = X[train], X[test]
            Y_train, Y_test = Y[train], Y[test]

            # 3 --- Train
            rfc = RandomForestClassifier(
                max_depth = 50,
                oob_score = True,
                max_samples = 0.75,
                class_weight = 'balanced',
                n_jobs = -1
            )
            rfc.fit(X_train, Y_train[:,0])

            # 4 --- Test
            Y_predict = rfc.predict(X_test)

            # compute quality
            list_cm.append(confusion_matrix(Y_test, Y_predict))
            list_accuracy.append(accuracy_score(Y_test, Y_predict))
            report = classification_report(Y_test, Y_predict,
                                            labels=np.unique(Y_predict),
                                            output_dict=True,
                                            zero_division = 0)

            # store them
            list_report.append(report_from_dict_to_df(report))
    return rfc, list_cm, list_accuracy, list_report, Y_predict


def save_classif(
    image_filename,
    model,
    out_classif
):
    """
    Produit la carte finale de classification.

    Parameters:
        image_filename (str): Chemin du fichier vers l'image utilisée pour la classification.
        model (sklearn): Modèle utilisé lors de l'apprentissage.
        out_classif (str): Chemin de sauvegarde de la classif finale.

    Returns:
        None
    """
    X_img, _, t_img = cla.get_samples_from_roi(image_filename, image_filename)
    Y_predict = model.predict(X_img)
    # Get image dimension
    ds = rw.open_image(image_filename)
    nb_row, nb_col, _ = rw.get_image_dimension(ds)
    #initialization of the array
    img = np.zeros((nb_row, nb_col, 1), dtype='uint8')
    img[t_img[0], t_img[1], 0] = Y_predict
    rw.write_image(out_classif, img, data_set=ds, gdal_dtype=gdal.GDT_Byte,
                transform=None, projection=None, driver_name=None,
                nb_col=None, nb_ligne=None, nb_band=1)
    return None