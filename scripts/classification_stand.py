import os
import geopandas as gpd
from rasterstats import zonal_stats
import pandas as pd
import sys
sys.path.append("/home/onyxia/work/Depot_Git/scripts")
from my_function import (apply_decision_rules, compute_confusion_matrix_with_plots)


# Définition des chemins d'accès
my_folder = "/home/onyxia/work/results/data"
sample_filename = os.path.join(my_folder, "sample/Sample_BD_foret_T31TCJ.shp")
image_filename = os.path.join(my_folder, "classif/carte_essences_echelle_pixel.tif")

# Utilisation de zonal_stats pour obtenir le total des pixels par polygone
zonal_statistics = zonal_stats(
    sample_filename,
    image_filename,
    stats=["count"],  # Nombre total de pixels par polygone
    categorical=True,  # Activation du mode catégoriel pour extraire les classes
)

# Liste de dictionnaires pour stocker les pourcentages des classes par polygone
polygon_classes_percentages = []

# Parcourir les statistiques zonales
for idx, stats in enumerate(zonal_statistics):
    polygon_id = idx + 1  
    total_pixels = stats["count"]  # Nombre total de pixels dans le polygone

    # Initialisation d'un dictionnaire pour stocker les pourcentages des classes
    class_percentages = {}
    
    for class_value, pixel_count in stats.items():
        if class_value == "count":  # Ignorer le total
            continue
        
        # Calcul du pourcentage
        percentage = (pixel_count / total_pixels) * 100
        class_percentages[class_value] = percentage

    # Ajout des résultats pour ce polygone
    polygon_classes_percentages.append({
        "polygon_id": polygon_id,
        "class_percentages": class_percentages,
    })

# Affichage des résultats
for polygon_result in polygon_classes_percentages:
    print(f"Polygone {polygon_result['polygon_id']} :")
    for class_value, percentage in polygon_result["class_percentages"].items():
        print(f"  Classe {class_value}: {percentage:.2f}%")

# Transformation des résultats sous forme de DataFrame

df_polygon_classes_percentages = pd.DataFrame(polygon_classes_percentages)
df_polygon_classes_percentages.head(5)

# Application des règles de décision
predictions = apply_decision_rules(df_polygon_classes_percentages, sample_filename)

# Chargement des échantillons à partir de la couche vectorielle
polygons = gpd.read_file(sample_filename)

# Stockage des codes prédits dans un attribut "code_predit"
polygons["code_predit"] = predictions

# Enregistrement du jeu de données reclassifié
output_path_samples = os.path.join(my_folder, "classif/carte_essences_echelle_peuplement.shp")
polygons.to_file(output_path_samples)

# Calcul de la matrice de confusion
confusion_matrix = compute_confusion_matrix_with_plots(polygons, "Code", "code_predit")
print(confusion_matrix)
