# Importation des bibliothèques 
import geopandas as gpd
import rasterstats
import os 
from my_function import calculate_class_percentages, apply_decision_rules, compute_confusion_matrix_with_plots
# Définition des cheminss d'accès 
my_folder = '/home/onyxia/work/Depot_Git/results/data'
sample_filename = os.path.join(my_folder, 'sample/Sample_BD_foret_T31TCJ.shp')
image_filename = os.path.join (my_folder,'classif/carte_essences_echelle_pixel.tif')


# Chargement de jeu de données 
polygons = gpd.read_file(sample_filename)

# Calcul la surface de chaque polygone
polygons["Surface"] = polygons.geometry.area

# Afficher les premières lignes
print(polygons.head())

# Calcul des pourcentages des classes dans les polygones
class_percentages = calculate_class_percentages(polygons, image_filename)

# Ajout des classifications prédites selon les règles de décision
polygons["code_predit"] = apply_decision_rules(class_percentages, sample_filename)
polygons.head(5)

# Calcul de  la matrice de confusion
confusion_matrix = compute_confusion_matrix_with_plots(polygons,"Nom","code_predit")
print(confusion_matrix)