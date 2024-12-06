# Importation des bibliothèques
import geopandas as gpd
import sys
import os
from my_function import filter_classes
import pandas as pd

# Ajout du chemin au PATH
sys.path.append('/home/onyxia/Projet_Teledec/scripts')

# definition des paramètres 
my_folder = '/home/onyxia/work/data/project'
out_folder = '/home/onyxia/Projet_Teledec/results/data'
in_vector = os.path.join(my_folder, 'FORMATION_VEGETALE.shp')
emprise_path = os.path.join(my_folder, 'emprise_etude.shp')
out_file = os.path.join(out_folder, 'sample/Sample_BD_foret_T31TCJ.shp')

# Chargement des données
# Chargement de BD Forêt
bd_foret = gpd.read_file(in_vector)
print(bd_foret.head())
print(bd_foret.shape)

# Chargement de l'emprise d'étude
emprise_etude = gpd.read_file(emprise_path)
print(emprise_etude.head())

# Découpage de la base filtrée en utilisant l'emprise d'étude comme masque
bd_foret = bd_foret.to_crs(emprise_etude.crs)
bd_foret_clipped = bd_foret.clip(emprise_etude)
print(bd_foret_clipped)

# Identification des classes dans la base BD Forêt
bd_foret_classes = bd_foret["TFV"].unique()
print(bd_foret_classes)

# Liste des classes valides
valide_classes = [
    "Forêt fermée d’un autre feuillu pur",
    "Forêt fermée de châtaignier pur",
    "Forêt fermée de hêtre pur",
    "Forêt fermée de chênes décidus purs",
    "Forêt fermée de robinier pur",
    "Peupleraie",
    "Forêt fermée à mélange de feuillus",
    "Forêt fermée de feuillus purs en îlots",
    "Forêt fermée d’un autre conifère pur autre que pin\xa0",
    "Forêt fermée de mélèze pur",
    "Forêt fermée de sapin ou épicéa",
    "Forêt fermée à mélange d’autres conifères",
    "Forêt fermée d’un autre pin pur",
    "Forêt fermée de pin sylvestre pur",
    "Forêt fermée à mélange de pins purs",
    "Forêt fermée de douglas pur",
    "Forêt fermée de pin laricio ou pin noir pur",
    "Forêt fermée de pin maritime pur",
    "Forêt fermée à mélange de conifères",
    "Forêt fermée de conifères purs en îlots",
    "Forêt fermée à mélange de conifères prépondérants et feuillus",
    "Forêt fermée à mélange de feuillus prépondérants et conifères",
]

# Filtrage des échantillons selon les classes valides
filtered_samples = filter_classes(bd_foret, valide_classes)
print(filtered_samples.head())

# Identification des classes dans les échantillons filtrés
filtered_samples_classes = filtered_samples["TFV"].unique()
print(filtered_samples_classes)

# Création du dictionnaire de correspondance des catégories
category_mapping = {
    "Forêt fermée d’un autre feuillu pur": {
        "nom_pixel": "Autres feuillus",
        "Code_pixel": "11",
        "Objet_nom": "Autres feuillus",
        "Objet_code": "11",
    },
    "Forêt fermée de châtaignier pur": {
        "nom_pixel": "Autres feuillus",
        "Code_pixel": "11",
        "Objet_nom": "Autres feuillus",
        "Objet_code": "11",
    },
    "Forêt fermée de hêtre pur": {
        "nom_pixel": "Autres feuillus",
        "Code_pixel": "11",
        "Objet_nom": "Autres feuillus",
        "Objet_code": "11",
    },
    "Forêt fermée de chênes décidus purs": {
        "nom_pixel": "Chêne",
        "Code_pixel": "12",
        "Objet_nom": "Chêne",
        "Objet_code": "12",
    },
    "Forêt fermée de robinier pur": {
        "nom_pixel": "Robinier",
        "Code_pixel": "13",
        "Objet_nom": "Robinier",
        "Objet_code": "13",
    },
    "Peupleraie": {
        "nom_pixel": "Peupleraie",
        "Code_pixel": "14",
        "Objet_nom": "Peupleraie",
        "Objet_code": "14",
    },
    "Forêt fermée à mélange de feuillus": {
        "nom_pixel": "",
        "Code_pixel": "",
        "Objet_nom": "Mélange de feuillus",
        "Objet_code": "15",
    },
    # Ajoutez le reste des catégories ici...
}

# Ajout des colonnes via mapping et conversion en DataFrame
df = filtered_samples.join(
    filtered_samples["TFV"].map(lambda x: category_mapping.get(x, {})).apply(pd.Series)
)

print(df.head())

# Enregistrement de la couche de séléction des échantillons
df.to_file(out_file)
print("l'enregistrement est réalisé avec succès ")