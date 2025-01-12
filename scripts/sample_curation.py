# Importation des bibliothèques
import geopandas as gpd
import sys 
import os
import pandas as pd

# Ajout du chemin au PATH
sys.path.append('/home/onyxia/Depot_Git/scripts')
from my_function import filter_classes

# definition des paramètres 
my_folder = '/home/onyxia/work/data/project'
out_folder = '/home/onyxia/work/Depot_Git/results/data'
in_vector = os.path.join(my_folder, 'FORMATION_VEGETALE.shp')
emprise_path = os.path.join(my_folder, 'emprise_etude.shp')
out_file = os.path.join(out_folder,'sample/Sample_BD_foret_T31TCJ.shp')

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
filtered_samples = filter_classes(bd_foret_clipped, valide_classes)
print(filtered_samples.head())

# Identification des classes dans les échantillons filtrés
filtered_samples_classes = filtered_samples["TFV"].unique()
print(filtered_samples_classes)

# Création du dictionnaire de correspondance des catégories
category_mapping = {
    'Forêt fermée d’un autre feuillu pur': {'Nom': 'Autres feuillus', 'Code': '11'},
    'Forêt fermée de châtaignier pur': {'Nom': 'Autres feuillus', 'Code': '11'},
    'Forêt fermée de hêtre pur': {'Nom': ' Autres feuillus', 'Code': '11'},
    'Forêt fermée de chênes décidus purs': {'Nom': 'Chêne', 'Code': '12'},
    'Forêt fermée de robinier pur': {'Nom': 'Robinier', 'Code': '13'},
    'Peupleraie': {'Nom': 'Peupleraie', 'Code': '14'},
    'Forêt fermée à mélange de feuillus': {'Nom': 'Mélange de feuillus', 'Code': '15'},
    'Forêt fermée de feuillus purs en îlots': {'Nom': 'Feuillus en îlots', 'Code': '16'},
    'Forêt fermée d’un autre conifère pur autre que pin': {'Nom': 'Autres conifères autre que pin', 'Code': '21'},
    'Forêt fermée de mélèze pur': {'Nom': 'Autres conifères autre que pin', 'Code': '21'},
    'Forêt fermée de sapin ou épicéa': {'Nom': 'Autres conifères autre que pin', 'Code': '21'},
    'Forêt fermée à mélange d’autres conifères': {'Nom': 'Autres conifères autre que pin', 'Code': '21'},
    'Forêt fermée d’un autre pin pur': {'Nom': 'Autres Pin', 'Code': '22'},
    'Forêt fermée de pin sylvestre pur': {'Nom': 'Autres Pin', 'Code': '22'},
    'Forêt fermée à mélange de pins purs': {'Nom': 'Autres Pin', 'Code': '22'},
    'Forêt fermée de douglas pur': {'Nom': 'Douglas', 'Code': '23'},
    'Forêt fermée de pin laricio ou pin noir pur': {'Nom': 'Pin laricio ou pin noir', 'Code': '24'},
    'Forêt fermée de pin maritime pur': {'Nom': 'Pin_maritime', 'Code': '25'},
    'Forêt fermée à mélange de conifères': {'Nom': 'Mélange conifères', 'Code': '26'},
    'Forêt fermée de conifères purs en îlots': {'Nom': 'Conifères en îlots', 'Code': '27'},
    'Forêt fermée à mélange de conifères prépondérants et feuillus': {'Nom': 'Mélange de conifères prépondérants et_feuillus', 'Code': '28'},
    'Forêt fermée à mélange de feuillus prépondérants et conifères': {'Nom': 'Mélange_de_feuillus prépondérants et conifères', 'Code': '29'},
}

# Ajout des colonnes via mapping et conversion en DataFrame
df = filtered_samples.join(
    filtered_samples["TFV"].map(lambda x: category_mapping.get(x, {})).apply(pd.Series)
)

print(df.head())

# Enregistrement de la couche de séléction des échantillons
df.to_file(out_file)
print("l'enregistrement est réalisé avec succès ")