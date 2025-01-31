# Importation des bibliothèques
import geopandas as gpd
import sys
import os
import pandas as pd

# Ajout du chemin au PATH
sys.path.append('/home/onyxia/Depot_Git/scripts')
from my_function import filter_classes

# Définition des paramètres
racine = '/home/onyxia/work'
my_folder = os.path.join(racine, 'data/project')
output_folder = os.path.join(racine, "results/data/sample")
os.makedirs(output_folder, exist_ok=True)
in_vector = os.path.join(my_folder, 'FORMATION_VEGETALE.shp')
emprise_path = os.path.join(my_folder, 'emprise_etude.shp')
out_file = os.path.join(output_folder, 'Sample_BD_foret_T31TCJ.shp')

# Chargement des données
# Chargement de BD Forêt
bd_foret = gpd.read_file(in_vector)

# Chargement de l'emprise d'étude
emprise_etude = gpd.read_file(emprise_path)

# Découpage de la base filtrée en utilisant l'emprise d'étude comme masque
bd_foret = bd_foret.to_crs(emprise_etude.crs)
bd_foret_clipped = bd_foret.clip(emprise_etude)

# Identification des classes dans la base BD Forêt
bd_foret_classes = bd_foret["TFV"].unique()

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

# Identification des classes dans les échantillons filtrés
filtered_samples_classes = filtered_samples["TFV"].unique()

category_mapping = {
    'Forêt fermée d’un autre feuillu pur': \
        {'Nom': 'Autres_feuillus', 'Code': 11},
    'Forêt fermée de châtaignier pur': {'Nom': 'Autres_feuillus', 'Code': 11},
    'Forêt fermée de hêtre pur': {'Nom': 'Autres_feuillus', 'Code': 11},
    'Forêt fermée de chênes décidus purs': {'Nom': 'Chene', 'Code': 12},
    'Forêt fermée de robinier pur': {'Nom': 'Robinier', 'Code': 13},
    'Peupleraie': {'Nom': 'Peupleraie', 'Code': 14},
    'Forêt fermée à mélange de feuillus': \
        {'Nom': 'Melange_de_feuillus', 'Code': 15},
    'Forêt fermée de feuillus purs en îlots': \
        {'Nom': 'Feuillus_en_ilots', 'Code': 16},
    'Forêt fermée d’un autre conifère pur autre que pin': \
        {'Nom': 'Autres_coniferes_autre_que_pin', 'Code': 21},
    'Forêt fermée de mélèze pur': \
        {'Nom': 'Autres_coniferes_autre_que_pin', 'Code': 21},
    'Forêt fermée de sapin ou épicéa': \
        {'Nom': 'Autres_coniferes_autre_que_pin', 'Code': 21},
    'Forêt fermée à mélange d’autres conifères': \
        {'Nom': 'Autres_coniferes_autre_que_pin', 'Code': 21},
    'Forêt fermée d’un autre pin pur': {'Nom': 'Autres_Pin', 'Code': 22},
    'Forêt fermée de pin sylvestre pur': {'Nom': 'Autres_Pin', 'Code': 22},
    'Forêt fermée à mélange de pins purs': {'Nom': 'Autres_Pin', 'Code': 22},
    'Forêt fermée de douglas pur': {'Nom': 'Douglas', 'Code': 23},
    'Forêt fermée de pin laricio ou pin noir pur': \
        {'Nom': 'Pin_laricio_ou_pin_noir', 'Code': 24},
    'Forêt fermée de pin maritime pur': \
        {'Nom': 'Pin_maritime', 'Code': 25},
    'Forêt fermée à mélange de conifères': \
        {'Nom': 'Melange_coniferes', 'Code': 26},
    'Forêt fermée de conifères purs en îlots': \
        {'Nom': 'Coniferes_en_ilots', 'Code': 27},
    'Forêt fermée à mélange de conifères prépondérants et feuillus': \
        {'Nom': 'Melange_de_coniferes_preponderants_et_feuillus', 'Code': 28},
    'Forêt fermée à mélange de feuillus prépondérants et conifères': \
        {'Nom': 'Melange_de_feuillus_preponderants_et_coniferes', 'Code': 29},
}
# Ajout des colonnes via mapping et conversion en DataFrame
df = filtered_samples.join(
    filtered_samples["TFV"].map(lambda x: category_mapping.\
        get(x, {})).apply(pd.Series)
)
df['Code'] = df['Code'].values.astype('uint8')
df_f = df[['Nom', 'Code', 'geometry']]

df.to_file(out_file)
