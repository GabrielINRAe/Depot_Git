import geopandas as gpd
import numpy as np
import os
from my_function import masque_shp,
    rasterization,supprimer_dossier_non_vide)

# Paramètrage des paths
racine = "/home/onyxia/work"    # Racine du projet
output_dir = os.path.join(racine, "output_masque")    # Dossier de sortie
os.makedirs(output_dir, exist_ok=True)       # Crée le dossier output temporaire
path_f_vege = os.path.join(racine,"data/project/FORMATION_VEGETALE.shp")    # Path pour le fichier shp formation végétale
path_masque_traite = os.path.join(output_dir,'mask_traite.shp')    # Path pour le fichier shp masque traité 
path_emprise = os.path.join(racine,"data/project/emprise_etude.shp")    # Path pour le fichier shp emprise
path_masque_raster = os.path.join(racine, 'Depot_Git/results/data/img_pretraitees/masque_foret.tif')

# Formatage du fichier masque en format shp à partir
masque_shp(path_input=path_f_vege,
    path_output=path_masque_traite)

# Rasterisation du fichier masque
emprise = gpd.read_file(path_emprise)
rasterization(
    in_vector=path_masque_traite,
    out_image=path_masque_raster,
    field_name='value',
    sp_resol=10,
    emprise=emprise,
    dtype='Byte')

supprimer_dossier_non_vide(output_dir)