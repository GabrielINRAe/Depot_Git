import geopandas as gpd
import numpy as np
import os
from my_function import (masque_shp,
    rasterization,supprimer_dossier_non_vide)

# Paramètrage des paths
racine = "/home/onyxia/work"
output_dir = os.path.join(racine, "output_masque")
os.makedirs(output_dir, exist_ok=True)
path_f_vege = os.path.join(racine,"data/project/FORMATION_VEGETALE.shp")
path_masque_traite = os.path.join(output_dir,'mask_traite.shp')
path_emprise = os.path.join(racine,"data/project/emprise_etude.shp")
path_masque_raster = os.path.\
    join(racine,'results/data/img_pretraitees/masque_foret.tif')

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