import geopandas as gpd
import numpy as np
import os
from my_function import masque_shp,supprimer_dossier_non_vide

# Paramètrage des paths
racine = "/home/onyxia/work"    # Racine du projet
output_dir = os.path.join(racine, "output_masque")    # Dossier de sortie
os.makedirs(output_dir, exist_ok=True)       # Crée le dossier output temporaire
path_f_vege = os.path.join(racine,"data/project/FORMATION_VEGETALE.shp")    # Path pour le fichier shp formation végétale
path_masque_traite = os.path.join(output_dir,'mask_traite.shp')    # Path pour le fichier shp masque traité 

masque_shp(path_f_vege,path_masque_traite)

emprise = gpd.read_file(os.path.join(racine,"data/project/emprise_etude.shp"))

## Rasterization
my_folder = '/home/onyxia/work'
in_vector = path_masque_traite
out_image = os.path.join(my_folder, 'Depot_Git/results/data/img_pretraitees/masque_foret.tif')
field_name = 'value'  # field containing the numeric label of the classes

sptial_resolution = 10
xmin,ymin,xmax,ymax=emprise.total_bounds
# xmin = shp['MINX'][0]
# ymin = shp['MINY'][0]
# xmax = shp['MAXX'][0]
# ymax = shp['MAXY'][0]

# Créer le répertoire de sortie si nécessaire
out_dir = os.path.dirname(out_image)
os.makedirs(out_dir, exist_ok=True)  # Crée les répertoires manquants

# define command pattern to fill with paremeters
cmd_pattern = ("gdal_rasterize -a {field_name} "
               "-tr {sptial_resolution} {sptial_resolution} "
               "-te {xmin} {ymin} {xmax} {ymax} -ot Byte -of GTiff "
               "{in_vector} {out_image}")

# fill the string with the parameter thanks to format function
cmd = cmd_pattern.format(in_vector=in_vector, xmin=xmin, ymin=ymin, xmax=xmax,
                         ymax=ymax, out_image=out_image, field_name=field_name,
                         sptial_resolution=sptial_resolution)

# execute the command in the terminal
os.system(cmd)

supprimer_dossier_non_vide(output_dir)