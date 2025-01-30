import os
import sys
from osgeo import gdal
import geopandas as gpd
import numpy as np
sys.path.append('/home/onyxia/work/libsigma')
import read_and_write as rw
sys.path.append('/home/onyxia/Depot_Git/scripts')
from my_function import (
    pre_traitement_img,
    supprimer_dossier_non_vide,
    compute_ndvi,
)
racine = "/home/onyxia/work"

# Partie Serie_temp_S2_allbands.tif
print("Partie pretraitement image")

# Définition des paramètres
print("Définitions des paramètres")
input_raster_dir = os.path.join(racine, "data/images")
l_images = sorted(os.listdir(input_raster_dir))
if ".keep" in l_images:
    l_images.remove(".keep")
shapefile_path = os.path.join(racine, "data/project/emprise_etude.shp")

output_dir = os.path.join(racine, "output_pretraitement")
os.makedirs(output_dir, exist_ok=True)

masque_path = os.path.join(racine, "results/data/img_pretraitees/masque_foret.tif")

# Réalisation des pré-traitements sur les images individuelles
pre_traitement_img(
    p_emprise=shapefile_path,
    l_images=l_images,
    input_raster_dir=input_raster_dir,
    output_dir=output_dir
)

# Construction array
print("Construction de l'array")
ref_raster_path = os.path.join(output_dir, "traitement_20220125_B2.tif")
L_images_clip = sorted(os.listdir(output_dir))
x, y = rw.get_image_dimension(rw.open_image(ref_raster_path))[:2]
bandes = 60
array_tot = np.zeros((x, y, bandes))

masque = rw.load_img_as_array(masque_path)

# Initialisation de la liste pour stocker les arrays masqués
L_array_masqued = []

# Parcourir toutes les images de L_images_clip
for img in L_images_clip:
    path = os.path.join(output_dir, img)
    array = rw.load_img_as_array(path)
    array_masqued = np.where(masque == 1, array, 0)
    L_array_masqued.append(array_masqued)

# Concaténation des arrays masqués
print("Concaténation en cours")
array_final_masqued = np.concatenate(L_array_masqued, axis=2)
print("Tableau concaténé avec masque appliqué")

# Sauvegarde de l'array en image
out_masqued = os.path.join(racine, "results/data/img_pretraitees/Serie_temp_S2_allbands.tif")
print("Écriture en cours")
rw.write_image(
    out_filename=out_masqued,
    array=array_final_masqued,
    data_set=rw.open_image(ref_raster_path)
)
print("Écriture terminée")

# Partie Serie_temp_S2_ndvi.tif
print("Partie NDVI")

# Définition des paramètres
traitements_dir = output_dir
l_traitements = [os.path.join(traitements_dir, i) for i in sorted(os.listdir(traitements_dir))]
ref_raster_path = os.path.join(traitements_dir, "traitement_20220125_B2.tif")
masque = rw.load_img_as_array(masque_path)

ndvi_masked = compute_ndvi(masque, ref_raster_path, l_traitements)

# Sauvegarde de l'array en image
out_ndvi = os.path.join(racine, "results/data/img_pretraitees/Serie_temp_S2_ndvi.tif")
rw.write_image(
    out_filename=out_ndvi,
    array=ndvi_masked,
    data_set=rw.open_image(ref_raster_path),
    gdal_dtype=gdal.GDT_Float32
)
print("Écriture terminée")

# Nettoyage des dossiers
supprimer_dossier_non_vide(traitements_dir)
