import geopandas as gpd
from osgeo import gdal
import os
import numpy as np
from my_function import (
    pre_traitement_img,
    supprimer_dossier_non_vide)
import sys
sys.path.append('/home/onyxia/work/libsigma')
import read_and_write as rw

racine = "/home/onyxia/work"

## Partie Serie_temp_S2_allbands.tif
print("Partie pretraitement image")
#  Définition des paramètres 

print("Définitions des paramètres")
input_raster_dir = os.path.join(racine,"data/images")
l_images = sorted(os.listdir(input_raster_dir))
if ".keep" in l_images:
    l_images.remove(".keep")
shapefile_path = os.path.join(racine,"data/project/emprise_etude.shp")

output_dir = os.path.join(racine,"output_pretraitement")
os.makedirs(output_dir, exist_ok=True)

masque_path = os.path.join(racine,"results/data/img_pretraitees/masque_foret.tif")

# Réalisation des pré-traitements sur les images individuelles 
pre_traitement_img(
    p_emprise = shapefile_path,
    l_images = l_images,
    input_raster_dir = input_raster_dir,
    output_dir = output_dir)

# Construction array
print("Construction de l'array")
ref_raster_path = os.path.join(output_dir,"traitement_20220125_B2.tif")
L_images_clip = sorted(os.listdir(output_dir))
x,y = rw.get_image_dimension(rw.open_image(ref_raster_path))[:2]
bandes = 60
array_tot = np.zeros((x,y,bandes))

masque = rw.load_img_as_array(masque_path)

# Initialisation de la liste pour stocker les arrays masqués
L_array_masqued = []

# Parcourir toutes les images de L_images_clip
for i, img in enumerate(L_images_clip):
    path = os.path.join(output_dir, img)
    array = rw.load_img_as_array(path)
    array_masqued = np.where(masque == 1, array, 0)
    L_array_masqued.append(array_masqued)

# Concaténation des arrays masqués
print("Concaténation en cours")
array_final_masqued = np.concatenate(L_array_masqued, axis=2)
print("Tableau concaténé avec masque appliqué")

# Save array into image
out_masqued = os.path.join(racine,"results/data/img_pretraitees/Serie_temp_S2_allbands.tif")
print("Ecriture en cours")
rw.write_image(out_filename=out_masqued, array=array_final_masqued, data_set=rw.open_image(ref_raster_path))
print("Ecriture terminée")


## Partie Serie_temp_S2_ndvi.tif
print("Partie ndvi")

# Definitions des paramètres
traitements_dir = output_dir
L_traitements = [os.path.join(traitements_dir,i) for i in sorted(os.listdir(traitements_dir))]
# L_traitements = os.path.join(traitements_dir,sorted(os.listdir(traitements_dir)))
ref_raster_path = os.path.join(traitements_dir,"traitement_20220125_B2.tif")
masque_path = "/home/onyxia/work/results/data/img_pretraitees/masque_foret.tif"
masque = rw.load_img_as_array(masque_path)

# Pour les 6 dates 
x,y = rw.get_image_dimension(rw.open_image(ref_raster_path))[:2]
bandes = 6

dates = ["20220125","20220326","20220405","20220714","20220922","20221111"] # Liste des 6 dates
nir_name = 'B8.'
r_name = 'B4.'

ndvi_blank = np.zeros((x,y,bandes), dtype=np.float32)  # Créer un array NDVI avec les mêmes dimensions que nir

print("Calcul des NDVI")
for i,date in enumerate(dates) :
    print(i,date)
    for img in L_traitements:
        if date in img and r_name in img :
            red = rw.load_img_as_array(img)[:,:,0].astype('float32')
            print (f"Bande rouge date {date}")
        if date in img and nir_name in img :
            nir = rw.load_img_as_array(img)[:,:,0].astype('float32')
            print (f"Bande infra-rouge date {date}")
    nominator = nir-red
    nominator_masked = np.where(nominator>=0,nominator,0)
    denominator = nir+red
    ndvi_blank[:,:,i] = np.where(denominator!=0,nominator_masked/denominator,0)
ndvi_masked = np.where(masque == 1, ndvi_blank, -9999)

# Save array into image
out_ndvi = "/home/onyxia/work/results/data/img_pretraitees/Serie_temp_S2_ndvi.tif"
print("Ecriture en cours")
rw.write_image(out_filename=out_ndvi, array=ndvi_masked, data_set=rw.open_image(ref_raster_path), gdal_dtype=gdal.GDT_Float32)
print("Ecriture terminée")


## Nettoyage des dossiers
supprimer_dossier_non_vide(traitements_dir)