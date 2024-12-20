import geopandas as gpd
from osgeo import gdal
import os
import numpy as np
from my_function import supprimer_dossier_non_vide
import sys
sys.path.append('/home/onyxia/work/libsigma')
import read_and_write as rw

#  Définition des paramètres 

input_raster_dir = "/home/onyxia/work/data/images"
L_images = sorted(os.listdir(input_raster_dir))
if ".keep" in L_images:
    L_images.remove(".keep")
shapefile_path = "/home/onyxia/work/data/project/emprise_etude.shp"

output_dir = "/home/onyxia/work/output"
os.makedirs(output_dir, exist_ok=True)

masque_path = "/home/onyxia/work/Depot_Git/results/data/img_pretraitees/masque_foret.tif"

# Charger le vecteur avec Geopandas
emprise = gpd.read_file(shapefile_path).to_crs("EPSG:2154")

# Extraire le GeoJSON sous forme de string
geojson_str = emprise.to_json()

for i,img in enumerate(L_images) :
    date = img[11:19]
    bande = img[53:]
    ds_img = rw.open_image(os.path.join(input_raster_dir,img))
    name_file = f"traitement_{date}_{bande}"
    output_file = os.path.join(output_dir,name_file)
    # Appliquer le clip avec GDAL
    resolution = 10  # Résolution (10 m)
    output_raster_test = gdal.Warp(
        output_file, # Chemin de fichier, car on utilise GTiff
        # "",  # Pas de chemin de fichier, car on utilise MEM
        ds_img,  # Fichier en entrée (chemin ou objet gdal)
        format = "GTiff", # Utiliser GTiff comme format
        # format = "MEM",  # Utiliser MEM comme format
        cutlineDSName = geojson_str,  # Passer directement le GeoJSON
        cropToCutline = True,
        outputType = gdal.GDT_UInt16, #UInt16
        dstSRS = "EPSG:2154",  # Reprojection
        xRes = resolution,  # Résolution X
        yRes = resolution,  # Résolution Y
        dstNodata = 0  # Valeur NoData
    )
    print(f"Image {i+1}/{len(L_images)} traitée")
ds_img = None
emprise = None
geojson_str = None

# Construction array
ref_raster_path = "/home/onyxia/work/output/traitement_20220125_B2.tif"
L_images_clip = sorted(os.listdir(output_dir))
x,y = rw.get_image_dimension(rw.open_image(ref_raster_path))[:2]
bandes = 60
array_tot = np.zeros((x,y,bandes))

masque = rw.load_img_as_array(masque_path)

L_array = []
for img in L_images_clip:
    path = os.path.join(output_dir,img) 
    array = rw.load_img_as_array(path)
    L_array.append(array)    # Sans le masque
    # array_masqued = array * masque    # Avec le masque
    # L_array.append(array_masqued)     # Avec le masque

# Concat array
print("Concaténation en cours")
array_final = np.concatenate(L_array,axis = 2)
print("Tableau concaténé")

# Save array into image
out = "/home/onyxia/work/Depot_Git/results/data/img_pretraitees/Serie_temp_S2_allbands_nomask.tif"
print("Ecriture en cours")
rw.write_image(out_filename=out, array = array_final, data_set = rw.open_image(ref_raster_path))
print("Ecriture terminée")

# Nettoyage des dossiers
supprimer_dossier_non_vide("/home/onyxia/work/output")