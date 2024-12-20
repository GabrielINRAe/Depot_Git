import os
import sys 
sys.path.append('/home/onyxia/Depot_Git/scripts')
import numpy as np
from osgeo import gdal
from my_function import (
    clip_raster_with_shapefile,
    reproject_and_resample,
    compute_ndvi,
    save_raster)

#  Définition des paramètres 

input_raster_dir = "/home/onyxia/work/data/images"
shapefile_path = "/home/onyxia/work/data/project/emprise_etude.shp"
mask_foret_path = "/home/onyxia/Depot_Git/results/data/img_pretraitees/masque_foret.tif"
output_dir = "/home/onyxia/work"

os.makedirs(output_dir, exist_ok=True)

# Partie 1 : Image multibande
output_multiband_path = os.path.join(output_dir, "Serie_temp_S2_allbands.tif")
bands_to_use = ["2", "3", "4", "5", "6", "7", "8", "8A", "11", "12"] #Pas de bandes 9 c'ets la 8A

# Création d'une image multibande
driver = gdal.GetDriverByName('GTiff')
with gdal.Open(os.path.join(input_raster_dir, "SENTINEL2B_20220125-105852-948_L2A_T31TCJ_C_V3-0_SRE_B2.tif")) as ref_raster:
    geo_transform = ref_raster.GetGeoTransform()
    projection = ref_raster.GetProjection()

  
    out_raster = driver.Create(output_multiband_path, ref_raster.RasterXSize, ref_raster.RasterYSize,
                               len(bands_to_use) * 6, gdal.GDT_UInt16)
    out_raster.SetGeoTransform(geo_transform)
    out_raster.SetProjection(projection)

    band_idx = 1
    for date_idx in range(6):
        for band in bands_to_use:
            raster_path = f"{input_raster_dir}/date_{date_idx}_band_{band}.tif"
            with gdal.Open(raster_path) as src_band:
                out_raster.GetRasterBand(band_idx).WriteArray(src_band.ReadAsArray())
                band_idx += 1
    del out_raster

# Découper selon le shapefile
clip_raster_with_shapefile(output_multiband_path, shapefile_path, output_multiband_path)

# Application de masque forêt
with gdal.Open(output_multiband_path) as src, gdal.Open(mask_foret_path) as mask:
    mask_data = mask.ReadAsArray()
    out_image = src.ReadAsArray()
    out_image[:, mask_data == 0] = 0  # Masquage des pixels qui correspondent pas à une classe forêt
    save_raster(out_image, output_multiband_path, output_multiband_path, gdal.GDT_UInt16, 0)

# Partie 2 : NDVI
output_ndvi_path = os.path.join(output_dir, "Serie_temp_S2_ndvi.tif")

# Calcul de NDVI pour chaque date
with gdal.Open(output_multiband_path) as src:
    geo_transform = src.GetGeoTransform()
    projection = src.GetProjection()
    ndvi_meta = {
        "geo_transform": geo_transform,
        "projection": projection,
        "dtype": gdal.GDT_Float32,
        "nodata": -9999
    }

    ndvi_data = driver.Create(output_ndvi_path, src.RasterXSize, src.RasterYSize, 6, gdal.GDT_Float32)
    ndvi_data.SetGeoTransform(geo_transform)
    ndvi_data.SetProjection(projection)

    for date_idx in range(6):
        nir_band = src.GetRasterBand(7 + date_idx * 10).ReadAsArray()  # Bande NIR (B8)
        red_band = src.GetRasterBand(3 + date_idx * 10).ReadAsArray()  # Bande rouge (B4)
        ndvi = compute_ndvi(nir_band, red_band)
        ndvi_data.GetRasterBand(date_idx + 1).WriteArray(ndvi)
        ndvi_data.GetRasterBand(date_idx + 1).SetNoDataValue(-9999)

    del ndvi_data