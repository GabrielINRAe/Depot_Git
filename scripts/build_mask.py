import geopandas as gpd
import numpy as np
import os

f_vege = gpd.read_file('/home/onyxia/work/data/project/FORMATION_VEGETALE.shp')
L_mask = ['Lande','Formation herbacée','Forêt ouverte de conifères purs','Forêt ouverte de feuillus purs','Forêt ouverte sans couvert arboré',
        'Forêt ouverte à mélange de feuillus et conifères','Forêt fermée sans couvert arboré']
ones = np.ones((24041,1),dtype=int)
f_vege['value'] = ones

for i,j in zip(f_vege['TFV'],range(len(f_vege['value']))):
    if i in L_mask:
        #f_vege['mask'][j]=0
        f_vege.loc[j,'value'] = 0

for i in range(len(f_vege['value'])):
    if f_vege['value'][i] == 1:
        f_vege['Classe'] = 'Zone de forêt'
    else:
        f_vege['Classe'] = 'Zone hors forêt'

Masque = f_vege[['ID','Classe','value','geometry']]
Masque.loc[:,'value'] = Masque['value'].astype('uint8')
Masque.to_file('/home/onyxia/work/data/project/mask_traite.shp')  ##Potentiellement ça c'est un probleme si le prof à pas les dossiers projects
# Change et met dans ton dossier output comme pour pretraitement

shp = gpd.read_file('/home/onyxia/work/data/project/emprise_etude.shp')

## Rasterization
my_folder = '/home/onyxia/work'
in_vector = os.path.join(my_folder, 'data/project/mask_traite.shp')
out_image = os.path.join(my_folder, 'Depot_Git/results/data/img_pretraitees/masque_foret.tif')
field_name = 'value'  # field containing the numeric label of the classes

sptial_resolution = 10
xmin,ymin,xmax,ymax=shp.total_bounds
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
os.system(cmd)3