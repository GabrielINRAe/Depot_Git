import os
import numpy as np
import matplotlib.pyplot as plt
import sys
# Ajout du chemin vers les bibliothèques personnalisées
sys.path.append('/home/onyxia/work/libsigma')
import classification as cla
import read_and_write as rw

# Définition des paramètres
racine = '/home/onyxia/work'
my_folder = os.path.join(racine, 'results/data')
in_vector = os.path.join(my_folder, 'sample/Sample_BD_foret_T31TCJ.shp')
ref_image = os.path.join(my_folder, 'img_pretraitees/Serie_temp_S2_ndvi.tif')
out_image = os.path.splitext(in_vector)[0] + '_v2.tif'
field_name = 'Code'  # Champ contenant le label numérique des classes
output_path = os.path.join(racine, "results/figure/temp_mean_ndvi.png")

# Caractéristiques de raster de référence
sptial_resolution = 10
xmin = 501127.9696999999
ymin = 6240654.023599998
xmax = 609757.9696999999
ymax = 6314464.023599998

# Définition de la commande "pattern"
cmd_pattern = (
    "gdal_rasterize -a {field_name} "
    "-tr {sptial_resolution} {sptial_resolution} "
    "-te {xmin} {ymin} {xmax} {ymax} -ot Byte -of GTiff "
    "{in_vector} {out_image}"
)

cmd = cmd_pattern.format(
    in_vector=in_vector, xmin=xmin, ymin=ymin,
    xmax=xmax, ymax=ymax,
    out_image=out_image,
    field_name=field_name, sptial_resolution=sptial_resolution
)

# Exécution de la commande
os.system(cmd)

# Définition des chemins d'accès vers les couches d'entrée
sample_filename = os.path.join\
    (my_folder, 'sample/Sample_BD_foret_T31TCJ_v2.tif')
image_filename = os.path.join\
    (my_folder, 'img_pretraitees/Serie_temp_S2_ndvi.tif')
X, Y, t = cla.get_samples_from_roi(image_filename, sample_filename)

# Conversion de t à une liste de tuples (x, y)
coords = list(zip(t[0], t[1]))
Y = Y.flatten()

# Liste de codes correspondant aux classes d'intérêt
list_of_interest = ['12', '13', '14', '23', '24', '25']

# Filtrage des échantillons pour ne garder que les classes d'intérêt
# Supprimer les espaces dans les labels
Y_cleaned = np.array([str(y).strip() for y in Y])  
mask = np.isin(Y.astype(str), list_of_interest)

X_filtered = X[mask]
Y_filtered = Y[mask]

codes_of_interest = [12, 13, 14, 23, 24, 25]

# Liste des dates relatives à chaque bande NDVI
dates = ["25/1", "26/3", "5/4", "14/7", "22/9", "11/11"]

# Codes de couleur pour chaque classe
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# Labels des classes
class_names = [
    'Chêne', 'Robinier', 'Peupleraie', 
    'Douglas', 'Pin laricio ou pin noir', 'Pin maritime']

# Préparation du graphique
fig, ax = plt.subplots(figsize=(10, 6))

# Calcul et affichage des moyennes et écarts-types
for idx, code in enumerate(codes_of_interest):
    X_class = X_filtered[Y_filtered == code]
    if X_class.shape[0] > 0:
        means = X_class.mean(axis=0)
        stds = X_class.std(axis=0)
        ax.plot(dates, means, color=colors[idx], label=class_names[idx])
        ax.fill_between(
            dates, means + stds, means - stds, 
            color=colors[idx], alpha=0.3)

# Configuration des axes et titre
ax.set_xlabel('Date')
ax.set_ylabel('Moyenne de NDVI')
ax.set_title('NDVI moyen par Classe et Date avec Écart-Type')
ax.set_ylim(0, 1)
ax.legend(title="Classes", loc='upper left')

plt.xticks(rotation=45)
plt.tight_layout()

# Enregistrement du graphique
plt.savefig(output_path, dpi=300)
print(f"Graphique enregistré dans : {output_path}")