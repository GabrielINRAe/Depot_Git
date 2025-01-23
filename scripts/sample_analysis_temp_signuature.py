# Importation des bibliothèques nécessaires
import os
import numpy as np
import matplotlib.pyplot as plt

# Personal libraries
import sys 
sys.path.append('/home/onyxia/work/libsigma')
import classification as cla
import read_and_write as rw

# Définition des paramètres
my_folder = '/home/onyxia/work/Depot_Git/results/data'
in_vector = os.path.join(my_folder, 'sample/Sample_BD_foret_T31TCJ.shp')
ref_image = os.path.join(my_folder, 'img_pretraitees/Serie_temp_S2_ndvi.tif')
out_image = os.path.splitext(in_vector)[0] + '_v2.tif'
field_name = 'Code'  # field containing the numeric label of the classes
output_path = os.path.join(my_folder, "../figure/temp_mean_ndvi.png")

# Caractéristiques de raster de référence
sptial_resolution = 10
xmin = 501127.9696999999
ymin = 6240654.023599998
xmax = 609757.9696999999
ymax = 6314464.023599998

# Définition de la commande "pattern" avec les paramètres de raster de référence
cmd_pattern = ("gdal_rasterize -a {field_name} "
               "-tr {sptial_resolution} {sptial_resolution} "
               "-te {xmin} {ymin} {xmax} {ymax} -ot Byte -of GTiff "
               "{in_vector} {out_image}")

cmd = cmd_pattern.format(in_vector=in_vector, xmin=xmin, ymin=ymin, xmax=xmax,
                         ymax=ymax, out_image=out_image, field_name=field_name,
                         sptial_resolution=sptial_resolution)

# Exécution de la commande
os.system(cmd)

# Définition des chemins d'accès vers les couches d'entrée
sample_filename = os.path.join(my_folder, 'sample/Sample_BD_foret_T31TCJ_v2.tif')
image_filename = os.path.join(my_folder, 'img_pretraitees/Serie_temp_S2_ndvi.tif')
X, Y, t = cla.get_samples_from_roi(image_filename, sample_filename)
print(X.shape)
print(Y.shape)

# Vérification des valeurs de X
min_X = X.min()
max_X = X.max()
print(f"Valeur minimale de X: {min_X}")
print(f"Valeur maximale de X: {max_X}")

# Vérification des valeurs de raster NDVI
out_of_range = X[(X < -1) | (X > 1)]
if len(out_of_range) > 0:
    print(f"Valeurs de NDVI non incluses dans l'intervalle [-1,1]: {out_of_range}")
else:
    print("Pas de valeurs de NDVI en dehors de l'intervalle [-1,1]")

# Conversion de t à une liste de tuples (x, y)
coords = list(zip(t[0], t[1]))
print(coords[:5])

Y = Y.flatten()

# Vérification de la taille des matrices X et Y
print(X.shape)
print(Y.shape)

# Liste de codes correspondant aux classes d'intérêt
list_of_interest = ['12', '13', '14', '23', '24', '25']

# Filtrage des échantillons pour ne garder que les classes d'intérêt
Y_cleaned = np.array([str(y).strip() for y in Y])  # Supprimer les espaces dans les labels
mask = np.isin(Y.astype(str), list_of_interest)

X_filtered = X[mask]
Y_filtered = Y[mask]

# Vérification des tailles après filtrage
print("Shape de X_filtered:", X_filtered.shape)
print("Shape de Y_filtered:", Y_filtered.shape)
print("Les cinq premiers labels dans Y:", Y[:5])

codes_of_interest = [12, 13, 14, 23, 24, 25]

# Liste des dates relatives à chaque bande NDVI
dates = ['2023-01-01', '2023-03-01', '2023-05-01', '2023-07-01', '2023-09-01', '2023-11-01']

# Codes de couleur pour chaque classe
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# Labels des classes
class_names = ['Chêne', 'Robinier', 'Peupleraie', 'Douglas', 'Pin laricio ou pin noir', 'Pin maritime']

# Préparation du graphique
fig, ax = plt.subplots(figsize=(10, 6))

# Calcul et affichage des moyennes et écarts-types
for idx, code in enumerate(codes_of_interest):
    X_class = X_filtered[Y_filtered == code]
    if X_class.shape[0] > 0:
        means = X_class.mean(axis=0)
        stds = X_class.std(axis=0)
        ax.plot(dates, means, color=colors[idx], label=class_names[idx])
        ax.fill_between(dates, means + stds, means - stds, color=colors[idx], alpha=0.3)

# Configuration des axes et titre
ax.set_xlabel('Date')
ax.set_ylabel('Moyenne de NDVI')
ax.set_title('NDVI moyen par Classe et Date avec Écart-Type')
ax.set_ylim(0, 1)
ax.legend(title="Classes", loc='upper left')

plt.xticks(rotation=45)
plt.tight_layout()

# Enregistrement du graphique
plt.savefig(out_image, dpi=300)
print(f"Graphique enregistré dans : {output_path}")

plt.show()

# Vérification de la quantité d'échantillons par classe et par date
for idx, code in enumerate(codes_of_interest):
    X_class = X_filtered[Y_filtered == code]
    print(f"Classe : {class_names[idx]} (Code : {code})")
    print(f"Nombre d'échantillons pour cette classe : {X_class.shape[0]}")
    if X_class.shape[0] > 0:
        print(f" les cinq premiers échantillons : {X_class[:5]}")
    else:
        print("Il n'y a pas d'échantillons pour cette classe.")
        
# Vérification de la variance pour chaque classe 
for idx, code in enumerate(codes_of_interest):
    X_class = X_filtered[Y_filtered == code]
    if X_class.shape[0] > 0:
        stds = X_class.std(axis=0)
        print(f"Classe : {class_names[idx]} (Code : {code})")
        print(f"Écart-type par date : {stds}")
        if np.all(stds == 0):
            print("L'écart-type est 0 pour toutes les dates.")