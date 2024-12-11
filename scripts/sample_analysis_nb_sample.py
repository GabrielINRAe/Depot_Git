# Importation des Bibliothèques 
import os
import sys
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rasterstats import zonal_stats
sys.path.append('/home/onyxia/work/Projet_Teledec/scripts')
from my_function import (
    count_pixels_by_class,
    count_polygons_by_class,
    plot_bar,
    plot_violin,
)

# definition des paramètres 
 
my_folder = '/home/onyxia/work/Projet_Teledec/results/data/sample'
out_folder = '/home/onyxia/work/Projet_Teledec/results/figure'
in_vector = os.path.join(my_folder, 'Sample_BD_foret_T31TCJ.shp')
diag_baton_poly_classe_path = os.path.join(out_folder,'diag_baton_nb_poly_by_class.png')
diag_baton_pixel_classe_path = os.path.join(out_folder,'diag_baton_nb_pix_by_class.png')
violin_plot_pix_by_poly_by_class_path = os.path.join(out_folder,'violin_plot_nb_pix_by_poly_by_class.png')
violin_plot_pix_by_poly_by_class_path_filtred = os.path.join(out_folder,'violin_plot_nb_pix_by_poly_by_class_filtred.png')
raster_path = "/home/onyxia/work/Projet_Teledec/results/data/img_pretraitees/masque_foret.tif"

# Chargement des données de BD_Forêt
echantillons = gpd.read_file(in_vector)
echantillons.head()

# Visualisation sous forme d'un diagramme en bâton du nombre des polygones par classe 
# Définition des un variable stockant le nom de colone classif polygone
nom_poly_col = "classif_ob" 
# Comptage de  nombre des polygones par classe
nb_pol_by_class = count_polygons_by_class(echantillons, nom_poly_col)
print(nb_pol_by_class)

# Visualisation Grapique de distribution des polygones sur les différentes classes 
plot_bar(
    nb_pol_by_class,
    title = "Nombre de polygones par classe",
    xlabel = "Classe",
    ylabel = "Nombre de polygones",
    output_path = diag_baton_poly_classe_path)

# Rastérisation de couche vecteur à l'aide de la fonction zonal_stat et calcul statistique de l'effectif de pixels pour chaque classe 
stats = zonal_stats(
    echantillons,
    raster_path,
    stats=["count"],      # Nombre de pixels
    categorical=True,     # Regrouper les pixels par catégorie
    geojson_out=False     # Retourner les résultats sous forme de liste
)

# Traiter les statistiques pour obtenir le nombre de pixels par classe
results = []
for i, stat in enumerate(stats):
    classe = echantillons.iloc[i]["classif_pi"]  # Remplacer par le nom du champ de classe
    if stat:
        for category, count in stat.items():
            results.append({"Classe": classe, "Catégorie": category, "Pixels": count})
    else:
        results.append({"Classe": classe, "Catégorie": "N/A", "Pixels": 0})
# Convertir les résultats en DataFrame et regrouper par classe
df = pd.DataFrame(results)
grouped = df.groupby("Classe")["Pixels"].sum().reset_index()

# Créer un graphique à barres avec matplotlib
plt.figure(figsize=(10, 6))
plt.bar(grouped["Classe"], grouped["Pixels"], color="skyblue")

# Ajouter les étiquettes et le titre
plt.xlabel("Classe", fontsize=12)
plt.ylabel("Nombre de pixels", fontsize=12)
plt.title("Nombre de pixels par classe", fontsize=14)
plt.xticks(rotation=45, ha="right")

# Enregistrer et afficher le graphique
plt.tight_layout()
plt.savefig(diag_baton_pixel_classe_path)
plt.show()

# Créer le "violin plot" de la distribution du nombre de pixels par polygone, par classe
plt.figure(figsize=(12, 8))
sns.violinplot(
    data=df,
    x="Classe",
    y="Pixels",
    palette="muted"
)

# Ajouter les étiquettes et le titre
plt.xlabel("Classe", fontsize=12)
plt.ylabel("Nombre de pixels par polygone", fontsize=12)
plt.title("Distribution du nombre de pixels par polygone, par classe", fontsize=14)
plt.xticks(rotation=45, ha="right")

# Enregistrer et afficher le graphique
plt.tight_layout()
plt.savefig("violin_plot_nb_pix_by_poly_by_class.png")
plt.show()

# Créer le "violin plot" de la distribution du nombre de pixels par polygone, par classe sans chene
df_filtered = df[df["Classe"] != "Chêne"]

plt.figure(figsize=(12, 8))
sns.violinplot(
    data=df_filtered,
    x="Classe",
    y="Pixels",
    palette="muted"
)


plt.xlabel("Classe", fontsize=12)
plt.ylabel("Nombre de pixels par polygone", fontsize=12)
plt.title("Distribution du nombre de pixels par polygone, par classe (sans 'Chêne')", fontsize=14)
plt.xticks(rotation=45, ha="right")


plt.tight_layout()
plt.savefig(violin_plot_pix_by_poly_by_class_path_filtred)
plt.show()