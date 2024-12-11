# Importation des bibliothèques
from my_function import (
    count_polygons_by_class,
    plot_bar,
    violin_plot)
import os
import geopandas as gpd
import pandas as pd
from rasterstats import zonal_stats
import sys
sys.path.append('/home/onyxia/work/Projet_Teledec/scripts')

# Définition des paramètres
my_folder = '/home/onyxia/work/Projet_Teledec/results/data/sample'
out_folder = '/home/onyxia/work/Projet_Teledec/results/figure'
in_vector = os.path.join(my_folder, 'Sample_BD_foret_T31TCJ.shp')
diag_baton_poly_classe_path = os.path.join(out_folder, 'diag_baton_nb_poly_by_class.png')
diag_baton_pixel_classe_path = os.path.join(out_folder, 'diag_baton_nb_pix_by_class.png')
violin_plot_path = os.path.join(out_folder, 'violin_plot_nb_pix_by_poly_by_class.png')
violin_plot_filt_path = os.path.join(out_folder, 'violin_plot_nb_pix_by_poly_by_class_filtred.png')
raster_path = "/home/onyxia/work/Projet_Teledec/results/data/img_pretraitees/masque_foret.tif"

# Chargement des données de BD_Forêt
echantillons = gpd.read_file(in_vector)
echantillons.head()

# Définition d'une variable stockant le nom de la colonne de classification des polygones
nom_poly_col = "classif_ob"

# Comptage du nombre de polygones par classe
nb_pol_by_class = count_polygons_by_class(echantillons, nom_poly_col)
print(nb_pol_by_class)

# Visualisation graphique de la distribution 
# des polygones sur les différentes classes sélectionnées
plot_bar(
    nb_pol_by_class,
    title="Nombre de polygones par classe",
    xlabel="Classe",
    ylabel="Nombre de polygones",
    output_path=diag_baton_poly_classe_path
)

# Rastérisation de la couche des échantillons à l'aide de la fonction zonal_stat 
# et calcul de l'effectif de pixels pour chaque classe
stats = zonal_stats(
    echantillons,
    raster_path,
    stats=["count"],  # Nombre de pixels
    categorical=True,  # Regrouper les pixels par catégorie
    geojson_out=False  # Retourner les résultats sous forme de liste
)

# Pour chaque catégorie dans la liste stats générée précédemment, on associe la classe
# correspondante pour obtenir le nombre de pixels par classe
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
nb_pixel_by_class = df.groupby("Classe")["Pixels"].sum().reset_index()

# Affichage des résultats
print(nb_pixel_by_class)

# Visualisation graphique sous forme d'un diagramme en bâton de la distribution des pixels sur les différentes classes
plot_bar(
    nb_pixel_by_class,
    title="Nombre de pixels par classe",
    xlabel="Classe",
    ylabel="Nombre de pixels",
    output_path=diag_baton_pixel_classe_path
)

# Création de "violin plot" pour visualiser la distribution du nombre de pixels par polygone, par classe
violin_plot(
    df=df,
    x_col="Classe",
    y_col="Pixels",
    output_file=violin_plot_path,
    title="Distribution du nombre de pixels par polygone, par classe",
    xlabel="Classe",
    ylabel="Nombre de pixels par polygone",
    palette="muted"
)

# Création de "violin plot" pour visualiser la distribution du nombre de pixels par polygone, par classe sans tenir compte de la classe dominante "Chêne"
df_filtered = df[df["Classe"] != "Chêne"]
violin_plot(
    df=df_filtered,
    x_col="Classe",
    y_col="Pixels",
    output_file=violin_plot_filt_path,
    title="Distribution du nombre de pixels par polygone, par classe",
    xlabel="Classe",
    ylabel="Nombre de pixels par polygone",
    palette="muted"
)
