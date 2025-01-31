import os
import sys
import geopandas as gpd
import pandas as pd
from rasterstats import zonal_stats
sys.path.append('/home/onyxia/work/Depot_Git/scripts')
from my_function import (
    count_polygons_by_class,
    plot_bar,
    violin_plot,
    sel_classif_pixel
)

# Définition des paramètres
racine = '/home/onyxia/work'
my_folder = os.path.join(racine, 'results/sample')
out_folder = os.path.join(racine, 'results/figure')
os.makedirs(out_folder, exist_ok=True)

in_vector = os.path.join(my_folder, 'Sample_BD_foret_T31TCJ.shp')
diag_baton_poly_classe_path = os.path.join(out_folder, 'diag_baton_nb_poly_by_class.png')
diag_baton_pixel_classe_path = os.path.join(out_folder, 'diag_baton_nb_pix_by_class.png')
violin_plot_path = os.path.join(out_folder, 'violin_plot_nb_pix_by_poly_by_class.png')
violin_plot_filt_path = os.path.join(out_folder, 'violin_plot_nb_pix_by_poly_by_class_filtred.png')
raster_path = os.path.join(racine, "results/data/img_pretraitees/masque_foret.tif")

# Chargement des données de BD_Forêt
echantillons = gpd.read_file(in_vector)

# Filtrage des classes pour la classification pixel
echantillons_px = sel_classif_pixel(echantillons)

# Définition du nom de la colonne des classifications de polygones
nom_poly_col = "Nom"

# Comptage du nombre de polygones par classe
nb_pol_by_class = count_polygons_by_class(echantillons_px, nom_poly_col)
nb_pol_by_class = nb_pol_by_class.set_index(['Nom'])

# Visualisation graphique des polygones par classe
plot_bar(
    nb_pol_by_class,
    title="Nombre de polygones par classe",
    xlabel="Classe",
    ylabel="Nombre de polygones",
    output_path=diag_baton_poly_classe_path
)

# Rasterisation et comptage des pixels par classe
stats = zonal_stats(
    echantillons_px,
    raster_path,
    stats=["count"],
    categorical=False,
    geojson_out=False,
    nodata=0
)

# Association des classes aux résultats de zonal_stats
results = []
for i, stat in enumerate(stats):
    classe = echantillons_px.iloc[i]["Nom"]
    if stat:
        for category, count in stat.items():
            results.append({"Classe": classe, "Catégorie": category, "Pixels": count})
    else:
        results.append({"Classe": classe, "Catégorie": "N/A", "Pixels": 0})

# Conversion en DataFrame et regroupement par classe
df = pd.DataFrame(results)
nb_pixel_by_class = df.groupby("Classe")["Pixels"].sum().reset_index()
nb_pixel_by_class = nb_pixel_by_class.set_index(['Classe'])

# Visualisation graphique du nombre de pixels par classe
plot_bar(
    nb_pixel_by_class,
    title="Nombre de pixels par classe",
    xlabel="Classe",
    ylabel="Nombre de pixels",
    output_path=diag_baton_pixel_classe_path
)

# Création d'un violin plot pour la distribution des pixels par polygone
diag_title = "Distribution du nombre de pixels par polygone, par classe"
violin_plot(
    df=df,
    x_col="Classe",
    y_col="Pixels",
    output_file=violin_plot_path,
    title=diag_title,
    xlabel="Classe",
    ylabel="Nombre de pixels par polygone",
    palette="muted"
)

# Création d'un violin plot sans la classe dominante "Chêne"
df_filtered = df[df["Classe"] != "Chene"]
violin_plot(
    df=df_filtered,
    x_col="Classe",
    y_col="Pixels",
    output_file=violin_plot_filt_path,
    title=diag_title,
    xlabel="Classe",
    ylabel="Nombre de pixels par polygone",
    palette="muted"
)