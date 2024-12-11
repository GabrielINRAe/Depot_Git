import geopandas as gpd
from rasterstats import zonal_stats
import matplotlib.pyplot as plt
import pandas as pd

# Chemins vers le shapefile et le raster
shapefile_path = "/home/onyxia/work/Projet_Teledec/results/data/sample/Sample_BD_foret_T31TCJ.shp"
raster_path = "/home/onyxia/work/Projet_Teledec/results/data/img_pretraitees/masque_foret.tif"

# Charger le shapefile en utilisant geopandas
shapefile = gpd.read_file(shapefile_path)
print(shapefile.columns)

# Calculer les statistiques zonales
stats = zonal_stats(
    shapefile,
    raster_path,
    stats=["count"],      # Nombre de pixels
    categorical=True,     # Regrouper les pixels par catégorie
    geojson_out=False     # Retourner les résultats sous forme de liste
)

# Traiter les statistiques pour obtenir le nombre de pixels par classe
results = []
for i, stat in enumerate(stats):
    classe = shapefile.iloc[i]["classif_pi"]  # Remplacer par le nom du champ de classe
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
plt.savefig("/home/onyxia/work/Projet_Teledec/results/figure/diag_baton_nb_pix_by_class.png")
plt.show()
