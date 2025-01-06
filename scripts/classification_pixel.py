from my_function import train_random_forest, save_classification

# Charger les données
samples = 'results/data/sample/Sample_BD_foret_T31TCJ.shp'
images = 'results/data/img_pretraitees/Serie_temp_S2_allbands.tif'

# Entraîner le modèle
model = train_random_forest(samples, images)

# Sauvegarder la carte
save_classification(model, images, 'results/data/classif/carte_essences_echelle_pixel.tif')
