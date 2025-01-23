import os
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
# personal librairies
import sys 
sys.path.append('/home/onyxia/work/libsigma')
import classification as cla
import read_and_write as rw

# define parameters
my_folder = '/home/onyxia/work/Depot_Git/results/data'
in_vector = os.path.join(my_folder, 'sample/Sample_BD_foret_T31TCJ.shp')
ref_image = os.path.join(my_folder, 'img_pretraitees/Serie_temp_S2_ndvi.tif')
out_image = os.path.splitext(in_vector)[0] + '_v2.tif'
field_name = 'Code'  # field containing the numeric label of the classes
output_path = os.path.join(my_folder, "../figure/temp_mean_ndvi.png")

# for those parameters, you know how to get theses information if you had to
sptial_resolution = 10
xmin = 501127.9696999999
ymin = 6240654.023599998
xmax = 609757.9696999999
ymax = 6314464.023599998

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
os.system(cmd)

my_folder = '/home/onyxia/work/Depot_Git/results/data'
sample_filename = os.path.join(my_folder, 'sample/Sample_BD_foret_T31TCJ_v2.tif')
image_filename = os.path.join(my_folder, 'img_pretraitees/Serie_temp_S2_ndvi.tif')
X, Y, t = cla.get_samples_from_roi(image_filename, sample_filename)
print(X.shape)
print(Y.shape)

# Convertir t a una lista de tuplas (x, y)
coords = list(zip(t[0], t[1]))

Y = Y.flatten()

list_of_interest = ['12', '13', '14', '23', '24', '25']
mask = np.isin(Y.astype(str), list_of_interest)
X_filtered = X[mask]  # Seleccionar solo las muestras correspondientes
Y_filtered = Y[mask]


# Verificar el tamaño después del filtro
print("Shape de X_filtered:", X_filtered.shape)
print("Shape de Y_filtered:", Y_filtered.shape)

codes_of_interest = [12, 13, 14, 23, 24,25]

# Fechas correspondientes a las 6 bandas
dates = ['2023-01-01', '2023-03-01', '2023-05-01', '2023-07-01', '2023-09-01', '2023-11-01']

# Colores más diferenciados
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']  # Paleta de colores

# Nombres de las clases
class_names = ['Chêne', 'Robinier', 'Peupleraie', 'Douglas', 'Pin laricio ou pin noir', 'Pin maritime']

# Preparar un gráfico
fig, ax = plt.subplots(figsize=(10, 6))

# Recorrer las clases y calcular la media y desviación estándar para cada una
for idx, code in enumerate(codes_of_interest):
    # Filtrar X y Y para la clase actual
    X_class = X_filtered[Y_filtered == code]
    
    # Verificar si hay muestras para la clase
    if X_class.shape[0] > 0:
        # Calcular la media y desviación estándar para cada banda (columna)
        means = X_class.mean(axis=0)
        stds = X_class.std(axis=0)
        
        # Graficar la media
        ax.plot(dates, means, color=colors[idx], label=class_names[idx])
        
        # Graficar el área sombreada para la desviación estándar
        ax.fill_between(dates, means + stds, means - stds, color=colors[idx], alpha=0.3)

# Configurar etiquetas y título
ax.set_xlabel('Fecha')
ax.set_ylabel('NDVI Promedio')
ax.set_title('NDVI Promedio por Clase y Fecha con Desviación Estándar')

# Limitar el rango de NDVI entre -1 y 1
ax.set_ylim(0, 1)

# Añadir una leyenda
ax.legend(title="Clases", loc='upper left')



# Enregistrement du graphique
plt.savefig(output_path, dpi=300)
print(f"Graphique enregistré dans : {output_path}")

# Mostrar gráfico
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()