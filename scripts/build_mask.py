import geopandas as gpd
import numpy as np
import os

f_vege=gpd.read_file('/home/onyxia/work/data/project/FORMATION_VEGETALE.shp')
L_mask=['Lande','Formation herbacée','Forêt ouverte de conifères purs','Forêt ouverte de feuillus purs','Forêt ouverte sans couvert arboré',
        'Forêt ouverte à mélange de feuillus et conifères','Forêt fermée sans couvert arboré']
ones=np.ones((24041,1),dtype=int)
f_vege['mask']=ones

for i,j in zip(f_vege['TFV'],range(len(f_vege['mask']))):
    if i in L_mask:
        #f_vege['mask'][j]=0
        f_vege.loc[j,'mask']=0

f_vege['Valeur du pixel']=f_vege['mask']

for i in range(len(f_vege['mask'])):
    if f_vege['Valeur du pixel'][i]==1:
        f_vege['Classe']='Zone de forêt'
        # f_vege.loc[i,'mask']='Zone de forêt'
        # f_vege['mask'][i]='Zone de forêt'
    else:
        f_vege['Classe']='Zone hors forêt'
        # f_vege.loc[i,'mask']='Zone hors forêt'
        #f_vege['mask'][i]='Zone hors forêt'
# f_vege['Classe']=f_vege['mask']
Masque=f_vege[['ID','Classe','Valeur du pixel','geometry']]
Masque.head()

Masque.loc[:,'Valeur du pixel']=Masque['Valeur du pixel'].astype('uint8')