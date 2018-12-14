import numpy as np
import pandas as pd
from numpy import genfromtxt
import cv2
import csv, operator

import glob


fields = ['Speeed','Angle']


csvDataDirectory = './Data_CSV/Saved_data.csv'

"""
with open(csvDataDirectory) as csvfile:
    reader = csv.DictReader(csvfile,  delimiter = ",")
    for row in reader:
        angles =  (row['Angle'])
        print(angles)

   """

NImages = len(glob.glob("entrenamiento/*.jpg"))
print("NUmero de imagesnes")
print(NImages)


#Metemos las imagenes en un array(XTRain)
for i in range (NImages):
    print("entrenamiento/{}.jpg".format(i))
    imagen = cv2.imread("entrenamiento/{i}.jpg")
    XTrain.append(imagen) 
    