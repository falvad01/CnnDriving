import numpy as np
import pandas as pd
from numpy import genfromtxt
import cv2
import csv, operator

import glob


fields = ['Date','Speeed','Angle']


csvDataDirectory = './Data_CSV/Saved_data.csv'


X_data = []
files = glob.glob ("./Entrenamiento/*.png")
for myFile in files:
    image = cv2.imread (myFile)
    X_data.append (image)

print('X_data shape:', np.array(X_data).shape)





with open(csvDataDirectory) as csvfile:
    reader = csv.DictReader(csvfile,  delimiter = ";")
    for row in reader:
        angles =  (row['Angle'])

    

