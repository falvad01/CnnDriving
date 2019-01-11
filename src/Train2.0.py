import sys 
import os 
import numpy as np
import csv, operator
import cv2
import glob
import matplotlib.pyplot as plt 
from PIL import Image
from resizeimage import resizeimage

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator #Ayuda a preprocesar las imagenes
from tensorflow.python.keras import optimizers #optimizar para entrenar el algotirmo
from tensorflow.python.keras.models import Sequential #Nos permite hacer redes neuronales secuenciales, para que las capas esten en orden
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation #
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D #Capas de la red neuronal
from tensorflow.python.keras import backend as K #Si hay una sesion de keras la mata
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

K.clear_session()

#xTrain = []
#yTrain = []
img = []

numImages = len(glob.glob("entrenamiento/*.jpg"))

for i in range (numImages):
    image = cv2.imread("entrenamiento/{}.jpg".format(i))
    #realizar crop de las imagenes
    img.append(image)
   
    
   

xTrain=np.array(img,'float32')

#Visualizacion de una imagen para su comprobacion   
#print(xTrain)
#imgplot = plt.imshow(img[2])  
#plt.show()  


with open('./Data_CSV/Saved_data.csv') as csvfile:
    reader = csv.DictReader(csvfile, delimiter = ",")
    for row in reader:
        yTrain = np.array(row['Angle']) #relleno de datos de giro
       # print(yTrain)



model = Sequential()


model.add(Convolution2D(32, (3,3),padding = 'same',input_shape = (697, 1364, 3),activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Convolution2D(64, (2,2),padding = 'same',activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Flatten())
model.add(Dense(256,activation = 'relu'))
model.add(Dropout(0,5))


model.summary()


model.compile(optimizer = optimizers.adam(lr = 0.0005),loss = 'categorical_crossentropy',metrics = ['accuracy'])

model.fit(xTrain,validation_split=0.2,shuffle=True,epochs=10 ) #Fallo en validation_split,
#por lo que intuimos un fallo en la introduccion de datos, pero no encontramo en el fallo, las imagenes estan
#introducidas como un numpy array y los datos como un arry de floats



#scores = model.evaluate(xTrain, yTrain)
#print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100)

dir = './model/' #directorio del modelo de salida
os.mkdir(dir)

model.save('./model/model.h5') #guardamos la estructura del modelo



