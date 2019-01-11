import sys 
import os 
import numpy
import csv, operator
import cv2
import glob

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator #Ayuda a preprocesar las imagenes
from tensorflow.python.keras import optimizers #optimizar para entrenar el algotirmo
from tensorflow.python.keras.models import Sequential #Nos permite hacer redes neuronales secuenciales, para que las capas esten en orden
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation #
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D #Capas de la red neuronal
from tensorflow.python.keras import backend as K #Si hay una sesion de keras la mata


K.clear_session() #Matamos la sescion de keras anterior
 
epochs = 5 
alt = 150 # tamanio imagenes
lon = 150
batchSize = 32 
conv1Filters = 32 #profundidad de la imaiter en cada convolucion
conv2Filters = 64
filter1Size =  (3,3) #tamanio de los filtros 
filter2Size = (2,2)
poolSize = (2,2)
clas = 1
learnigRate = 0.0005 #ajustes de la red neuronal para ajuste optimo
XTrain = [] #Vector de imagenes
NImages = 0


NImages = len(glob.glob("entrenamiento/*.jpg"))

#Metemos las imagenes en un array(XTRain)

for i in range (NImages):
    imagen = cv2.imread("entrenamiento/{}.jpg".format(i))
    XTrain.append(imagen)
   


#Cargamos los archivos de los angulos de giro y los metemos en una lista(YTrain)
csvDataDirectory = './Data_CSV/Saved_data.csv'
with open(csvDataDirectory) as csvfile:
    reader = csv.DictReader(csvfile,  delimiter = ",")
    for row in reader:
        Ytrain =  (row['Angle'])
       

#CREAMOS LA RED CNN

cnn = Sequential() #La red son varias capas apiladas

cnn.add(#creamos la primera capa
    Convolution2D(  #sera una convolucion
        conv1Filters,  #32 filtros
        filter1Size, # de tamanio 3,3
        padding = 'same',
       # input_shape = (alt,lon,3),  #las imagenes de entrada son de este tamanio
        activation = 'relu' 
    )
)

cnn.add( #creamos una segunda capa
    MaxPooling2D( # de tipo maxPooling
        pool_size = poolSize  
    )
)

cnn.add( #tercera capa
    Convolution2D(
        conv2Filters,
        filter2Size,
        padding = 'same'   #puede ser valid o same, con el valid cogemos el limite del cuadrado, mediate el same asignamos nuevos piexeles a la  image, que se les pone valor 0, 
                            #si se le pone el sttep en 1, eñ tamaño de la imaiter es lemismo que la oriiter, pero con valid seria un pixel mas pequeño por cada lado
    )
)

cnn.add( #cuarta capa
    MaxPooling2D(
        pool_size = poolSize
    )
)

cnn.add(
    Flatten()#la imaiter que es muy profunda despues de pasar por los filtros la aplanamos para concentrar toda la informacion
)

cnn.add( #capa normal(5)
    Dense(
        256, #numero de nuronas
        activation = 'relu'
    )
)

cnn.add(
    Dropout(0,5) #durante el entrenamiento apagamos la mitad de las neuronas,esto nos permite evitar el sobreajuste
)

#TODO ESTA CAPA PUEDE SER ELIMINADA
cnn.add(#ultima capa(6)
    Dense(
        clas,
        activation = 'softmax' #funcion de activacion  #delimitar
    )
)

cnn.compile(
    optimizer = optimizers.Adam(
         lr = learnigRate #optimizador 
    ),
    loss = 'categorical_crossentropy', #indica si va bien o mal
    metrics = ['accuracy']
)

#Hcaer iterrador para meter imaiterers poco a poco

cnn.fit(
    XTrain,
    Ytrain, 
    batch_size=256, 
    epochs= epochs, #Epocas de entrenamiento
    verbose=1,  
    validation_split=0.4 #cojemos el 40% de las imagnes para validacion
)

"""
tupla = (XTrain,Ytrain)
cnn.fit_generator(
    generator = tupla,
    steps_per_epoch=500,
    epochs = epochs,
    verbose = 2,
    validation_data= tupla
)
"""

#GUARDAMOS EL MODELO EN UN ARCHIVO
 
dir = './model/' #directorio del modelo de salida
os.mkdir(dir)

cnn.save('./model/model.h5') #guardamos la estructura del modelo
#cnn.save_weights('./model/model_wheights.h5') #guardamos los pesos de cada capa