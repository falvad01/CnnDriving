#enconding: utf-8
##si da error instalar pillow con CONDA NO CON PIP

##DUDAS##
##OPTIMIZACION EN UNITY
##LAS IMAiterES DE ENTRENAMIENTO Y VALIDACION, DIFERENCIAS Y SI PUEDEN SER IGUALES
##PROBLEMA AL CARGAR EL MODELO
##EN LA RED ESTOY USANDO UNA FUNCION RELU Y UNA SOFTMAX, SERI AMEJOR CAMBAIRLAS POR UNA SIGMOIDAL???
##A LA HORA DE CLASICAR LAS IMAiterES SERIA EN CURVAS DERECHA, CURVAS IZQUIERDA Y RECTAS O TENDRIA QUE PENSAR EN OTRO TIPO DE CLASIFICACION
##NUMERO DE FOTOS OPTIMO PARA ENTRENAR LA RED
##ERROES EN EL FROM, QUE SE SOLUCIONAN SOLOS
 

import sys #movernos en nuestro So
import os #movernos en nuestro So 
import pandas as pd # data processing, CSV file


from tensorflow.python.keras.preprocessing.image import ImageDataitererator #Ayuda a preprocesar las imaiteres
from tensorflow.python.keras import optimizers #optimizar para entrenar el algotirmo
from tensorflow.python.keras.models import Sequential #Nos permite hacer redes neuronales secuenciales, para que las capas esten en orden
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation #
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D #Capas de la red neuronal
from tensorflow.python.keras import backend as K #Si hay una sesion de keras la mata


K.clear_session() #Matamos la sescion de keras anterior

imaiterDirectory = './Entrenamiento' #Guardamos el directorio de las imaiteres
validationDirectory = './Validacion'
csvDataDirectory = './Csv'
 
iter = 5 #itereraciones #cambiar a 5-1 0
alt = 150 # tamanio imaiteres 
lon = 150
batchSize = 32 
steps = 1000 # numero de veces que se procesa la imaiter por iter
validationSteps = 200
conv1Filters = 32 #profundidad de la imaiter en cada convolucion
conv2Filters = 64
filter1Size =  (3,3) #tamanio de los filtros 
filter2Size = (2,2)
poolSize = (2,2)
clas = 3 #curvasD, curvasI, rectas
learnigRate = 0.0005 #ajustes de la red neuronal para ajuste optimo

#meter datos csv y relacionar el nombre de la imaiter con cada Y(anguos de giro)
 #XTRain immaiteres //YTrain datos csv

#PREPROCESAMIENTO DE LAS IMAiterES

dataiterTraining = ImageDataitererator(
    rescale = 1./255, #los valores de los pixeles se reducen a 0 1
)

dataiterTraining = pd.read_csv(csvDataDirectory).values #cargamos el archivo csv

dataiterValidation = ImageDataitererator(
    rescale = 1./255 #para validar las imaiteres tal cual son, sin girar ni zoom ni nada
)

trainingImage = dataiterTraining.flow_from_directory(#transformamos las imaiteres
    imaiterDirectory,
    target_size = (alt,lon),
    batch_size = batchSize ,
    class_mode = 'categorical' #clasifacion categorica de cada imiter, recta curbad y cubai

)

validationImage = dataiterValidation.flow_from_directory(
    validationDirectory,
    target_size = (alt,lon),
    batch_size = batchSize ,
    class_mode = 'categorical'
)

#CREAMOS LA RED CNN

cnn = Sequential() #La red son varias capas apiladas

cnn.add(#creamos la primera capa
    Convolution2D(  #sera una convolucion
        conv1Filters,  #32 filtros
        filter1Size, # de tamanio 3,3
        padding = 'same',
        input_shape = (alt,lon,3),  #las imaiteres de entrada son de este tamanio
        activation = 'relu' 
    )
)

cnn.add( #creamos una segunda capa
    MaxPooling2D( # de tipo maxPooling
        pool_size = poolSize  #y este es su tamanio 
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

cnn.add(#ultima capa(6)
    Dense(
        clas,
        activation = 'softmax' #funcion de activacion  #delimitar
    )
)

cnn.compile(
    loss = 'categorical_crossentropy', #indica si va bien o mal
    optimizer = optimizers.Adam(
        lr = learnigRate #optimizador 
    ),
    metrics = ['accuracy']
)

#Hcaer iterrador para meter imaiterers poco a poco
cnn.fit_itererator( 
    trainingImage,
    steps_per_epoch = steps,
    epochs = iter,
    initial_epoch = 0,
    validation_data = validationImage,
    validation_steps = validationSteps
) #con los que vamos a entrenar la imaiter

#GUARDAMOS EL MODELO EN UN ARCHIVO
 

dir = './model/' #directorio del modelo de salida
os.mkdir(dir)

cnn.save('./model/model.h5') #guardamos la estructura del modelo
#cnn.save_weights('./model/model_wheights.h5') #guardamos los pesos de cada capa


"""
dir = './Data/' #directorio del modelo de salida
os.mkdir(dir)
## serialize model to JSON
model_json = cnn.to_json()
with open("Data/model.json", "w") as json_file:
    json_file.write(simplejson.dumps(simplejson.loads(model_json), indent=4))

# serialize weights to HDF5
cnn.save_weights("Data/model.h5")
print("Saved model to disk")
"""
