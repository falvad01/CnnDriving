#enconding: utf-8
##si da error instalar pillow con CONDA NO CON PIP

##DUDAS##
##OPTIMIZACION EN UNITY
##LAS IMAGENES DE ENTRENAMIENTO Y VALIDACION, DIFERENCIAS Y SI PUEDEN SER IGUALES
##SI SE PODRIA ENTRENAR LA RED CNN CON IMAGENES REALES O TIENEN QUE SER DEL JUEGO
##PROBLEMA AL CARGAR EL MODELO
##A LA HORA DE CLASICAR LAS IMAGENES SERIA EN CURVAS DERECHA, CURVAS IZQUIERDA Y RECTAS O TENDRIA QUE PENSAR EN OTRO TIPO DE CLASIFICACION
##NUMERO DE FOTOS OPTIMO PARA ENTRENAR LA RED
##ERROES EN EL FROM, QUE SE SOLUCIONAN SOLOS
 

import sys #movernos en nuestro So
import os #movernos en nuestro So


from tensorflow.python.keras.preprocessing.image import ImageDataGenerator #Ayuda a preprocesar las imagenes
from tensorflow.python.keras import optimizers #optimizar para entrenar el algotirmo
from tensorflow.python.keras.models import Sequential #Nos permite hacer redes neuronales secuenciales, para que las capas esten en orden
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation #
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D #Capas de la red neuronal
from tensorflow.python.keras import backend as K #Si hay una sesion de keras la mata


K.clear_session() #Matamos la sescion de keras anterior

trainingData = './Entrenamiento' #Guardamos el directorio de las imagenes
validationData = './Validacion'
 
gen = 20 #generaciones
alt = 150 # tamanio imagenes 
lon = 150
batchSize = 32 
steps = 1000 # numero de veces que se procesa la imagen por gen
validationSteps = 200
conv1Filters = 32 #profundidad de la imagen en cada convolucion
conv2Filters = 64
filter1Size =  (3,3) #tamanio de los filtros 
filter2Size = (2,2)
poolSize = (2,2)
clas = 3 #curvasD, curvasI, rectas
learnigRate = 0.0005 #ajustes de la red neuronal para ajuste optimo


#PREPROCESAMIENTO DE LAS IMAGENES

dataGenTraining = ImageDataGenerator(
    rescale = 1./255, #los valores de los pixeles se reducen a 0 1
    shear_range = 0.3, #enseniamos al algortimo que las imagenes pueden estar inclinadas
    zoom_range = 0.3, #enseniamos al algoritmo que algunas imaenes estan mas cerca que otras
    horizontal_flip = True #invertimos las imagenes para enseniar al aglorimo direccionalidad
)

dataGenValidation = ImageDataGenerator(
    rescale = 1./255 #para validar las imagenes tal cual son, sin girar ni zoom ni nada
)

trainingImage = dataGenTraining.flow_from_directory(#transformamos las imagenes
    trainingData,
    target_size = (alt,lon),
    batch_size = batchSize ,
    class_mode = 'categorical' #clasifacion categorica de cada imgen, recta curbad y cubai

)

validationImage = dataGenValidation.flow_from_directory(
    validationData,
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
        input_shape = (alt,lon,3),  #las imagenes de entrada son de este tamanio
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
        padding = 'same' 
    )
)

cnn.add( #cuarta capa
    MaxPooling2D(
        pool_size = poolSize
    )
)

cnn.add(
    Flatten()#la imagen que es muy profunda despues de pasar por los filtros la aplanamos para concentrar toda la informacion
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
        activation = 'softmax' #nos da la provavilidad de que sea una cosa u otra
    )
)

cnn.compile(
    loss = 'categorical_crossentropy', #indica si va bien o mal
    optimizer = optimizers.Adam(
        lr = learnigRate #optimizador 
    ),
    metrics = ['accuracy']
)

cnn.fit_generator( 
    trainingImage,
    steps_per_epoch = steps,
    epochs = gen,
    initial_epoch = 0,
    validation_data = validationImage,
    validation_steps = validationSteps
) #con los que vamos a entrenar la imagen

#GUARDAMOS EL MODELO EN UN ARCHIVO

dir = './model/' #directorio del modelo de salida
os.mkdir(dir)

cnn.save('./model/model.h5') #guardamos la estructura del modelo
cnn.save_weights('./model/model_wheights.h5') #guardamos los pesos de cada capa