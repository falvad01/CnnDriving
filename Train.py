import sys #movernos en nuestro So
import os #movernos en nuestro So
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator #Ayuda a preprocesar las imagenes
from tensorflow.python.keras import optimizers #optimizar para entrenar el algotirmo
from tensorflow.python.keras.models import Sequential #Nos permite hacer redes neuronales secuenciales, para que las capas esten en orden
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation #
from tensorflow.python.keras.layers import  Convolution2D, MaxPooling2D #Capas de la red neuronal
from tensorflow.python.keras import backend as K #Si hay una sesion de keras la mata


K.clear_session() #Matamos la sescion de keras anterior



