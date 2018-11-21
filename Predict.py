import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

#############FUNCIONES#####################
def predict(file):
    x = load_img(file,target_size =(lon,alt)) #guardamos la imagen en x
    x = img_to_array(x)#trnsformamos la imagen en array
    x = np.expand_dims(x,axis =  0)

    array = cnn.predict(x) #Creamos el array de las soluciones
    result = array[0]
    answer = np.argmax(result)

    if answer == 0:
        print("Derecha")
    elif answer == 1:
        print("Izquierda")
    elif answer == 2:
        print("Recta")

    return answer

###############################################

lon = 150
alt = 150 



cnn = load_model('./model/model.h5') #cargamos el modelo de la red
cnn.load_weights('./model/model_weights.h5') #cargamos los pesos de la red

predict('1.jpeg')
