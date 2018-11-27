import numpy as np
import simplejson
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

#############FUNCIONES#####################
def predict(file):
    x = load_img(file,target_size =(lon,alt)) #guardamos la imagen en x
    x = img_to_array(x)#trnsformamos la imagen en array
    x = np.expand_dims(x,axis =  0)

    array= cnn.predict(x) #Creamos el array de las soluciones
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


cnn = load_model('./model2/model.h5')
cnn.load_weights('./model2/model_weights.h5')


"""
# load json and create model
json_file = open('Data/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("Data/model.h5")
print("Loaded model from disk")
"""
"""

"""
predict('1.jpeg')
