import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

#############FUNCIONES#####################
def predict(file):
    x = load_img(file,target_size =(lon,alt)) #guardamos la imagen en x
    x = img_to_array(x)#trnsformamos la imagen en array
    x = np.expand_dims(x,axis =  0)

    array = loaded_model.predict(x) #Creamos el array de las soluciones
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



jsonFile = open('Data/model.json', 'r')
loadedModelJson = jsonFile.read()
jsonFile.close()
loadedModel = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("Data/model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data 
# Define X_test & Y_test data first
loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#score = loaded_model.evaluate(X_test, Y_test, verbose=0)
print ("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

predict('1.jpeg')
