import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
# import keras
from keras.models import load_model
import tensorflow as tf
# from keras import preprocessing
import tensorflow_hub as hub
from keras.applications.mobilenet import decode_predictions
st.title("Traffic Signals")

def main():
    file_uploaded = st.file_uploader("Choose the file" ,type = ['Jpg', "png","jpeg"])
    if file_uploaded is not None:
        image = Image.open(file_uploaded)
        figure = plt.figure()
        plt.imshow(image)
        plt.axis("off")
        result = predict_class(image)
        st.write(result)
        st.pyplot(figure)

def predict_class(image):
    model = tf.keras.models.load_model(r'C:\Users\HP\Desktop\streamlit\Traffic_Signal\traffic_aug_traingen_98.h5')
    # shape = ((299,299,3))
    # model = tf.keras.Sequential(hub[hub.KerasLayer(model,input_shape = shape)])
    # test_image=preprocessing.image.img_to_array(image)
    test_image = np.array(image)
    test_image = cv2.resize(test_image,(224,224))
    test_image = test_image/255.0
    test_image = np.expand_dims(test_image,axis = 0)
    class_names = [' Speed limit (5km/h)','Speed limit (15km/h)','Dont Go straigh','Dont Go Left','Dont Go Left or Right',' Dont Go Right','Dont overtake from Left','No Uturn','No Car','No horn','Speed limit (40km/h)',' Speed limit (50km/h)','Speed limit (30km/h)','Go straight or right','Go straight','Go Left','Go Left or right','Go Right',' keep Left','keep Right','Roundabout mandatory','watch out for cars','Horn','Speed limit (40km/h)','Bicycles crossing','Uturn','Road Divider','Traffic signals','Danger Ahead','Zebra Crossing','Bicycles crossing','Children crossing','Dangerous curve to the left','Dangerous curve to the right','Speed limit (50km/h)','Unknown1','Unknown2','Unknown3',' Go right or straight',' Go left or straight',' Unknown4','ZigZag Curve','Train Crossing','Under Construction','Unknown5','Speed limit (60km/h)','Fences','Heavy Vehicle Accidents','Unknown6','Give Way','No stopping','No entry','Unknown7','Unknown8','Speed limit (70km/h)','speed limit (80km/h)','Dont Go straight or left','Dont Go straight or Right']


    predictions = model.predict(test_image)
    scores = tf.nn.softmax(predictions[0])
    scores = scores.numpy()
    image_class = class_names[np.argmax(scores)]
    # predicted = class_names[np.argmax(model.predict(test_image)[0])]
    result = "The image uploaded is:- {}".format(image_class)
    # st.write(scores*100)
    return result
if __name__ == "__main__": 
    main()
 