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
from keras.applications.inception_resnet_v2 import decode_predictions
st.title("Intel Image Classifier")

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
    model = tf.keras.models.load_model(r'C:\Users\HP\Desktop\streamlit\Intel_Image_Classifier\intel_techsunny.hdf5')
    # shape = ((299,299,3))
    # model = tf.keras.Sequential(hub[hub.KerasLayer(model,input_shape = shape)])
    # test_image=preprocessing.image.img_to_array(image)
    test_image = np.array(image)
    test_image = cv2.resize(test_image,(299,299))
    test_image = test_image/255.0
    test_image = np.expand_dims(test_image,axis = 0)
    class_names = ['buildings','forest','glacier','mountain','sea','street']

    predictions = model.predict(test_image)
    scores = tf.nn.softmax(predictions[0])
    scores = scores.numpy()
    image_class = class_names[np.argmax(scores)]
    # predicted = class_names[np.argmax(model.predict(test_image)[0])]
    result = "The image uploaded is:{}".format(image_class)
    # st.write(scores*100)
    return result
if __name__ == "__main__": 
    main()
 