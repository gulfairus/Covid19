import streamlit as st
from PIL import Image
import time
import h5py
import boto3
import numpy as np
import streamlit as st
from PIL import Image

import tensorflow
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import load_model
import tensorflow as tf

def aws_model(file_model):
    s3 = boto3.resource('s3')
    obj = s3.Object(bucket_name='covidprojectmodel', key=file_model)
    with open('object1', 'wb') as data:
        obj.download_fileobj(data)
    model = tf.keras.models.load_model('object1')
    return model

file_model = 'cnn_model_fine.h5'
file_model = 'covidprojectmodel.h5'

@st.cache(allow_output_mutation=True)
def download_model():
    #st.warning("""Loading model""")
    start_time = time.time()
    model = aws_model(file_model)
    end_time = time.time()
    load_time = (end_time - start_time) / 60
    return model, load_time
# model, load_time = download_model()
# st.write("loading time, min: ", load_time)
# shape = (150,150

def predict_covid(filename, model):
    #model = create_model()
    dict = {}
    try:
        img = Image.open(filename)
    except:
        print("ERROR: Couldn't open image! Make sure the image path and extension is correct")
    #img = image.load_img(filename, target_size=(150,150))
    imges = img.resize((150,150))
    images = np.array(imges)
    # for images that has 4 channels, we convert them into 3 channels
    if len(images.shape)>2:
        if images.shape[2] == 4:
            images = images[..., :3]
    else:
        images = images.reshape((150, 150, 1))
    images = np.expand_dims(images, axis=0)
    #images = images/255
    images = preprocess_input(images)
    prediction = model(images).numpy()
    prediction1 = prediction[0]
    max_ind = np.argmax(prediction1)
    max_val = prediction1[max_ind]
    if max_ind==0:
        caption = f"COVID19 predicted with {max_val:.2f} probability"
    elif max_ind==1:
        caption = f"NORMAL predicted with {max_val:.2f} probability"
    elif max_ind==2:
        caption = f"OPACITY predicted with {max_val:.2f} probability"
    elif max_ind==3:
        caption = f"PNEUMONIA predicted with {max_val:.2f} probability"
    #prediction2 = prediction1[0]

    dict["COVID19"] = prediction1[0]
    dict["NORMAL"] = prediction1[1]
    dict["OPACITY"] = prediction1[2]
    dict["PNEUMONIA"] = prediction1[3]
    return dict, caption

#backgroundColor: str = "#F63366"
st.warning("""For educational purposes only""")
st.title("""
 Classification of detection COVID-19, OPACITY, PNEUMONIA, NORMAL from Chest X-ray images
""")
#st.header("from Chest X-ray images")
st.warning("""This is a COVID-19 classification web app to classify patients as either COVID 
infected or healthy or have opacity/pneumonia using their chest x-ray scans""")
file = st.file_uploader("Please upload an image file", type=["jpg", "png", "jpeg"])

if file is not None:
    caption1, caption2 = predict_covid(file, model)


#st.image(pics[pic], use_column_width=True, caption=0)

    st.image(file, use_column_width=True, caption=st.write(caption2, caption1))
