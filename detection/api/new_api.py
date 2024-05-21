import numpy as np
import fastapi
from fastapi import FastAPI, Request
import base64

from detection.cnn_scratch.ml_logic.registry import load_model as cnn_scratch_model
from detection.cnn_scratch.ml_logic.preprocessor import preprocess_features
from detection.cnn_pre.ml_logic.registry import load_model as cnn_pre_model
from detection.svm.ml_logic.registry import load_model as svm_model

#from io import BytesIO
#from PIL import Image

app = FastAPI()
app.state.model_scratch = cnn_scratch_model()
app.state.model_pre = cnn_pre_model()
app.state.model_svm = svm_model()


# Define a root `/` endpoint
@app.get('/')
def index():
    return {'ok': "Essa é minha primeira API. Thainá vai me ajudar"}

@app.post('/predict')
async def predict(request: Request):
    content = await request.json()
    try:
        img_bin = base64.b64decode(content['image'])
    except:
        return {'status': 'fail'}

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


    # Make a tensor with binary data
    tensor = preprocess_features(img_bin)

    predict_scratch = app.state.mmodel_scratch.predict(tensor)
    predict_pre = app.state.mmodel_pre.predict(tensor)
    predict_svm = app.state.mmodel_svm.predict(tensor)
    proba = predict[0]
    print(f"proba: {proba}")
    print(type(proba))
    index = np.argmax(proba)

    if index == 0:
        target='COVID-19'
    elif index == 1:
        target='Normal'
    elif index == 2:
        target='Opacity'
    elif index == 3:
        target='Pneumonia'
    else:
        target='ERROR'

    return {'status': 'ok', 'probability': proba.tolist(), 'class': target}

@app.get('/predict')
def predict():
    return {'status': 'ok'}
