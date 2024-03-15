import numpy as np
from fastapi import FastAPI, Request
import base64

from detection.cnn_scratch.ml_logic.registry import load_model
from detection.cnn_scratch.ml_logic.preprocessor import preprocess_features

#from io import BytesIO
#from PIL import Image

app = FastAPI()
app.state.model = load_model()

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

    # Make a tensor with binary data
    tensor = preprocess_features(img_bin)

    predict = app.state.model.predict(tensor)
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
