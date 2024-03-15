import base64
import requests
from io import StringIO
import streamlit as st


# Título da aplicação
st.title('Envio de imagem como payload para API')

# Criação do uploader de arquivos
uploaded_file = st.file_uploader("Escolha uma imagem...", type=['.jpg', '.png'])


if uploaded_file is not None:
    st.image(uploaded_file)

    encoded_image = base64.b64encode(uploaded_file.getvalue()).decode('utf-8')
    #print(encoded_image)

    payload = {
         "image": encoded_image
         }

    api_url = "http://127.0.0.1:8000/predict/"

    response = requests.post(api_url, json=payload)

    if response.status_code == 200:
        if response.json()['status'] == 'fail':
            st.error('Falha ao decodificar a imagem')
        else:
            st.success('Imagem enviada com sucesso!')
            st.write(f"Class: {response.json()['class']}")
            st.write(f"Probability: {response.json()['probability']}")

    else:
        st.error('Falha ao enviar imagem.')
