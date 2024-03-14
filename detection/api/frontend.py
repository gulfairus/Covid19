import base64
import requests
import streamlit as st

# Título da aplicação
st.title('Envio de imagem como payload para API')

# Criação do uploader de arquivos
uploaded_file = st.file_uploader("Escolha uma imagem...", type=['jpg', 'png'])

if uploaded_file is not None:
    # Ler a imagem carregada e convertê-la em base64sx
    encoded_image = base64.b64encode(uploaded_file.getvalue()).decode('utf-8')
    
    # Construir o payload da requisição, incluindo a imagem codificada
    payload = {
        "image": encoded_image,
        # Você pode adicionar mais campos ao payload se necessário
    }
    
    # URL da API para a qual você deseja enviar a imagem
    api_url = "https://suaapi.com/seuendpoit"

    # Enviar a imagem para a API
    response = requests.post(api_url, json=payload)
    
    # Checar se a requisição foi bem-sucedida
    if response.status_code == 200:
        st.success('Imagem enviada com sucesso!')
        # Aqui você pode processar e mostrar a resposta da API
    else:
        st.error('Falha ao enviar imagem.')
