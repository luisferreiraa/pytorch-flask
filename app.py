# Importação de bibliotecas
import io   # Utilizado para manipulação de fluxos de dados, neste caso, para carregar imagem como bytes
import json # Carrega o arquivo de mapeamento de classes do ImageNet, que contém os nomes das classes para a previsão
from torchvision import models  # Contém modelos pré-treinados como DenseNet, que é usado para fazer previsões em imagens
from flask import Flask, jsonify, render_template, request
import torchvision.transforms as transforms # Utilizado para aplicar transformações nas imagens, para serem usadas pelo modelo
from PIL import Image   # Biblioteca para carregar e manipular imagens

# Configuração da aplicação Flask
app = Flask(__name__)
imagenet_class_index = json.load(open('imagenet_class_index.json'))
model = models.densenet121(pretrained=True)
model.eval()


@app.route('/')
def index():
    return render_template('index.html')


def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx]


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        class_id, class_name = get_prediction(image_bytes=img_bytes)
        return jsonify({'class_id': class_id, 'class_name': class_name})
    
    
if __name__ == '__main__':
    app.run(debug=True)
