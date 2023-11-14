from flask import Flask, render_template, request
from PIL import Image
import numpy as np
from tensorflow.keras.applications.vgg19 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import VGG19
import os
from io import BytesIO

app = Flask(__name__)
model = VGG19(weights='imagenet')
#model = load_model('vgg19DBPmodel.h5')

# Create the 'static' directory if it doesn't exist
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
if not os.path.exists(static_dir):
    os.makedirs(static_dir)
    
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/feedback')
def feedback():
    return render_template('feedback.html')

@app.route('/team')
def team():
    return render_template('team.html')

@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', prediction="No file selected!")

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', prediction="No file selected!")

    img = Image.open(file)
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=1)[0][0]

    breed_prediction = decoded_predictions[1]

    temp_img_path = os.path.join('static', 'temp_img.jpg')
    img.save(temp_img_path)
    return render_template('index.html', prediction=breed_prediction, image_path=temp_img_path)

if __name__ == '__main__':
    app.run(debug=True)