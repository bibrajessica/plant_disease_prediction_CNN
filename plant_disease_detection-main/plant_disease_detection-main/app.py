import os
import numpy as np
from keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)

# Load your model
model = load_model('model.h5')
print('Model loaded. Check http://127.0.0.1:5000/')

# Define the uploads directory
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create uploads directory if it doesn't exist
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Label mapping
labels = {0: 'Healthy', 1: 'Powdery', 2: 'Rust'}

# Prediction Function
def getResult(image_path):
    img = load_img(image_path, target_size=(225, 225))
    x = img_to_array(img)
    x = x.astype('float32') / 255.0
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)[0]
    return predictions

# Routes
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file part", 400
    
    f = request.files['file']
    
    if f.filename == '':
        return "No selected file", 400

    # Save the file in the uploads directory
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename))
    f.save(file_path)

    predictions = getResult(file_path)
    predicted_label = labels[np.argmax(predictions)]
    return predicted_label  # Return predicted label as response

if __name__ == '__main__':
    app.run(debug=True)
