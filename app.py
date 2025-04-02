import tensorflow as tf
from tensorflow import keras
from flask import Flask, request, jsonify
import numpy as np
from PIL import Image

# Load the trained model
model = keras.models.load_model('mnist_model.h5')  

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    image = Image.open(file).convert('L').resize((28, 28))  # Convert to grayscale and resize
    img_array = np.array(image) / 255.0  # Normalize
    img_array = img_array.reshape(1, 28, 28)  # Reshape for model compatibility
    
    prediction = model.predict(img_array)
    digit = np.argmax(prediction)  # Get the predicted number

    return jsonify({'predicted_digit': int(digit)})

if __name__ == '__main__':
    app.run(debug=True)