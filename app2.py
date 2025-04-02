import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained MNIST model
model = tf.keras.models.load_model("mnist_model.h5")

# Define function for prediction
def predict_digit(image):
    image = Image.fromarray(image).convert('L').resize((28, 28))
    img_array = np.array(image) / 255.0  # Normalize
    img_array = img_array.reshape(1, 28, 28)  # Reshape for model input

    prediction = model.predict(img_array)
    digit = np.argmax(prediction)  # Get the predicted number

    return int(digit)

# Create Gradio interface
interface = gr.Interface(fn=predict_digit, inputs="image", outputs="text")
interface.launch()