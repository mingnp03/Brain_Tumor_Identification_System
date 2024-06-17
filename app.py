import os
import numpy as np
import gradio as gr
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from tensorflow.keras.layers import DepthwiseConv2D as OriginalDepthwiseConv2D

import gdown
url = 'https://drive.google.com/file/d/1LlML6NGwEG8dtjgKoOquxJTCTQ_pYyX2/view?usp=sharing'
output = 'efficientnetv2-s-BTI44-96.91.h5'
gdown.download(url, output, quiet=False)

class CustomDepthwiseConv2D(OriginalDepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop('groups', None)  # Remove the 'groups' parameter if present
        super(CustomDepthwiseConv2D, self).__init__(*args, **kwargs)

custom_objects = {'DepthwiseConv2D': CustomDepthwiseConv2D}

model = tf.keras.models.load_model('efficientnetv2-s-BTI44-96.91.h5', custom_objects=custom_objects)
model.compile(optimizer='adam', loss='categorical_crossentropy')  # Adjust the optimizer and loss as per your requirements

class_labels = [
    'Astrocitoma T1', 'Astrocitoma T1C+', 'Astrocitoma T2', 'Carcinoma T1', 'Carcinoma T1C+', 'Carcinoma T2', 
    'Ependimoma T1', 'Ependimoma T1C+', 'Ependimoma T2', 'Ganglioglioma T1', 'Ganglioglioma T1C+', 'Ganglioglioma T2', 
    'Germinoma T1', 'Germinoma T1C+', 'Germinoma T2', 'Glioblastoma T1', 'Glioblastoma T1C+', 'Glioblastoma T2', 
    'Granuloma T1', 'Granuloma T1C+', 'Granuloma T2', 'Meduloblastoma T1', 'Meduloblastoma T1C+', 'Meduloblastoma T2', 
    'Meningioma T1', 'Meningioma T1C+', 'Meningioma T2', 'Neurocitoma T1', 'Neurocitoma T1C+', 'Neurocitoma T2', 
    'Oligodendroglioma T1', 'Oligodendroglioma T1C+', 'Oligodendroglioma T2', 'Papiloma T1', 'Papiloma T1C+', 'Papiloma T2', 
    'Schwannoma T1', 'Schwannoma T1C+', 'Schwannoma T2', 'Tuberculoma T1', 'Tuberculoma T1C+', 'Tuberculoma T2', 
    '_NORMAL T1', '_NORMAL T2'
]

def preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to match batch size
    return img_array

def predict_image_class(img_path):
    img_array = preprocess_image(img_path)
    preds = model.predict(img_array)
    class_idx = np.argmax(preds)
    confidence = preds[0][class_idx]
    return class_labels[class_idx], confidence

def predict_and_display(image):
    predicted_class, confidence = predict_image_class(image)
    confidence_percentage = confidence * 100  # Convert to percentage
    return predicted_class, f"{confidence_percentage:.2f}%"

# Create Gradio interface
interface = gr.Interface(
    fn=predict_and_display,
    inputs=gr.Image(type="filepath", label="Upload MRI Image"),
    outputs=[gr.Textbox(label="Predicted Class"), gr.Textbox(label="Confidence")],
    title="Brain Tumor Identification System",
    description="Upload an MRI image to get the predicted class of brain tumor along with the confidence level."
)

# Launch the Gradio app
interface.launch(share=True)  
