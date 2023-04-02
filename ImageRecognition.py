import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

# Load the pre-trained MobileNetV2 model
model = MobileNetV2(weights='imagenet')

def recognize_image(image_path):
    # Load and preprocess the image
    img = Image.open(image_path).resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_array)

    # Make predictions and decode them
    predictions = model.predict(img_preprocessed)
    decoded_predictions = decode_predictions(predictions, top=3)[0]

    return decoded_predictions

# Test the image recognition app
image_path = "4.jpg"
predictions = recognize_image(image_path)

print("Predictions:")
for index, (imagenet_id, label, confidence) in enumerate(predictions):
    print(f"{index+1}. {label}: {confidence*100:.2f}%")