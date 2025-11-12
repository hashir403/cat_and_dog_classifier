import os
import base64
from django.shortcuts import render
from django.core.files.storage import default_storage
from tensorflow.keras.models import load_model
import cv2
import numpy as np

MODEL_PATH = os.path.join("classifier", "trained_model.h5")
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
else:
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found.")

def label_to_string(label):
    """Convert numerical label to string."""
    return 'cat' if label == 0 else 'dog'

def upload(request):
    prediction_text = None
    uploaded_image = None

    if request.method == "POST" and request.FILES.get('image'):
        # Handle file upload
        image_file = request.FILES['image']
        file_path = default_storage.save(image_file.name, image_file)

        # Full path to the uploaded image
        full_path = os.path.join(default_storage.location, file_path)

        # Preprocess the uploaded image
        img = cv2.imread(full_path)
        img_resized = cv2.resize(img, (256, 256))
        img_normalized = img_resized / 255.0
        img_reshaped = np.reshape(img_normalized, (1, 256, 256, 3))

        # Make prediction
        prediction = model.predict(img_reshaped)
        predicted_label = 1 if prediction > 0.5 else 0
        prediction_text = label_to_string(predicted_label)

        # Convert image to Base64 to send to template
        with open(full_path, "rb") as img_file:
            uploaded_image = f"data:image/jpeg;base64,{base64.b64encode(img_file.read()).decode()}"

        # Clean up the uploaded file
        default_storage.delete(file_path)

    return render(request, 'upload.html', {
        'prediction': prediction_text,
        'uploaded_image': uploaded_image
    })
