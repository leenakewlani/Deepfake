from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__, static_folder=".")
CORS(app)

MODEL_PATH = "resnet_deepfake_model.h5"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model missing: resnet_deepfake_model.h5")

model = tf.keras.models.load_model(MODEL_PATH)

@app.route("/")
def home():
    return send_from_directory(".", "index.html")

def preprocess_image(file_bytes):
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    img = preprocess_image(file.read())

    pred = float(model.predict(img, verbose=0)[0][0])

    label = "FAKE" if pred >= 0.5 else "REAL"
    confidence = pred if pred >= 0.5 else 1 - pred

    return jsonify({
        "prediction": label,
        "confidence": round(confidence * 100, 2)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
