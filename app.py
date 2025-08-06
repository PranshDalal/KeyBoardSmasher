from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import os

app = Flask(__name__)

model = None
tokenizer = None
label_enc = None
MAX_LEN = None

def load_resources():
    global model, tokenizer, label_enc, MAX_LEN
    if model is None:
        model = load_model("model/keyboard_smash_model.h5")
        with open("model/tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        with open("model/label_encoder.pkl", "rb") as f:
            label_enc = pickle.load(f)
        MAX_LEN = model.input_shape[1]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def predict():
    load_resources()
    data = request.json
    if not data or "smash" not in data:
        return jsonify({"error": "Missing 'smash' in request"}), 400

    smash = data["smash"]
    seq = tokenizer.texts_to_sequences([smash])
    padded_seq = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
    pred = model.predict(padded_seq, verbose=0)
    mood = label_enc.inverse_transform([np.argmax(pred)])[0]
    return jsonify({"mood": mood})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=os.environ.get("PORT", 5000))

