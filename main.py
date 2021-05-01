from flask import Flask, render_template, request, jsonify
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

import numpy as np
import pickle
import re
import tensorflow as tf

from tensorflow.python.lib.io import file_io

app = Flask(__name__)

LINKS_re = re.compile(r"https?://.+?(\s|$)")
NONALPHANUMERIC_re = re.compile(r"[^\w ]")
MODEL_GS = "gs://detoxify/tf2_cpu_friendly_model.h5"
TOK_GS = "gs://detoxify/tokenizer.pickle"
TOK_PATH = "tokenizer.pickle"
MODEL_PATH = "cpu_friendly_model.h5"
MAXLEN = 200  # more than 99th percentile

LABELS = ["Toxicity", "Severe Toxicity", "Identity Attack", "Insult", "Threat"]

def load_from_gs(gs_path, local_path): # TODO this uses a tf io method, does it work for the pickle?
    """
    Load a file from google storage into a local path.
    """

    gs_file = file_io.FileIO(gs_path, mode='rb')
    local_file = open(local_path, 'wb')

    local_file.write(gs_file.read())
    local_file.close()
    gs_file.close()

load_from_gs(MODEL_GS, MODEL_PATH)
load_from_gs(TOK_GS, TOK_PATH)

print("Loading Tokenizer..")
with open(TOK_PATH, "rb") as handle:
    tokenizer = pickle.load(handle)

print("Loading Model..")
model = load_model(MODEL_PATH)


print("Loaded model and tokenizer.")


def clean_text(comment):
    cleaned = re.sub(LINKS_re, " ", comment)
    cleaned = re.sub(NONALPHANUMERIC_re, " ", cleaned)
    cleaned = re.sub(r" +", " ", cleaned)  # collapse consequtive spaces
    return cleaned


def preprocess(text_list):
    cleaned_text = [clean_text(text) for text in text_list]
    processed_input = tokenizer.texts_to_sequences(cleaned_text)
    processed_input = pad_sequences(processed_input, MAXLEN)
    return processed_input


@app.route("/")
def hello():
    """simple landing Page."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    form_inputs = request.form.to_dict()
    text = form_inputs["text"]
    model_input = preprocess([text])
    prediction = model.predict(model_input, batch_size=1, verbose=0).flatten()
    prediction = {
        label: str(round(res * 100)) for (label, res) in zip(LABELS, prediction)
    }
    print(prediction)
    return render_template("index.html", text=text, prediction=prediction.items())


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json()
    if data is None:
        return jsonify({"error": "Missing input"}), 400
    text = data.get("text", "N/A")
    model_input = preprocess([text])
    prediction = model.predict(model_input, batch_size=1, verbose=0).flatten()

    return jsonify({label: str(res) for (label, res) in zip(LABELS, prediction)})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)
