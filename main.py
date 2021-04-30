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

graph = tf.get_default_graph()


app = Flask(__name__)

LINKS_re = re.compile(r"https?://.+?(\s|$)")
NONALPHANUMERIC_re = re.compile(r"[^\w ]")
TOK_PATH = "serialized/tokenizer.pickle"
MODEL_PATH = "serialized/cpu_friendly_model.h5"
MAXLEN = 200  # more than 99th percentile

LABELS = ["toxicity", "severe_toxicity", "identity_attack", "insult", "threat"]


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


@app.route("/add", methods=["POST"])
def add():
    """"""
    form_inputs = request.form.to_dict()
    result = int(form_inputs["a"]) + int(form_inputs["b"])

    return render_template("index.html", result_text=f"A + B = {result}")


@app.route("/secret_add", methods=["POST"])
def secret_add():
    data = request.get_json()
    if data is None:
        return jsonify({"error": "Missing input"}), 400
    a = int(data.get("a", 0))
    b = int(data.get("b", 0))
    return jsonify({"result": a + b})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json()
    if data is None:
        return jsonify({"error": "Missing input"}), 400
    text = data.get("text", "N/A")
    model_input = preprocess([text])
    with graph.as_default():
        prediction = model.predict(model_input, batch_size=1, verbose=0).flatten()

    return jsonify({label: str(res) for (label, res) in zip(LABELS, prediction)})


if __name__ == "__main__":
    print("loading tokenizer..")
    with open(TOK_PATH, "rb") as handle:
        tokenizer = pickle.load(handle)

    print("loading model..")
    model = load_model(MODEL_PATH)

    print("done")
    app.run(host="127.0.0.1", port=8080, debug=True)
