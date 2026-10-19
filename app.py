from flask import Flask, request, jsonify
import joblib
import numpy as np

model = joblib.load("data/model.pkl")

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    body = request.get_json(silent=True)

    if not body or "features" not in body:
        return jsonify({"error": "Missing JSON field: features"}), 400

    features = body["features"]

    try:
        prediction = model.predict([np.array(features)])
        return jsonify({"prediction": int(prediction[0])})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/prediction_batch", methods=["POST"])
def predict_batch():
    body = request.get_json(silent=True)

    if not body or "features" not in body:
        return jsonify({"error": "Missing JSON field: features"}), 400

    features = body["features"]

    try:
        predictions = model.predict(np.array(features)).tolist()
        return jsonify({"predictions": predictions})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)