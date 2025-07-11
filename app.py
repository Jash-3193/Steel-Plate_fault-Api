import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import os

# --- Initialize Flask App ---
app = Flask(__name__, template_folder='.') # Look for templates in the root folder
CORS(app) # Enable Cross-Origin Resource Sharing

# --- Global Variables ---
MODEL = None
SCALER = None
FAULT_NAMES = [
    'Pastry', 'Z_Scratch', 'K_Scatch', 'Stains',
    'Dirtiness', 'Bumps', 'Other_Faults'
]

def load_model_assets():
    """Load the trained model and scaler from disk."""
    global MODEL, SCALER
    model_path = 'steel_fault_classifier.pkl'
    scaler_path = 'scaler.pkl'

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(
            "Model or scaler not found. Please run 'train_model.py' first."
        )

    print("Loading model assets...")
    MODEL = joblib.load(model_path)
    SCALER = joblib.load(scaler_path)
    print("Model and scaler loaded successfully.")

@app.route('/')
def home():
    """Serve the main HTML page with the prediction form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """The main prediction endpoint. Expects JSON input with a 'features' key."""
    if MODEL is None or SCALER is None:
        return jsonify({'error': 'Model not loaded. Check server logs.'}), 500

    try:
        data = request.get_json()
        if 'features' not in data:
            return jsonify({'error': "Missing 'features' key in request."}), 400

        features = data['features']
        if not isinstance(features, list) or len(features) != 27:
            return jsonify({'error': f"Expected a list of 27 features."}), 400

        input_data = np.array(features).reshape(1, -1)

    except Exception as e:
        return jsonify({'error': f"Invalid JSON or data format: {e}"}), 400

    try:
        scaled_data = SCALER.transform(input_data)
        prediction_array = MODEL.predict(scaled_data)
        results = prediction_array[0]
        identified_faults = [FAULT_NAMES[i] for i, val in enumerate(results) if val == 1]

        if not identified_faults:
            identified_faults = ["No Faults Detected"]

        response = {
            'prediction': {
                'status': 'success',
                'identified_faults': identified_faults
            },
            'raw_output': results.tolist()
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': f"An error occurred during prediction: {e}"}), 500

# --- Load Model Assets on Startup ---
try:
    load_model_assets()
    print("Model assets loaded successfully on startup.")
except FileNotFoundError as e:
    print(f"CRITICAL ERROR: Could not load model assets. {e}")

# This part is for local testing only. Gunicorn will run the app in production.
if __name__ == '__main__':
    app.run(debug=True)
