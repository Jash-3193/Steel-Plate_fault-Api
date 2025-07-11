import numpy as np
from flask import Flask, request, jsonify
import joblib
import os

# --- Initialize Flask App ---
app = Flask(__name__)

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
            "Model or scaler not found. Please run the 'train_model.py' script first."
        )

    print("Loading model assets...")
    MODEL = joblib.load(model_path)
    SCALER = joblib.load(scaler_path)
    print("Model and scaler loaded successfully.")

@app.route('/')
def home():
    """A simple welcome route to test if the API is running."""
    return "<h1>Steel Fault Prediction API</h1><p>Send a POST request to /predict to get a fault classification.</p>"

@app.route('/predict', methods=['POST'])
def predict():
    """
    The main prediction endpoint.
    Expects JSON input with a 'features' key.
    """
    if MODEL is None or SCALER is None:
        return jsonify({'error': 'Model not loaded. Check server logs.'}), 500

    # --- 1. Get and Validate Input Data ---
    try:
        data = request.get_json()
        if 'features' not in data:
            return jsonify({'error': "Missing 'features' key in request."}), 400

        features = data['features']
        # Ensure features are in a list of lists for single sample prediction
        if not isinstance(features, list) or not all(isinstance(i, (int, float)) for i in features):
             return jsonify({'error': "'features' must be a list of numbers."}), 400

        if len(features) != 27:
            return jsonify({'error': f"Expected 27 features, but got {len(features)}."}), 400

        # Convert to numpy array and reshape for a single sample
        input_data = np.array(features).reshape(1, -1)

    except Exception as e:
        return jsonify({'error': f"Invalid JSON or data format: {e}"}), 400

    # --- 2. Make Prediction ---
    try:
        # Scale the input data using the loaded scaler
        scaled_data = SCALER.transform(input_data)

        # Get the prediction from the model
        prediction_array = MODEL.predict(scaled_data)

        # --- 3. Format the Response ---
        # The prediction is a 2D array (e.g., [[0, 1, 0, 0, 0, 1, 0]])
        # We extract the first (and only) prediction
        results = prediction_array[0]

        # Create a dictionary of the identified faults
        identified_faults = [FAULT_NAMES[i] for i, val in enumerate(results) if val == 1]

        # If no faults are found, say so explicitly
        if not identified_faults:
            identified_faults = ["No Faults Detected"]

        response = {
            'prediction': {
                'status': 'success',
                'identified_faults': identified_faults
            },
            'raw_output': results.tolist() # Also return the raw 0/1 array
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

=======
import numpy as np
from flask import Flask, request, jsonify
import joblib
import os

# --- Initialize Flask App ---
app = Flask(__name__)

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
            "Model or scaler not found. Please run the 'train_model.py' script first."
        )

    print("Loading model assets...")
    MODEL = joblib.load(model_path)
    SCALER = joblib.load(scaler_path)
    print("Model and scaler loaded successfully.")

@app.route('/')
def home():
    """A simple welcome route to test if the API is running."""
    return "<h1>Steel Fault Prediction API</h1><p>Send a POST request to /predict to get a fault classification.</p>"

@app.route('/predict', methods=['POST'])
def predict():
    """
    The main prediction endpoint.
    Expects JSON input with a 'features' key.
    """
    if MODEL is None or SCALER is None:
        return jsonify({'error': 'Model not loaded. Check server logs.'}), 500

    # --- 1. Get and Validate Input Data ---
    try:
        data = request.get_json()
        if 'features' not in data:
            return jsonify({'error': "Missing 'features' key in request."}), 400

        features = data['features']
        # Ensure features are in a list of lists for single sample prediction
        if not isinstance(features, list) or not all(isinstance(i, (int, float)) for i in features):
             return jsonify({'error': "'features' must be a list of numbers."}), 400

        if len(features) != 27:
            return jsonify({'error': f"Expected 27 features, but got {len(features)}."}), 400

        # Convert to numpy array and reshape for a single sample
        input_data = np.array(features).reshape(1, -1)

    except Exception as e:
        return jsonify({'error': f"Invalid JSON or data format: {e}"}), 400

    # --- 2. Make Prediction ---
    try:
        # Scale the input data using the loaded scaler
        scaled_data = SCALER.transform(input_data)

        # Get the prediction from the model
        prediction_array = MODEL.predict(scaled_data)

        # --- 3. Format the Response ---
        # The prediction is a 2D array (e.g., [[0, 1, 0, 0, 0, 1, 0]])
        # We extract the first (and only) prediction
        results = prediction_array[0]

        # Create a dictionary of the identified faults
        identified_faults = [FAULT_NAMES[i] for i, val in enumerate(results) if val == 1]

        # If no faults are found, say so explicitly
        if not identified_faults:
            identified_faults = ["No Faults Detected"]

        response = {
            'prediction': {
                'status': 'success',
                'identified_faults': identified_faults
            },
            'raw_output': results.tolist() # Also return the raw 0/1 array
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

