import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# --- Configuration ---
DATA_FILE = 'steel_faults.csv'
MODEL_FILE = 'steel_fault_classifier.pkl'
SCALER_FILE = 'scaler.pkl'

def train_and_save_model():
    """
    Loads data, trains a multi-output random forest classifier,
    and saves the model and scaler to disk.
    """
    print("--- Starting Model Training ---")

    # --- 1. Load Data ---
    print(f"Loading data from '{DATA_FILE}'...")
    if not os.path.exists(DATA_FILE):
        print(f"Error: Data file '{DATA_FILE}' not found.")
        print("Please make sure you have created this file from the .nna files first.")
        return

    df = pd.read_csv(DATA_FILE)

    # Define feature and target columns
    target_columns = [
        'Pastry', 'Z_Scratch', 'K_Scatch', 'Stains',
        'Dirtiness', 'Bumps', 'Other_Faults'
    ]
    feature_columns = [col for col in df.columns if col not in target_columns]

    X = df[feature_columns]
    y = df[target_columns]

    print("Data loaded successfully.")
    print(f"Features shape: {X.shape}")
    print(f"Targets shape: {y.shape}")

    # --- 2. Data Splitting and Scaling ---
    print("Splitting and scaling data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and fit the scaler on the training data ONLY
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) # Use the same scaler for the test set

    # --- 3. Model Training ---
    print("Training the Random Forest model...")
    # Define the base classifier
    forest = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    # Wrap it with MultiOutputClassifier for multi-label classification
    multi_target_forest = MultiOutputClassifier(forest)

    # Train the model
    multi_target_forest.fit(X_train_scaled, y_train)
    print("Model training complete.")

    # --- 4. Model Evaluation (Optional) ---
    print("Evaluating model performance on the test set...")
    y_pred = multi_target_forest.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Subset Accuracy on Test Set: {accuracy:.4f}")

    # --- 5. Save the Model and Scaler ---
    print(f"Saving model to '{MODEL_FILE}'...")
    joblib.dump(multi_target_forest, MODEL_FILE)

    print(f"Saving scaler to '{SCALER_FILE}'...")
    joblib.dump(scaler, SCALER_FILE)

    print("\n--- Model and scaler have been saved successfully! ---")
    print("You are now ready to run the Flask API.")

if __name__ == '__main__':
    train_and_save_model()
=======
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# --- Configuration ---
DATA_FILE = 'steel_faults.csv'
MODEL_FILE = 'steel_fault_classifier.pkl'
SCALER_FILE = 'scaler.pkl'

def train_and_save_model():
    """
    Loads data, trains a multi-output random forest classifier,
    and saves the model and scaler to disk.
    """
    print("--- Starting Model Training ---")

    # --- 1. Load Data ---
    print(f"Loading data from '{DATA_FILE}'...")
    if not os.path.exists(DATA_FILE):
        print(f"Error: Data file '{DATA_FILE}' not found.")
        print("Please make sure you have created this file from the .nna files first.")
        return

    df = pd.read_csv(DATA_FILE)

    # Define feature and target columns
    target_columns = [
        'Pastry', 'Z_Scratch', 'K_Scatch', 'Stains',
        'Dirtiness', 'Bumps', 'Other_Faults'
    ]
    feature_columns = [col for col in df.columns if col not in target_columns]

    X = df[feature_columns]
    y = df[target_columns]

    print("Data loaded successfully.")
    print(f"Features shape: {X.shape}")
    print(f"Targets shape: {y.shape}")

    # --- 2. Data Splitting and Scaling ---
    print("Splitting and scaling data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and fit the scaler on the training data ONLY
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) # Use the same scaler for the test set

    # --- 3. Model Training ---
    print("Training the Random Forest model...")
    # Define the base classifier
    forest = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    # Wrap it with MultiOutputClassifier for multi-label classification
    multi_target_forest = MultiOutputClassifier(forest)

    # Train the model
    multi_target_forest.fit(X_train_scaled, y_train)
    print("Model training complete.")

    # --- 4. Model Evaluation (Optional) ---
    print("Evaluating model performance on the test set...")
    y_pred = multi_target_forest.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Subset Accuracy on Test Set: {accuracy:.4f}")

    # --- 5. Save the Model and Scaler ---
    print(f"Saving model to '{MODEL_FILE}'...")
    joblib.dump(multi_target_forest, MODEL_FILE)

    print(f"Saving scaler to '{SCALER_FILE}'...")
    joblib.dump(scaler, SCALER_FILE)

    print("\n--- Model and scaler have been saved successfully! ---")
    print("You are now ready to run the Flask API.")

if __name__ == '__main__':
    train_and_save_model()
