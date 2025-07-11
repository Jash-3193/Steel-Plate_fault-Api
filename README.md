AI Steel Plate Fault Detector
A user-friendly web application that uses a machine learning model to classify seven different types of faults on steel plates. This tool provides an interactive interface for real-time predictions.

Live Application
Try the live version deployed on Render:

https://steel-plate-fault-api.onrender.com

Application Preview
How to Use the Web App
The application provides three easy ways to get a prediction:

Load Example: Click the "Load Example" button to instantly fill the form with random data from the dataset. This is the quickest way to see the model in action.

Upload CSV: Click "Upload CSV" to select a file from your computer. The data will appear in a table; click any row to load its values into the form.

Manual Entry: Directly type or paste the 27 feature values into the input fields.

Once the data is loaded, click the "Predict Fault" button to see the result.

Features
Interactive UI: A clean and responsive user interface for easy interaction.

Multiple Input Methods: Supports manual entry, random examples, and CSV file uploads.

Real-Time Predictions: Get instant fault classifications from the trained model.

Multi-Label Classification: Predicts one or more of seven possible fault types.

RESTful API Backend: Built on a robust Flask backend that also serves the front end.

Technology Stack
Frontend: HTML, Tailwind CSS, JavaScript

Backend: Python, Flask

Machine Learning: Scikit-learn, Pandas, NumPy

WSGI Server: Gunicorn

Deployment: Render, Git & Git LFS

Local Development Setup
To run this project on your local machine, follow these steps:

1. Prerequisites

Python 3.11

Git

Git LFS (Installation Guide)

2. Clone the Repository

# Initialize Git LFS
git lfs install

# Clone the repository
git clone https://github.com/Ashvin1125/Steel-Plate_fault-Api.git
cd Steel-Plate_fault-Api

3. Create a Virtual Environment (Recommended)

# Create a virtual environment
python -m venv env

# Activate it
# On Windows:  env\Scripts\activate
# On macOS/Linux: source env/bin/activate

4. Install Dependencies
The required packages are listed in requirements.txt.

pip install -r requirements.txt

5. Run the Flask App
The application uses the flask command to run the development server.

flask run

The application will be available at http://127.0.0.1:5000.

API Endpoint Details
While the project is a full web app, the backend still exposes a JSON API endpoint.

POST /predict
Request Body: A JSON object with a key features, which is a list containing 27 numeric values.

Example cURL Request:

curl -X POST http://127.0.0.1:5000/predict \
-H "Content-Type: application/json" \
-d '{
    "features": [42, 50, 2709, 2749, 267, 17, 44, 24220, 76, 108, 1687, 1, 0, 80, 0.0498, 0.2415, 0.1818, 0.0047, 0.4706, 0.1, 1, 2.4265, 0.9031, 1.6435, 0.8182, -0.2913, 0.5822]
}'

Success Response (200 OK):

{
  "prediction": {
    "identified_faults": ["Pastry"],
    "status": "success"
  },
  "raw_output": [1, 0, 0, 0, 0, 0, 0]
}
