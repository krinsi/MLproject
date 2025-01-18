import sys
import os
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Add the project root directory to the Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(PROJECT_ROOT)

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

## Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        # Get input data from the form and ensure correct variable mapping
        try:
            data = CustomData(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('ethnicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                reading_score=float(request.form.get('reading_score')),  # corrected field name
                writing_score=float(request.form.get('writing_score'))   # corrected field name
            )

            # Prepare the data for prediction
            pred_df = data.get_data_as_data_frame()
            print("Input Data for Prediction:", pred_df)

            # Initialize prediction pipeline
            predict_pipeline = PredictPipeline()
            print("Mid Prediction")

            # Make predictions
            results = predict_pipeline.predict(pred_df)
            print("After Prediction")

            # Return results to the template
            return render_template('home.html', results=results[0])

        except Exception as e:
            print(f"Error during prediction: {e}")
            return render_template('home.html', results="Error occurred during prediction. Please check your inputs.")

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
