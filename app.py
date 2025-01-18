import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, render_template
import numpy as np
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# Route for homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route for single data point prediction
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        # Collecting form data
        data = CustomData(
            Pregnancies=int(request.form.get("Pregnancies")),
            Glucose=float(request.form.get('Glucose')),
            BloodPressure=float(request.form.get('BloodPressure')),
            SkinThickness=float(request.form.get('SkinThickness')),
            Insulin=float(request.form.get('Insulin')),
            BMI=float(request.form.get('BMI')),
            DiabetesPedigreeFunction=float(request.form.get('DiabetesPedigreeFunction')),
            Age=float(request.form.get('Age'))
        )
        
        # Converting to DataFrame
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        
        # Predicting
        predict_pipeline = PredictPipeline()
        result = predict_pipeline.predict(pred_df)
        
        # Rendering result
        return render_template('home.html', result=result[0])

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
