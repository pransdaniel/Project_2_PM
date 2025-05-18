from flask import Flask, request, render_template
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

# Load pre-trained models from the root directory
clf = joblib.load(r'D:\Praktikum PM\Project_2_PM\models\clf_model.pkl')  # RandomForestClassifier for category
reg = joblib.load(r'D:\Praktikum PM\Project_2_PM\models\reg_model.pkl')  # LinearRegression for temperature

# Define features in the same order as during training
features = ['air_pressure', 'avg_wind_direction', 'avg_wind_speed', 
            'max_wind_direction', 'max_wind_speed', 'min_wind_direction', 
            'min_wind_speed', 'relative_humidity', 'hour', 'day', 'month', 
            'Prev_Air_Temp', 'Temp_3min_avg']

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    category = None
    if request.method == "POST":
        try:
            # Collect and validate inputs
            input_data = {
                'air_pressure': float(request.form['air_pressure']),
                'avg_wind_direction': float(request.form['avg_wind_direction']),
                'avg_wind_speed': float(request.form['avg_wind_speed']),
                'max_wind_direction': float(request.form['max_wind_direction']),
                'max_wind_speed': float(request.form['max_wind_speed']),
                'min_wind_direction': float(request.form['min_wind_direction']),
                'min_wind_speed': float(request.form['min_wind_speed']),
                'relative_humidity': float(request.form['relative_humidity']),
                'hour': int(request.form['hour']),
                'day': int(request.form['day']),
                'month': int(request.form['month']),
                'Prev_Air_Temp': float(request.form['Prev_Air_Temp']),
                'Temp_3min_avg': float(request.form['Temp_3min_avg'])
            }

            # Basic input validation
            if not (0 <= input_data['hour'] <= 23):
                raise ValueError("Hour must be between 0 and 23")
            if not (1 <= input_data['month'] <= 12):
                raise ValueError("Month must be between 1 and 12")
            if not (1 <= input_data['day'] <= 31):
                raise ValueError("Day must be between 1 and 31")

            # Create DataFrame for prediction (mimics Jupyter preprocessing)
            input_df = pd.DataFrame([input_data], columns=features)

            # Ensure no missing values (forward fill logic from Jupyter)
            input_df = input_df.fillna(method='ffill')

            # Predict temperature (regression) and category (classification)
            temp_pred = round(reg.predict(input_df)[0], 2)
            category_pred = clf.predict(input_df)[0]

            prediction = temp_pred
            category = category_pred
        except Exception as e:
            prediction = f"Error: {str(e)}"
            category = None

    return render_template("index.html", prediction=prediction, category=category)

if __name__ == "__main__":
    app.run(debug=True)