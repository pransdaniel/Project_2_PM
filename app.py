from flask import Flask, request, render_template
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

# Load pre-trained models from the root directory (use forward slashes)
clf = joblib.load('D:/Folder_Kuliah/Semester 6/MachineLearning/Proyek/Project_2_PM/models/clf_model.pkl')  # RandomForestClassifier for category
reg = joblib.load('D:/Folder_Kuliah/Semester 6/MachineLearning/Proyek/Project_2_PM/models/reg_model.pkl')  # LinearRegression for temperature

# Define features in the same order as during training (matching Jupyter Notebook)
features = ['air_pressure', 'avg_wind_direction', 'avg_wind_speed', 
            'max_wind_direction', 'max_wind_speed', 'min_wind_direction', 
            'min_wind_speed', 'relative_humidity', 'Hour', 'Day', 'Month', 
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
                'Hour': int(request.form['hour']),
                'Day': int(request.form['day']),
                'Month': int(request.form['month']),
                'Prev_Air_Temp': float(request.form['Prev_Air_Temp']),
                'Temp_3min_avg': float(request.form['Temp_3min_avg'])
            }

            # Enhanced input validation based on dataset ranges
            if not (900 <= input_data['air_pressure'] <= 920):  # Dataset range: 912.2–912.3 hPa
                raise ValueError("Air Pressure must be between 900 and 920 hPa")
            if not (0 <= input_data['avg_wind_direction'] <= 360):
                raise ValueError("Avg Wind Direction must be between 0 and 360 degrees")
            if not (0 <= input_data['avg_wind_speed'] <= 10):  # Dataset max: 3.0 mph
                raise ValueError("Avg Wind Speed must be between 0 and 10 mph")
            if not (0 <= input_data['max_wind_direction'] <= 360):
                raise ValueError("Max Wind Direction must be between 0 and 360 degrees")
            if not (0 <= input_data['max_wind_speed'] <= 10):  # Dataset max: 3.0 mph
                raise ValueError("Max Wind Speed must be between 0 and 10 mph")
            if not (0 <= input_data['min_wind_direction'] <= 360):
                raise ValueError("Min Wind Direction must be between 0 and 360 degrees")
            if not (0 <= input_data['min_wind_speed'] <= 10):  # Dataset max: 3.0 mph
                raise ValueError("Min Wind Speed must be between 0 and 10 mph")
            if not (0 <= input_data['relative_humidity'] <= 100):  # Dataset range: 33.2–65.8%
                raise ValueError("Relative Humidity must be between 0 and 100%")
            if not (0 <= input_data['Hour'] <= 23):
                raise ValueError("Hour must be between 0 and 23")
            if not (1 <= input_data['Month'] <= 12):
                raise ValueError("Month must be between 1 and 12")
            if not (1 <= input_data['Day'] <= 31):
                raise ValueError("Day must be between 1 and 31")
            if not (50 <= input_data['Prev_Air_Temp'] <= 90):  # Dataset range: 62.24–65.48°F
                raise ValueError("Previous Air Temp must be between 50 and 90°F")
            if not (50 <= input_data['Temp_3min_avg'] <= 90):  # Similar range as Prev_Air_Temp
                raise ValueError("3-Min Avg Temp must be between 50 and 90°F")

            # Create DataFrame for prediction (mimics Jupyter preprocessing)
            input_df = pd.DataFrame([input_data], columns=features)

            # Ensure no missing values (forward fill logic from Jupyter)
            input_df = input_df.fillna(method='ffill')

            # Predict temperature (regression) and category (classification)
            temp_pred_raw = reg.predict(input_df)[0]
            print("Raw regression prediction:", temp_pred_raw, "Type:", type(temp_pred_raw))
            
            # Ensure the prediction is a float before rounding
            temp_pred = round(float(temp_pred_raw), 2)

            category_pred = clf.predict(input_df)[0]
            print("Raw classification prediction:", category_pred, "Type:", type(category_pred))

            prediction = temp_pred
            category = category_pred
        except Exception as e:
            prediction = f"Error: {str(e)}"
            category = None
            print("Prediction error:", str(e))

    return render_template("index.html", prediction=prediction, category=category)

if __name__ == "__main__":
    app.run(debug=True)