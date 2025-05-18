from flask import Flask, request, render_template
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load dataset
data = pd.read_csv("dataset_cuaca3 (1).csv")

# Drop baris yang mengandung NaN
data.dropna(inplace=True)

# Tambah kolom waktu dari timestamp
data['timestamp'] = pd.to_datetime(data['hpwren_timestamp'])
data['hour'] = data['timestamp'].dt.hour
data['day'] = data['timestamp'].dt.day
data['month'] = data['timestamp'].dt.month



# Fitur yang digunakan
features = ['air_pressure', 'avg_wind_direction', 'avg_wind_speed',
            'max_wind_direction', 'max_wind_speed',
            'min_wind_direction', 'min_wind_speed',
            'relative_humidity', 'hour', 'day', 'month']

X = data[features]
y = data['air_temp']

# Latih model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

def get_temp_category(temp):
    if temp < 0:
        return "Dingin"
    elif 0 <= temp < 15:
        return "Sejuk"
    elif 15 <= temp < 27:
        return "Hangat"
    else:
        return "Panas"

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    category = None
    if request.method == "POST":
        try:
            input_data = [
                float(request.form['air_pressure']),
                float(request.form['avg_wind_direction']),
                float(request.form['avg_wind_speed']),
                float(request.form['max_wind_direction']),
                float(request.form['max_wind_speed']),
                float(request.form['min_wind_direction']),
                float(request.form['min_wind_speed']),
                float(request.form['relative_humidity']),
                int(request.form['hour']),
                int(request.form['day']),
                int(request.form['month'])
            ]

            temp_pred = round(model.predict([input_data])[0], 2)
            prediction = temp_pred
            category = get_temp_category(temp_pred)

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction, category=category)

if __name__ == "__main__":
    app.run(debug=True)
