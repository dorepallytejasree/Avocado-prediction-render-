import os
from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load models
clf_model = joblib.load("avocado_classification_model.pkl")
reg_model = joblib.load("avocado_regression_model.pkl")

# Load dataframe to get regions (for dropdown) using relative path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, 'avocado.csv')
df = pd.read_csv(csv_path, index_col=0)
regions = sorted(df['region'].unique())

# List numeric feature names (same as used in model)
numeric_features = [
    'Total Volume', '4046', '4225', '4770', 'Total Bags',
    'Small Bags', 'Large Bags', 'XLarge Bags', 'year', 'month', 'day'
]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction_type = None
    prediction_price = None
    error = None

    if request.method == "POST":
        region = request.form.get("region")
        numeric_vals = []
        for feat in numeric_features:
            val = request.form.get(feat)
            try:
                numeric_vals.append(float(val))
            except:
                error = f"Invalid input for {feat}"
                return render_template("index.html", regions=regions, numeric_features=numeric_features, error=error)

        # Prepare input dataframe
        input_dict = {feat: [val] for feat, val in zip(numeric_features, numeric_vals)}
        input_dict['region'] = [region]
        input_df = pd.DataFrame(input_dict)

        # Add missing 'Unnamed: 0' if model expects it
        if 'Unnamed: 0' not in input_df.columns:
            input_df['Unnamed: 0'] = 0

        try:
            prediction_type = clf_model.predict(input_df)[0]
            prediction_price = reg_model.predict(input_df)[0]
        except Exception as e:
            error = str(e)

    return render_template("index.html", regions=regions, numeric_features=numeric_features,
                           prediction_type=prediction_type, prediction_price=prediction_price, error=error)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
