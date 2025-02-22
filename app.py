from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model and encoder
model = joblib.load('model.pkl')
encoder = joblib.load('encoder.pkl')
model_features = joblib.load('model_features.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from form
    input_data = {
        "area": int(request.form['area']),
        "bedrooms": int(request.form['bedrooms']),
        "bathrooms": int(request.form['bathrooms']),
        "stories": int(request.form['stories']),
        "mainroad": request.form['mainroad'],
        "guestroom": request.form['guestroom'],
        "basement": request.form['basement'],
        "hotwaterheating": request.form['hotwaterheating'],
        "airconditioning": request.form['airconditioning'],
        "parking": int(request.form['parking']),
        "prefarea": request.form['prefarea'],
        "furnishingstatus": request.form['furnishingstatus']
    }

    # Convert input to DataFrame
    input_df = pd.DataFrame([input_data])

    # Apply encoding if needed
    categorical_cols = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea", "furnishingstatus"]
    if encoder:
        encoded_values = encoder.transform(input_df[categorical_cols])
        encoded_df = pd.DataFrame(encoded_values, columns=encoder.get_feature_names_out(categorical_cols))
        input_df = input_df.drop(columns=categorical_cols)
        input_df = pd.concat([input_df, encoded_df], axis=1)

    # Ensure column order matches model training
    for col in model_features:
        if col not in input_df:
            input_df[col] = 0  # Add missing columns

    # Predict house price
    predicted_price = model.predict(input_df)[0]

    return render_template('result.html', price=predicted_price)

if __name__ == '__main__':
    app.run(debug=True)
