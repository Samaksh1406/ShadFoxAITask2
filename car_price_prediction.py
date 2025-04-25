from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Initialize Flask App
app = Flask(__name__)

# Load and preprocess the dataset
df = pd.read_csv("car.csv")
df = df.drop('Car_Name', axis=1)
df = pd.get_dummies(df, drop_first=True)

# Define features and target variable
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

# Train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Calculate R² Score
r2 = r2_score(y_test, model.predict(X_test))

# Home route to display the form
@app.route('/')
def home():
    return render_template('index.html')

# Route to predict car selling price based on user input
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from the form
        year = int(request.form['year'])
        present_price = float(request.form['present_price'])
        kms_driven = int(request.form['kms_driven'])
        owner = int(request.form['owner'])
        fuel_type = request.form['fuel_type']
        seller_type = request.form['seller_type']
        transmission = request.form['transmission']

        # Convert categorical inputs to dummies
        input_dict = {
            'Year': year,
            'Present_Price': present_price,
            'Kms_Driven': kms_driven,
            'Owner': owner,
            'Fuel_Type_Diesel': 1 if fuel_type == 'Diesel' else 0,
            'Fuel_Type_Petrol': 1 if fuel_type == 'Petrol' else 0,
            'Seller_Type_Individual': 1 if seller_type == 'Individual' else 0,
            'Transmission_Manual': 1 if transmission == 'Manual' else 0
        }

        # Convert the input into a DataFrame and align columns
        input_df = pd.DataFrame([input_dict])
        input_df = input_df.reindex(columns=X.columns, fill_value=0)

        # Predict the selling price
        predicted_price = max(0, model.predict(input_df)[0])


        return render_template(
            'index.html',
            prediction_text=f"Estimated Selling Price: ₹ {predicted_price:.2f} Lakhs",
            r2_score=f"Model Accuracy (R² Score): {r2:.2f}"
        )

    except Exception as e:
        return render_template('index.html', prediction_text="❌ Error: Please check your inputs.")
        
if __name__ == "__main__":
    app.run(debug=True)
