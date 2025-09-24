import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler,LabelEncoder
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

# Connect to the MongoDB cluster to store the inputs and the prediction
uri = "mongodb+srv://anurag:07121998@cluster0.ugo9l.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(uri, server_api=ServerApi('1'))
db = client['Ice_Cream_Sales']     # Create a new database
collection = db['Ice_Cream_Sales_Prediction']     # Create a new collection/table in the database

def load_model():
    with open('ice_cream_sales_polynomial_regression_final_model.pkl', 'rb') as f:
        poly_model3 = pickle.load(f)

    return poly_model3

def preprocesssing_input_data(data):
    df = pd.DataFrame([data])

    return df

def predict_data(data):
    model = load_model()
    processed_data = preprocesssing_input_data(data)
    prediction = model.predict(processed_data)

    return prediction

def main():
    st.title("Ice Cream Sales Prediction")
    st.write("Enter the temperature below to get a prediction for your Ice Cream Sales (units)")

    temprature = st.number_input("Temperature (°C)", value = 20.0) 

    if st.button("Predict the Ice Cream Sales"):
        user_data = {
            "Temperature (°C)": temprature
        }

        prediction = predict_data(user_data)
        st.success(f"Your prediction result is: {round(float(prediction[0]), 3)}")

        user_data["prediction"] = round(float(prediction[0]), 3)     # Add the prediction to the user_data dictionary
        user_data = {key: int(value) if isinstance(value, np.integer) else float(value) if isinstance(value, float) else value for key, value in user_data.items()}    # Convert the values to int or float if they are of type np.integer or np.float
        collection.insert_one(user_data)     # Insert the user_data dictionary/record to the MongoDB collection

if __name__ == "__main__":
    main()