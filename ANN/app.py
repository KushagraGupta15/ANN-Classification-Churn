import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
import pickle
import tensorflow as tf


model = tf.keras.models.load_model("model.h5")
with open("onehot_encoder_geo.pkl", "rb") as f:
  onehot_encoder_geo = pickle.load(f)

with open("label_encoder_gender.pkl", "rb") as f:
    label_encoder_gender = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
st.title("Customer Churn Prediction")

geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
age = st.number_input("Age", min_value=18, max_value=92, value=30)
balance = st.number_input("Balance", min_value=0.0, value=10000.0)
credit_score = st.number_input("Credit Score", min_value=350, max_value=850, value=600)
estimated_salary = st.number_input("Estimated Salary")
tenure = st.number_input("Tenure", min_value=0, max_value=10)
num_of_products = st.number_input("Number of Products", min_value=1, max_value=4)
has_cr_card = st.selectbox("Has Credit Card", ["0", "1"])
is_active_member = st.selectbox("Is Active Member", ["0", "1"])


input_data = pd.DataFrame(
    {   
        "CreditScore": [credit_score],
        'Geography': [geography],
        "Gender": [label_encoder_gender.transform([gender])[0]],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_of_products],
        "HasCrCard": [has_cr_card],
        "IsActiveMember": [is_active_member],
        "EstimatedSalary": [estimated_salary],
    }
)

# geo_encoded = onehot_encoder_geo.transform([input_data ["Geography"]])
# geo_encoded_df = pd.DataFrame(
#     geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(["Geography"])
# )
# input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
# input_data_scaled = scaler.transform(input_data)



geo_encoded = onehot_encoder_geo.transform([[geography]])

geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geo.get_feature_names_out(["Geography"])
)

input_data = pd.concat(
    [input_data.drop("Geography", axis=1).reset_index(drop=True),
     geo_encoded_df],
    axis=1
)

# match training columns
input_data = input_data.reindex(
    columns=scaler.feature_names_in_,
    fill_value=0
)

input_data_scaled = scaler.transform(input_data)
prediciton = model.predict(input_data_scaled)
prediciton_proba = prediciton[0][0]

st.write(f"Prediction Probability: {prediciton_proba:.2f}")

if prediciton_proba > 0.5:
    st.write("The customer is likely to churn.")
else:
    st.write("The customer is unlikely to churn.")