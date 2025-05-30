# importing necessary libraries
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import pickle
import pandas as pd
import plotly.express as px

# Set streamlit layout to wide
st.set_page_config(layout="wide")

# Load the trained model
with open('best_model.pkl','rb') as file:
    model = pickle.load(file)

# Load the MinMaxScaler
with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)


# Define the input features for the model
feature_names = ['CreditScore','Age','Tenure','Balance','NumOfProducts','EstimatedSalary','Geography_France','Geography_Germany','Geography_Spain',	'Gender_Female'	,'Gender_Male','HasCrCard_0.0','HasCrCard_1.0','IsActiveMember_0.0','IsActiveMember_1.0']

# Columnsrequiring scaling
scale_vars = ['CreditScore','EstimatedSalary','Tenure','Balance','Age','NumOfProducts']

# updated default values
default_values = [
    600, 30, 2, 8000, 2 , 60000,
    True, False, False,True,False,False,True,False,True
]

# Sidebar setup
st.sidebar.image("Pic 1.jpeg",use_column_width=True)
st.sidebar.header("User Inputs")

# Collect user inputs
user_inputs = {}
for i,feature in enumerate(feature_names):
    if feature in scale_vars:
        user_inputs[feature] = st.sidebar.number_input(
            feature,value=default_values[i], step=1 if isinstance(default_values[i],int) else 0.01
        )
    elif isinstance(default_values[i],bool):
        user_inputs[feature] = st.sidebar.checkbox(feature,value=default_values[i])
    else:
        user_inputs[feature] = st.sidebar.number_input(
            feature,value=default_values[i], step=1
        )

# Convert inputs into data frame
input_data = pd.DataFrame([user_inputs])

# Apply MinMaxScaler to the required columns
input_data_scaled = input_data.copy()
input_data_scaled[scale_vars] = scaler.transform(input_data[scale_vars])


#App Header
st.sidebar.image("Pic 1.jpeg",use_column_width=True)
st.title("Customer Churn Prediction")

#Page Layout
left_col,right_col = st.columns(2)

# Left Page : Feature Importance
with left_col:
    st.header("Feature Importance")
    #Load feature importance data from the excel file
    feature_importance_df = read_excel("./feature_importance.xlsx",usecols=["Feature","feature Importance Score"])
    #Plot the feature importance bar chart
    fig = px.bar(
        feature_importance_df.sort_values(by="Feature Importance Score",ascending = True),
        x = "Feature Importance Score",
        y = "Feature",
        orientation="h",
        title = "Feature Importance",
        labels = {"Feature Importance Score":"Importance","Feature":"Features"},
        width = 400,
        height = 500
    )
    st.plotly_chart(fig)

with right_col:
    st.header("Prediction")
    if st.button("Predict"):
        #Get the predicted probabilties and label
        probabilities = model.predict_proba(input_data_scaled)[0]
        prediction = model.predict(input_data_scaled)[0]
        # Map prediction to label
        prediction_label = "Churned" if predcition == 1 else "Retain"

        # Display results
        st.subheader(f"Predcition Value: {prediction_label}")
        st.write(f"Predcition Probability: {probabilities[1]:.2%} (Churn)")
        st.write(f"Predcition Probability: {probabilities[0]:.2%} (Retain)")
        # Displaying a clear output for prediction
        st.markdown(f"### Output: **{prediction_label}**")
