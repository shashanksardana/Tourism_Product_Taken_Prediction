import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="ShashankSardana/Tourism-Product-Taken-Prediction", filename="best_tourism_product_taken_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Tourism Prediction
st.title("Tourism Package Prediction")
st.write("""
This application predicts the likelihood of predicting Wellness tourism packages being taken by customers.
Please enter the data below to get a prediction.
""")

# User input
Age = st.number_input("Age", min_value=18, max_value=100, value=30)
TypeOfContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
CityTier = st.selectbox("City Tier", [1, 2, 3])
DurationOfPitch = st.number_input("Duration of Sales Pitch (mins)", min_value=0, max_value=60, value=15)
Occupation = st.selectbox("Occupation", ["Salaried", "Self Employed", "Free Lancer", "Student", "Housewife"])
Gender = st.selectbox("Gender", ["Male", "Female"])
NumberOfPersonVisiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
NumberOfFollowups = st.number_input("Number of Follow-ups", min_value=0, max_value=20, value=3)
#PreferredPropertyStar = st.selectbox("Preferred Property Star Rating", [1, 2, 3, 4, 5])
PreferredPropertyStar = st.selectbox("Preferred Property Star Rating", [1, 2, 3, 4, 5])
MaritalStatus = st.selectbox("Marital Status", ["Married", "Single", "Divorced"])
NumberOfTrips = st.number_input("Number of Trips per Year", min_value=0, max_value=20, value=2)
Passport = st.selectbox("Passport", ["Yes", "No"])
PitchSatisfactionScore = st.number_input("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
OwnCar = st.selectbox("Own Car", ["Yes", "No"])
#NumberOfChildrenVisiting = st.number_input("Number of Children Visiting", min_value=0, max_value=5, value=3)
Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
MonthlyIncome = st.number_input("Monthly Income (₹)", min_value=1000, max_value=500000, value=50000, step=1000)

# ------- Prepare Input for API / Model ---------

input_data = {
    "Age": Age,
    "TypeOfContact": TypeOfContact,
    "CityTier": CityTier,
    "DurationOfPitch": DurationOfPitch,
    "Occupation": Occupation,
    "Gender": Gender,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "NumberOfFollowups": NumberOfFollowups,
    "PreferredPropertyStar": PreferredPropertyStar,
    "MaritalStatus": MaritalStatus,
    "NumberOfTrips": NumberOfTrips,
    "Passport": Passport,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "OwnCar": OwnCar,
    #"NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "Designation": Designation,
    "MonthlyIncome": MonthlyIncome
}

st.write("### Input Data Preview", input_data)

# ------- Prediction Button ---------
if st.button("Predict Package Purchase"):
    st.write("🔄 Sending data to model...")
    import pandas as pd
df_input = pd.DataFrame([input_data])
for _col in ['Passport','OwnCar']:
    if _col in df_input.columns:
        df_input[_col] = df_input[_col].replace({'Yes':1,'No':0,'yes':1,'no':0})
prediction = model.predict(df_input)[0]

result = "Tourism Package Purchased" if prediction == 1 else "Tourism Package Not Purchased"
st.subheader("Prediction Result:")
st.success(f"The model predicts: **{result}**")
