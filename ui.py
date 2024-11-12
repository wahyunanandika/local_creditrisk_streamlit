import streamlit as st
import requests

st.title("Loan Application Prediction")

person_age = st.number_input("Age", min_value=18, max_value=100, step=1, value=30)
person_income = st.number_input("Annual Income", min_value=0, step=1000, value=50000)
person_home_ownership = st.selectbox("Home Ownership", ["OWN", "RENT", "MORTGAGE", "OTHER"], index=0)
person_emp_length = st.number_input("Employment Length (years)", min_value=0, max_value=50, step=1, value=5)
loan_intent = st.selectbox("Loan Intent", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"], index=0)
loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"], index=0)
loan_amnt = st.number_input("Loan Amount", min_value=0, step=500, value=5000)
loan_int_rate = st.number_input("Loan Interest Rate", min_value=0.0, step=0.1, format="%.2f", value=5.0)
loan_status = st.selectbox("Loan Status", [0, 1], index=0)  # 0 = No Default, 1 = Default
person_default_on_file = st.selectbox("Default on File", ["Y", "N"], index=1)

input_data = {
    "person_age": person_age,
    "person_income": person_income,
    "person_home_ownership": person_home_ownership,
    "person_emp_length": person_emp_length,
    "loan_intent": loan_intent,
    "loan_grade": loan_grade,
    "loan_amnt": loan_amnt,
    "loan_int_rate": loan_int_rate,
    "loan_status": loan_status,
    "person_default_on_file": person_default_on_file
}

if st.button("Predict"):
    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=input_data)

        if response.status_code == 200:
            result = response.json()

            if 'prediction' in result and 'probability' in result:
                prediction = "Default" if result['prediction'] == 1 else "No Default"
                probability = result['probability']

                st.success(f"Prediction: {prediction}")
                st.write(f"Probability of Default: {probability:.2f}")

            else:
                st.error("Prediction response is missing 'prediction' or 'probability' keys.")
        else:
            st.error("Prediction failed. Please try again later.")

    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to the API: {e}")
