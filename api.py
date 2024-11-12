from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from src import utils


app = FastAPI()


model = utils.deserialize_data("models/RandomForest.pkl")
if isinstance(model, tuple):  
    model = model[0] 

ohe_home_ownership = utils.deserialize_data("models/ohe_home_ownership.pkl")
ohe_loan_grade = utils.deserialize_data("models/ohe_loan_grade.pkl")
ohe_loan_intent = utils.deserialize_data("models/ohe_loan_intent.pkl")


class Item(BaseModel):
    person_age: int
    person_income: int
    person_home_ownership: str
    person_emp_length: int
    loan_intent: str
    loan_grade: str
    loan_amnt: float
    loan_int_rate: float
    loan_status: int
    person_default_on_file: str

@app.post("/predict")
def predict(item: Item):
    try:
        # Convert input data to dataframe using model_dump() (replacement for dict())
        data = pd.DataFrame([item.model_dump()])


        home_ownership_encoded = ohe_home_ownership.transform(data[['person_home_ownership']])
        loan_grade_encoded = ohe_loan_grade.transform(data[['loan_grade']])
        loan_intent_encoded = ohe_loan_intent.transform(data[['loan_intent']])

        
        home_ownership_df = pd.DataFrame(home_ownership_encoded, columns=ohe_home_ownership.get_feature_names_out())
        loan_grade_df = pd.DataFrame(loan_grade_encoded, columns=ohe_loan_grade.get_feature_names_out())
        loan_intent_df = pd.DataFrame(loan_intent_encoded, columns=ohe_loan_intent.get_feature_names_out())


        data = pd.concat([data, home_ownership_df, loan_grade_df, loan_intent_df], axis=1)

        data.drop(columns=['person_home_ownership', 'loan_grade', 'loan_intent'], inplace=True)

        required_columns = list(model.feature_names_in_)

        for col in required_columns:
            if col not in data.columns:
                data[col] = 0

        data = data[required_columns]

        proba = model.predict_proba(data)[:, 1]

        if proba is None or len(proba) == 0:
            raise ValueError("Model returned empty probabilities.")

        threshold = 0.43434343434343436
        prediction = int(proba >= threshold)

        print(f"Prediction: {prediction}, Probability: {proba[0]}")

        return {"prediction": prediction, "probability": proba[0]}

    except Exception as e:
        print(f"Error: {e}")
