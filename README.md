
# Loan Default Prediction API

This project is a FastAPI-based web service that predicts the likelihood of loan default based on user inputs. The API takes in details such as income, employment length, loan amount, and more, and returns a prediction along with a probability score.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Running the API](#running-the-api)
- [Using the API](#using-the-api)
- [License](#license)

---

## Overview

The Loan Default Prediction API is designed to help classify loan applicants as likely to default or not, using machine learning. By providing personal and financial data, the API returns a prediction (e.g., "Default" or "No Default") and a probability score for the likelihood of default.

## Features

- **Machine Learning Model Integration**: Uses a trained machine learning model (Random Forest) to make predictions.
- **Data Validation**: Ensures incoming data meets the required format.
- **FastAPI**: Built using FastAPI for quick and easy API development.
- **Scalability**: Can be run locally or deployed to cloud services.

## Installation

### Prerequisites

- Python 3.8 or above
- `pip` package manager

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/wahyunanandika/LocalVer_CreditRisk_Streamlit.git
   cd loan-default-prediction-api
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Project Structure**:

   - **`models/`**: Contains the serialized machine learning model and encoders.
   - **`src/`**: Contains utility functions.
   - **`main.py`**: Entry point for the FastAPI app.
   - **`requirements.txt`**: List of required packages.

4. **Ensure Models are Loaded Correctly**:
   - Place your trained `RandomForest.pkl` model file and any necessary encoder files (e.g., `ohe_home_ownership.pkl`, `ohe_loan_grade.pkl`, `ohe_loan_intent.pkl`) in the `models/` directory.

## Running the API

To run the API locally:

```bash
uvicorn api:app --host 127.0.0.1 --port 8000
```
To run the Streamlit locally:
```bash
streamlit run ui.py
```

Access the API at `http://127.0.0.1:8000` in your browser or API client like Postman.

### API Endpoints

- **`POST /predict`**: Receives user input data, applies necessary transformations, and returns a prediction and probability score.

  **Example Request**:
  ```json
  {
    "person_age": 45,
    "person_income": 85000,
    "person_home_ownership": "MORTGAGE",
    "person_emp_length": 10,
    "loan_intent": "EDUCATION",
    "loan_grade": "B",
    "loan_amnt": 15000,
    "loan_int_rate": 12.5,
    "loan_status": 1,
    "person_default_on_file": "N"
  }
  ```

  **Example Response**:
  ```json
  {
    "prediction": "No Default",
    "probability": 0.34
  }
  ```

## License

This project is licensed under the MIT License.

---
