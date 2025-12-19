import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Loan Approval Prediction",
    layout="wide"
)

# --------------------------------------------------
# Title
# --------------------------------------------------
st.title("🏦 Loan Approval Prediction System")
st.markdown(
    "This application predicts whether a loan will be approved based on "
    "applicant details using Machine Learning."
)

# --------------------------------------------------
# Load Data
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("LoanApprovalPrediction.csv")
    return df

data = load_data()

# --------------------------------------------------
# Sidebar Inputs
# --------------------------------------------------
st.sidebar.header("📋 Applicant Details")

Gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
Married = st.sidebar.selectbox("Married", ["Yes", "No"])
Dependents = st.sidebar.selectbox("Dependents", ["0", "1", "2", "3+"])
Education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
Self_Employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])
ApplicantIncome = st.sidebar.number_input("Applicant Income", min_value=0)
CoapplicantIncome = st.sidebar.number_input("Coapplicant Income", min_value=0)
LoanAmount = st.sidebar.number_input("Loan Amount", min_value=0)
Loan_Amount_Term = st.sidebar.number_input("Loan Amount Term", min_value=0)
Credit_History = st.sidebar.selectbox("Credit History", [1.0, 0.0])
Property_Area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# --------------------------------------------------
# Feature Explanation
# --------------------------------------------------
with st.expander("ℹ️ Feature Explanation"):
    st.write("""
    - **ApplicantIncome**: Income of the applicant  
    - **CoapplicantIncome**: Income of co-applicant  
    - **LoanAmount**: Loan amount requested  
    - **Credit_History**: 1 = Good, 0 = Bad  
    - **Property_Area**: Location of property  
    """)

# --------------------------------------------------
# Dataset Preview
# --------------------------------------------------
st.subheader("📊 Dataset Preview")
st.dataframe(data.head())

# --------------------------------------------------
# Data Preprocessing
# --------------------------------------------------
data = data.drop("Loan_ID", axis=1)

# Fill missing values
for col in data.columns:
    if data[col].dtype == "object":
        data[col].fillna(data[col].mode()[0], inplace=True)
    else:
        data[col].fillna(data[col].mean(), inplace=True)

# Encode categorical variables
le = LabelEncoder()
for col in data.columns:
    if data[col].dtype == "object":
        data[col] = le.fit_transform(data[col])

# --------------------------------------------------
# EDA
# --------------------------------------------------
st.subheader("📈 Exploratory Data Analysis")

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, cmap="BrBG", ax=ax)
st.pyplot(fig)

# --------------------------------------------------
# Model Training
# --------------------------------------------------
X = data.drop("Loan_Status", axis=1)
y = data["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.info("🧠 Model Used: **Logistic Regression**")
st.success(f"✅ Model Accuracy: **{accuracy * 100:.2f}%**")

# --------------------------------------------------
# Prediction
# --------------------------------------------------
st.subheader("🔮 Loan Approval Prediction")

input_data = pd.DataFrame([{
    "Gender": Gender,
    "Married": Married,
    "Dependents": Dependents,
    "Education": Education,
    "Self_Employed": Self_Employed,
    "ApplicantIncome": ApplicantIncome,
    "CoapplicantIncome": CoapplicantIncome,
    "LoanAmount": LoanAmount,
    "Loan_Amount_Term": Loan_Amount_Term,
    "Credit_History": Credit_History,
    "Property_Area": Property_Area
}])

for col in input_data.columns:
    if input_data[col].dtype == "object":
        input_data[col] = le.fit_transform(input_data[col])

if st.button("Predict Loan Status"):
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Not Approved")

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.markdown(
    "Built by **Shivani Shetty** | Data Analyst Portfolio | "
    "Python • Machine Learning • Streamlit"
)
