# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Telecom Customer Churn Prediction", layout="wide")

@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        # Default bundled dataset
        df = pd.read_csv("telecom_churn.csv")

    # Clean column names
    df.columns = df.columns.str.strip()

    # Handle missing TotalCharges
    if "TotalCharges" not in df.columns and {"MonthlyCharges", "tenure"}.issubset(df.columns):
        df["TotalCharges"] = df["MonthlyCharges"] * df["tenure"]

    # Convert to numeric safely
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    df.dropna(inplace=True)
    return df

@st.cache_resource
def train_model(df):
    X = df.drop(["Churn", "customerID"], axis=1, errors="ignore")
    # Encode categorical variables
    X = pd.get_dummies(X, drop_first=True)
    y = df["Churn"].apply(lambda x: 1 if x in ["Yes", 1, "True", "true"] else 0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    return model, X.columns, acc

# -------------------- UI --------------------
st.title("📊 Telecom Customer Churn Prediction Demo")
st.markdown("Upload your own dataset or use the default one to test churn predictions.")

uploaded_file = st.file_uploader("Upload a telecom_churn.csv file", type=["csv"])
df = load_data(uploaded_file)

st.subheader("Sample Data")
st.write(df.head())

model, feature_cols, acc = train_model(df)
st.success(f"✅ Model trained with accuracy: {acc:.2%}")

# -------------------- Prediction Form --------------------
st.subheader("🔮 Try a Prediction")
with st.form("prediction_form"):
    input_data = {}
    for col in feature_cols:
        if "charges" in col.lower() or "tenure" in col.lower():
            input_data[col] = st.number_input(f"{col}", min_value=0.0, value=10.0)
        else:
            input_data[col] = st.text_input(f"{col}", "0")

    submitted = st.form_submit_button("Predict Churn")

if submitted:
    input_df = pd.DataFrame([input_data])
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=feature_cols, fill_value=0)

    prediction = model.predict(input_df)[0]
    st.write("### 🎯 Prediction Result:")
    if prediction == 1:
        st.error("⚠️ This customer is **likely to churn**.")
    else:
        st.success("✅ This customer is **not likely to churn**.")
