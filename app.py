import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Insurance Dashboard", layout="wide")

# ----------------------------
# DEFAULT DATA FUNCTION
# ----------------------------
@st.cache_data
def generate_data(n=3000):
    np.random.seed(42)

    data = {
        "Age": np.random.randint(18, 70, n),
        "Vehicle_Age": np.random.randint(0, 15, n),
        "Annual_Premium": np.random.randint(2000, 8000, n),
        "Claim_Amount": np.random.randint(500, 200000, n),
        "Fraud": np.random.choice([0, 1], n, p=[0.8, 0.2]),
        "Gender": np.random.choice(["Male", "Female"], n),
        "Vehicle_Type": np.random.choice(["Car", "Bike", "Truck"], n),
    }

    return pd.DataFrame(data)

# ----------------------------
# SIDEBAR
# ----------------------------
st.sidebar.title("⚙️ Settings")

data_option = st.sidebar.radio(
    "Select Data Source",
    ["Use Default Dataset", "Upload Your CSV"]
)

# ----------------------------
# LOAD DATA
# ----------------------------
if data_option == "Upload Your CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("✅ File Uploaded Successfully")
    else:
        st.warning("⚠️ Please upload a CSV file")
        st.stop()

else:
    df = generate_data()
    st.sidebar.info("Using Default Dataset")

# ----------------------------
# BASIC CLEANING (important)
# ----------------------------
df = df.dropna()

# ----------------------------
# TITLE
# ----------------------------
st.title("🚗 Automobile Insurance Claim Analysis")

# ----------------------------
# CHECK REQUIRED COLUMNS
# ----------------------------
required_cols = ["Age", "Vehicle_Age", "Annual_Premium", "Claim_Amount"]

missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    st.error(f"❌ Missing Columns: {missing_cols}")
    st.stop()

# ----------------------------
# SIDEBAR FILTERS (only if columns exist)
# ----------------------------
if "Gender" in df.columns:
    gender = st.sidebar.multiselect("Gender", df["Gender"].unique(), default=df["Gender"].unique())
    df = df[df["Gender"].isin(gender)]

if "Vehicle_Type" in df.columns:
    vehicle = st.sidebar.multiselect("Vehicle Type", df["Vehicle_Type"].unique(), default=df["Vehicle_Type"].unique())
    df = df[df["Vehicle_Type"].isin(vehicle)]

# ----------------------------
# KPI
# ----------------------------
col1, col2, col3 = st.columns(3)

col1.metric("Total Records", len(df))
col2.metric("Avg Claim", f"₹{df['Claim_Amount'].mean():,.0f}")

if "Fraud" in df.columns:
    col3.metric("Fraud Cases", df["Fraud"].sum())
else:
    col3.metric("Fraud Cases", "N/A")

# ----------------------------
# CHARTS
# ----------------------------
col4, col5 = st.columns(2)

with col4:
    st.subheader("Claim Distribution")
    fig, ax = plt.subplots()
    ax.hist(df["Claim_Amount"], bins=30)
    st.pyplot(fig)

with col5:
    if "Vehicle_Type" in df.columns:
        st.subheader("Vehicle Type Count")
        fig, ax = plt.subplots()
        df["Vehicle_Type"].value_counts().plot(kind="bar", ax=ax)
        st.pyplot(fig)

# ----------------------------
# ML MODEL (only if Fraud exists)
# ----------------------------
if "Fraud" in df.columns:
    st.subheader("🤖 Fraud Detection Model")

    X = df[["Age", "Vehicle_Age", "Annual_Premium"]]
    y = df["Fraud"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)

    st.success(f"Model Accuracy: {acc*100:.2f}%")

# ----------------------------
# DATA VIEW
# ----------------------------
st.subheader("📊 Data Preview")
st.dataframe(df.head(50))