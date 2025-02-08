import streamlit as st
import os
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.feature import VectorAssembler, StringIndexerModel
from pyspark.sql.functions import col

# Initialize Spark Session
def get_spark_session():
    return SparkSession.builder.appName("CreditRiskPrediction").getOrCreate()

spark = get_spark_session()

# Define paths
model_path = "./credit_risk_model"
indexer_dir = "./indexers"

@st.cache_resource
def load_model():
    return RandomForestClassificationModel.load(model_path)

@st.cache_resource
def load_indexers():
    indexers = {}
    for col_name in categorical_cols:
        indexers[col_name] = StringIndexerModel.load(os.path.join(indexer_dir, f"{col_name}_indexer"))
    return indexers

# Load the trained Random Forest model and indexers
model = load_model()
categorical_cols = ["Status", "CreditHistory", "Purpose", "Savings", "Employment", "PersonalStatus", "Debtors", 
                    "Property", "OtherInstallment", "Housing", "Job", "Telephone", "ForeignWorker"]
indexers = load_indexers()

# Define feature columns
feature_cols = ["Duration", "CreditAmount", "InstallmentRate", "ResidenceDuration", "Age", "ExistingCredits", "PeopleLiable"]
feature_cols += [col + "_Index" for col in categorical_cols]

# Streamlit App UI
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Credit Risk Prediction", "Feature Descriptions"])

if page == "Credit Risk Prediction":
    st.title("Credit Risk Prediction App")
    st.write("Enter the details below to predict the credit risk.")

    # User input fields
    input_data = {}
    input_data['Duration'] = st.number_input("Duration (in months)", min_value=1, max_value=100, value=12)
    input_data['CreditAmount'] = st.number_input("Credit Amount", min_value=100, max_value=100000, value=5000)
    input_data['InstallmentRate'] = st.number_input("Installment Rate (%)", min_value=1, max_value=10, value=4)
    input_data['ResidenceDuration'] = st.number_input("Residence Duration (years)", min_value=1, max_value=10, value=3)
    input_data['Age'] = st.number_input("Age", min_value=18, max_value=100, value=35)
    input_data['ExistingCredits'] = st.number_input("Number of Existing Credits", min_value=0, max_value=5, value=1)
    input_data['PeopleLiable'] = st.number_input("Number of People Liable", min_value=0, max_value=2, value=1)

    # Categorical inputs with user-friendly dropdowns
    categorical_options = {
        "Status": {"No checking account": "A11", "Less than 200 DM": "A12", "More than 200 DM": "A13", "No checking account (alternative)": "A14"},
        "CreditHistory": {"No credit taken": "A30", "All credits paid duly": "A31", "Existing credits paid till now": "A32", "Delay in paying previous credits": "A33", "Critical account": "A34"},
        "Purpose": {"Car (new)": "A40", "Car (used)": "A41", "Furniture/equipment": "A42", "Radio/television": "A43", "Domestic appliances": "A44", "Repairs": "A45", "Education": "A46", "Vacation": "A47", "Retraining": "A48", "Business": "A49", "Other": "A410"},
        "Savings": {"Less than 100 DM": "A61", "100 - 500 DM": "A62", "500 - 1000 DM": "A63", "More than 1000 DM": "A64", "No savings account": "A65"},
        "Employment": {"Unemployed": "A71", "Less than 1 year": "A72", "1 to 4 years": "A73", "4 to 7 years": "A74", "More than 7 years": "A75"},
        "PersonalStatus": {"Male, divorced/separated": "A91", "Female, divorced/separated/married": "A92", "Male, single": "A93", "Male, married/widowed": "A94", "Female, single": "A95"},
        "Debtors": {"None": "A101", "Co-applicant": "A102", "Guarantor": "A103"},
        "Property": {"Real estate": "A121", "Savings agreement/life insurance": "A122", "Car or other property": "A123", "No property": "A124"},
        "OtherInstallment": {"Bank": "A141", "Stores": "A142", "None": "A143"},
        "Housing": {"Rent": "A151", "Own": "A152", "For free": "A153"},
        "Job": {"Unemployed/unskilled - non-resident": "A171", "Unskilled - resident": "A172", "Skilled employee": "A173", "Management/self-employed": "A174"},
        "Telephone": {"No": "A191", "Yes, registered under own name": "A192"},
        "ForeignWorker": {"Yes": "A201", "No": "A202"}
    }

    for col_name, options in categorical_options.items():
        selected_label = st.selectbox(f"{col_name}", list(options.keys()))
        input_data[col_name] = options[selected_label]

    col1, col2 = st.columns(2)
    with col1:
            predict_button = st.button("Predict Credit Risk")
    with col2:
            reset_button = st.button("Reset")

    if predict_button:
        input_row = Row(**input_data)
        input_df = spark.createDataFrame([input_row])
        for col_name in categorical_cols:
            input_df = indexers[col_name].transform(input_df)
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        input_df = assembler.transform(input_df)
        predictions = model.transform(input_df)
        prediction_result = predictions.select("prediction").collect()[0]["prediction"]
        prediction_probability = predictions.select("probability").collect()[0]["probability"]
        confidence = max(prediction_probability)
        st.markdown("## Prediction Result")
        if prediction_result == 0:
            st.success(f"✅ The person has a **GOOD** credit risk. (Confidence: {confidence:.2%})")
        else:
            st.error(f"❌ The person has a **BAD** credit risk. (Confidence: {confidence:.2%})")
    if reset_button:
        st.rerun()

elif page == "Feature Descriptions":
    st.title("Feature Descriptions")
    st.write("Here are the descriptions of the features used in the model:")
    st.write("""
    - **Duration**: Loan duration in months
    - **Credit Amount**: The amount of credit requested
    - **Installment Rate**: Installment rate as a percentage of disposable income
    - **Residence Duration**: Number of years the person has lived at their current residence
    - **Age**: Age of the applicant
    - **Existing Credits**: Number of existing credits
    - **People Liable**: Number of people financially responsible
    - **Categorical Features**: Various financial and personal characteristics encoded as indices
    """)

spark.stop()
