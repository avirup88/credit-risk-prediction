# Credit Risk Prediction System

![Credit Risk App](https://img.shields.io/badge/Python-3.8%2B-blue) ![Spark](https://img.shields.io/badge/Apache%20Spark-3.0%2B-orange) ![Streamlit](https://img.shields.io/badge/Streamlit-Framework-red)

## 📌 Overview

The **Credit Risk Prediction System** is a machine learning application that evaluates a person's creditworthiness based on financial and personal details. It utilizes **Apache Spark** for scalable data processing and **Streamlit** for an interactive user interface.

This project consists of two main components:

1. **Model Training (`credit_risk_model_trainer.py`)**  
   - Trains a **Random Forest** classifier on the **German Credit Risk Dataset**.  
   - Performs **categorical encoding, feature engineering**, and **data balancing** to improve performance.
   - Saves the trained model and feature indexers for future predictions.

2. **Prediction Application (`credit_risk_detector.py`)**  
   - A **Streamlit-powered UI** where users input financial details.
   - Loads the trained model and predicts **Good** or **Bad** credit risk.
   - Displays confidence levels for each prediction.

---

## 🚀 Features

✔ **End-to-End Machine Learning Pipeline**  
✔ **Scalable Processing with Apache Spark**  
✔ **Interactive Credit Risk Prediction via Streamlit UI**  
✔ **Feature Engineering & Class Imbalance Handling**  
✔ **Model Performance Evaluation (ROC-AUC Score)**  

---

## 🏗️ Installation & Setup

### **Prerequisites**
Ensure you have the following installed:
- Python (>= 3.8)
- Apache Spark (>= 3.0)
- Streamlit (`pip install streamlit`)
- PySpark (`pip install pyspark`)

### **Clone the Repository**
```bash
git clone https://github.com/yourusername/credit-risk-prediction.git
cd credit-risk-prediction
```

### **Install Dependencies**
```bash
pip install -r requirements.txt
```

### **Download the Dataset**
The model training script automatically downloads the dataset if it's missing. However, you can manually download it from:  
[German Credit Dataset - UCI](https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data)

---

## ⚙️ How to Run

### **1️⃣ Train the Model**
Run the following command to train the credit risk model:
```bash
python credit_risk_model_trainer.py
```
This will:
- Download the dataset (if not present)
- Train a **Random Forest Classifier**
- Save the trained model in `./credit_risk_model/`
- Store feature indexers in `./indexers/`

### **2️⃣ Start the Credit Risk Prediction App**
Once the model is trained, launch the Streamlit app:
```bash
streamlit run credit_risk_detector.py
```

---

## 🎯 Usage Guide

- Open the **Streamlit UI** in your browser.
- Enter the applicant’s financial and demographic details.
- Click **Predict Credit Risk** to classify the applicant as **Good** ✅ or **Bad** ❌.
- View the **prediction confidence score**.
- Check **Feature Descriptions** for input details.

---

## 📊 Model Performance

After training, the model evaluates performance using **ROC-AUC Score**:

```
Random Forest ROC-AUC Score: 0.85 (Example)
```

Additionally, feature importance is displayed to highlight key predictive variables.

---

## 🏗️ Project Structure

```
📂 credit-risk-prediction
 ├── 📜 credit_risk_model_trainer.py  # Model training script
 ├── 📜 credit_risk_detector.py       # Streamlit UI for credit risk prediction
 ├── 📂 credit_risk_model/            # Trained Random Forest Model
 ├── 📂 indexers/                      # Categorical feature encoders
 ├── 📄 requirements.txt               # Python dependencies
 ├── 📄 README.md                      # Project documentation
```

---

## 🤝 Contribution

We welcome contributions!  
Feel free to **fork** this repo, create a **pull request**, or report issues.  

---

## 📄 License

This project is licensed under the **MIT License**.

---

🚀 **Start predicting credit risk today!**  
📧 For queries, contact: `your.email@example.com`
