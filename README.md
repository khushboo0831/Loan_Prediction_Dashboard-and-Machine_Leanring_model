# 🏦 Loan Prediction Dashboard & Machine Learning Model

## 📌 Project Overview

This end-to-end project focuses on analyzing, visualizing, and predicting loan approval status using both an interactive Power BI dashboard and a machine learning model (Random Forest Classifier). It aims to help financial institutions make smarter, data-driven decisions on loan approvals based on applicant demographics and financial attributes.

---

## ❓ Problem Statement

Financial institutions often face challenges in identifying suitable loan candidates due to:

* Manual processes that are **time-consuming** and **error-prone**
* **Biases** that occur without data-driven support

This project addresses these issues by:

✅ Visualizing loan applicant data for insight-driven decision-making
✅ Predicting loan approvals based on historical data using ML algorithms

---

## 📂 Dataset Details

The project uses two datasets (`loan_dataset.xlsx` and `loan_dataset26.xlsx`) with over 5,000 customer records. Key fields include:

* `Loan`: Loan taken (Yes/No)
* `Age`, `Income`, `Mortgage`, `Experience`
* `Education`: UG / PG / Graduate
* `Fixed Deposit`, `Demat`, `Net Banking` (Yes/No)

🔍 **Irrelevant columns** like Serial No., Family Members, Pin-code, and ID were removed for clean analysis.

---

## 📊 Power BI Dashboard

The dashboard offers rich insights into customer demographics and behavior.

### ✅ Key Insights:

* **Total Customers**: 5,000+
* **Average Mortgage**: ₹451.99K
* **Education vs Mortgage**: Postgraduates show highest mortgage rates
* **Loan Distribution**: Highest among ages 30–59
* **Net Banking Usage**: Common in middle-aged groups
* **Fixed Deposit & Demat**: More common among experienced/senior customers
* **High-Income Group**: Seen in customers aged 40–49

### 📈 Segment Analysis:

Insights into digital adoption, financial habits, and eligibility trends help banks target products effectively.

---

## 🧪 Exploratory Data Analysis (Python)

Script: `loan_prediction_analysis.py`

### 🔍 Preprocessing:

* Dropped irrelevant fields
* Binned `Age` and `Income` into segments
* Cleaned missing values and standardized categorical features

### 📊 Visual Analysis Includes:

* Mortgage by Education (Pie Chart)
* Net Banking vs Loan Status
* Age-wise Net Banking & Loan approvals
* Income vs Loan Approval
* FD and Demat analysis by age group

---

## 🤖 Machine Learning Model

### ✅ Model: Random Forest Classifier

An ensemble-based ML algorithm that builds multiple decision trees and combines them for robust classification.

### 🎯 Why Random Forest?

* Handles mixed data types (categorical + continuous)
* Performs well with imbalanced datasets
* Provides **feature importance** metrics
* Offers high accuracy without complex tuning

### ⚙️ Steps:

* **Feature Selection**: Used `Age`, `Income`, `Mortgage`, `Experience`
* **Train-Test Split**: 80/20
* **Model Training**:

  ```python
  model = RandomForestClassifier()
  model.fit(X_train, y_train)
  ```
* **Evaluation Metrics**:

  * Accuracy: \~85%
  * Precision, Recall, F1-score: High for both classes
  * Confusion Matrix for prediction quality
  * Feature Importance + Mutual Info Scores

---

## 🔍 Correlation Analysis

A heatmap was used to find key correlated features and refine model input:

```python
sns.heatmap(data1.corr(), annot=True).set_title('Correlation Heatmap')
```

---

## 💡 How This Project Helps

| Stakeholder                | Benefit                                                              |
| -------------------------- | -------------------------------------------------------------------- |
| **Financial Institutions** | Predict loan approvals with confidence                               |
| **Loan Officers**          | Understand customer behavior through visual insights                 |
| **Product Teams**          | Tailor offerings to fit specific age, income, and education segments |
| **Data Scientists**        | Learn to integrate dashboarding and ML for business-ready solutions  |

---

## 🚀 Future Enhancements

* Hyperparameter tuning using `GridSearchCV`
* Try alternate models like `Logistic Regression` or `XGBoost`
* Add domain-specific features (e.g., CIBIL score, dependents)
* Deploy as a REST API or via **Streamlit** Web App
* Link dashboard with live model predictions

---

## 📁 Project Structure

```
Loan-Prediction-Project/
│
├── 📊 loan_dataset.xlsx
├── 📊 loan_dataset26.xlsx
├── 🧠 loan_prediction_analysis.py
├── 🖼️ Loan Prediction Dashboard.png
└── 📄 README.md
```

---

## 👩‍💻 Author

**Khushboo Verma**
*Machine Learning & Data Analytics Enthusiast*
