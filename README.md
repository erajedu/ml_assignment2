# ml_assignment2
# Telco Customer Churn – Machine Learning Assignment 2

## Project Objective  
This assignment focuses on predicting customer churn for a telecom company using multiple machine learning classification models. The objective is to compare different models and select the best performing one based on multiple evaluation metrics.

---

## About the Dataset used in Project
**Dataset Name:** Telco Customer Churn  
**Source:** Kaggle  
**Records:** 7,043 customers  
**Features:** 21 columns  
**Target Variable:** `Churn (Yes/No)`
**Link:** https://www.kaggle.com/datasets/blastchar/telco-customer-churn

### Feature Types
**•	Categorical features:** gender, contract, payment method, internet service
**•	Binary features:** partner, dependents, paperless billing, senior citizen
**•	Numerical features:** tenure, monthly charges, total charges


**Download Dataset:**  
https://www.kaggle.com/datasets/blastchar/telco-customer-churn  

### Why this dataset is suitable for Machine Learning
This dataset is highly suitable for machine learning, particularly for customer churn prediction, because it provides a rich and diverse set of features that capture multiple dimensions of customer behaviour.
•	It includes behavioral data, such as tenure, service usage patterns, and subscription choices, which help the model understand how long customers stay with the company and how actively they use different services.
•	It contains detailed billing information, including monthly charges, total charges, contract type, and payment method. These financial attributes are strong indicators of customer satisfaction and likelihood of churn, making them valuable predictors for classification models.
•	The dataset captures service usage patterns, such as phone service, internet service, online security, streaming services, and technical support. These features allow machine learning models to learn relationships between specific services and customer retention or churn.
•	It includes important demographic attributes such as gender, senior citizen status, partner, and dependents. These variables help analyze how customer characteristics influence churn behavior.

Because the dataset combines behavioral, financial, service-related, and demographic information in a structured format, it is well-balanced, interpretable, and highly appropriate for supervised learning tasks such as binary classification, model comparison, and customer retention analysis.

---

## Machine Learning Models Implemented  
The following models were trained and compared:

- Logistic Regression  
- Decision Tree  
- K-Nearest Neighbors (KNN)  
- Naive Bayes  
- Random Forest  
- XGBoost  

---

## Evaluation Metrics Used  
Model performance was compared using:

- Accuracy  
- AUC Score  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)  

---

## Model Comparison Results  

| Sr.No | Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---|---|---|---|---|---|---|
| 1 | Logistic Regression | 0.811236 | 0.860905 | 0.678322 | 0.548023 | 0.606250 | 0.488710 |
| 2 | Decision Tree | 0.743820 | 0.670858 | 0.517143 | 0.511299 | 0.514205 | 0.340265 |
| 3 | KNN | 0.753558 | 0.775209 | 0.538941 | 0.488701 | 0.512593 | 0.348954 |
| 4 | Naive Bayes | 0.765543 | 0.838355 | 0.542797 | 0.734463 | 0.624250 | 0.470480 |
| 5 | Random Forest | 0.799251 | 0.836067 | 0.666667 | 0.485876 | 0.562092 | 0.445175 |
| 6 | XGBoost | 0.800000 | 0.845946 | 0.649485 | 0.533898 | 0.586047 | 0.459653 |

---

## Observations on Model Performance  

| Sr.No | ML Model | Observation on Performance |
|---|---|---|
| 1 | Logistic Regression | Showed the best overall balance between accuracy, AUC, F1-score, and MCC; stable, interpretable, and generalized well. |
| 2 | Decision Tree | Performed moderately but showed signs of overfitting and weaker generalization. |
| 3 | KNN | Sensitive to feature scaling and distance metric; delivered moderate but not optimal performance. |
| 4 | Naive Bayes | Achieved high recall but lower precision, indicating a tendency to over-predict churn cases. |
| 5 | Random Forest | Provided good accuracy and robustness but slightly weaker than Logistic Regression on MCC. |
| 6 | XGBoost | Captured complex patterns well but did not significantly outperform simpler models like Logistic Regression. |

---

## Best Selected Model  

The final model was selected based on a comprehensive comparison of all evaluation metrics including Accuracy, AUC, Precision, Recall, F1-score, and Matthews Correlation Coefficient (MCC).
**Logistic Regression** was selected as the final model because it achieved the best overall balance across metrics.

- Highest overall balanced performance  
- Strong AUC (**0.86**)  
- Good F1-score (**0.60**)  
- Robust MCC (**0.48**)  
- Better interpretability and stability  

Compared to other models, Logistic Regression demonstrated consistent performance without overfitting and provided better interpretability, making it the most reliable choice for this churn prediction task.

---

## Streamlit Web Application  
Streamlit Community Cloud Link
https://mlassignment2-lzabkonvtg9wztq38bpug6.streamlit.app/

---

**Live App:**  
https://mlassignment2-lzabkonvtg9wztq38bpug6.streamlit.app/

---

## 📁 Repository Structure  
ml_assignment2/
│
├── app.py
├── requirements.txt
├── sample_input_template.csv
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── ML_Assignment2_1.ipynb
│
└── model/
├── models.pkl
├── scaler.pkl
├── label_encoders.pkl
├── feature_names.pkl
└── model_results.csv


