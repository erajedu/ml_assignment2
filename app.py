import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Telco Customer Churn - Machine Learning Assignment 2",
    layout="wide"
)

st.title("Telco Customer Churn - Machine Learning Assignment 2")
st.markdown(
    "<p style='font-size:22px; color:gray; font-weight:bold; text-align:left;'>"
    "Student Name: Rajesh Dubey | Student ID: 2025AA05373 | Email: 2025aa05373@wilp.bits-pilani.ac.in"
    "</p>",
    unsafe_allow_html=True
)
st.markdown(
    "<hr style='border: 3px solid #1F4E79; margin-top:5px; margin-bottom:10px;'>",
    unsafe_allow_html=True
)
# --------------------------------------------------
# Load saved objects
# --------------------------------------------------
models = joblib.load("model/models.pkl")
scaler = joblib.load("model/scaler.pkl")
label_encoders = joblib.load("model/label_encoders.pkl")
feature_names = joblib.load("model/feature_names.pkl")
results = pd.read_csv("model/model_results.csv", index_col=0)

BEST_MODEL = "Logistic Regression"

# --------------------------------------------------
# Sidebar navigation
# --------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "ML Prediction", "Dataset Info"]
)

# ==================================================
# PAGE 1: OVERVIEW
# ==================================================
if page == "Overview":

    st.subheader("Project Overview")
    st.write("""
    This assignment focuses on predicting customer churn for a telecom company
    using multiple machine learning classification models. The objective is to
    compare different models and select the best performing one based on
    multiple evaluation metrics.
    """)

    st.subheader("Models Implemented")
    st.write(""" The following six models need to be tested on the dataset to compare their performance and accuracy.
    - Logistic Regression  
    - Decision Tree  
    - K-Nearest Neighbors (KNN)  
    - Naive Bayes  
    - Random Forest  
    - XGBoost  
    """)

    st.subheader("Evaluation Metrics Used")
    st.write(""" Below are the evaluation criteria for best model selection
    - Accuracy  
    - AUC  
    - Precision  
    - Recall  
    - F1 Score  
    - Matthews Correlation Coefficient (MCC)  
    """)

    st.subheader("Model Comparison Results")

    # --- Step 1: convert index to a real column ---
    comparison_df = results.reset_index()
    comparison_df.rename(columns={"index": "Model"}, inplace=True)

    # --- Step 2: add Sr.No correctly ---
    comparison_df.insert(0, "Sr.No", range(1, len(comparison_df) + 1))

    # --- Step 3: style safely ---
    styled_comp = (
        comparison_df.style
        .hide(axis="index")                      # remove Streamlit's auto index
        .set_properties(subset=["Model"], **{"text-align": "left"})
        .set_properties(
            subset=[c for c in comparison_df.columns if c not in ["Model"]],
            **{"text-align": "center"}
        )
        .set_table_styles([
            {
                "selector": "th",
                "props": [
                    ("background-color", "#D6EAF8"),
                    ("color", "black"),
                    ("font-weight", "bold"),
                    ("border", "1px solid black"),
                    ("text-align", "center")
                ]
            },
            {
                "selector": "td",
                "props": [
                    ("border", "1px solid black")
                ]
            }
        ])
    )

    st.write(styled_comp.to_html(), unsafe_allow_html=True)


    # --------------------------------------------------
    # Observations on Model Performance (Styled Table)
    # --------------------------------------------------
    st.subheader("Observations on Model Performance")

    observations = pd.DataFrame({
        "ML Model": [
            "Logistic Regression",
            "Decision Tree",
            "KNN",
            "Naive Bayes",
            "Random Forest",
            "XGBoost"
        ],
        "Observation on Performance": [
            "Logistic Regression model is showing the best overall balance between accuracy, AUC, F1-score, and MCC with stable and interpretable with good generalization.",
            "Decision Tree, Performed moderately but showed signs of overfitting and lower generalization compared to other models.",
            "KNN Model performance was sensitive to feature scaling and distance metric with moderate results but not optimal.",
            "This model achieved high recall but lower precision, indicating a tendency to over-predict churn cases.",
            "This model provided good accuracy and robustness but slightly weaker performance than Logistic Regression on MCC.",
            "XGBoost model captured complex patterns well but did not significantly outperform simpler models like Logistic Regression."
        ]
    })

    # Add correct Sr.No
    observations.insert(0, "Sr.No", range(1, len(observations) + 1))

    styled_obs = (
        observations.style
        .hide(axis="index")   # <-- KEY FIX
        .set_properties(subset=["ML Model"], **{"text-align": "left"})
        .set_properties(
            subset=["Sr.No", "Observation on Performance"],
            **{"text-align": "center"}
        )
        .set_table_styles([
            {
                "selector": "th",
                "props": [
                    ("background-color", "#D6EAF8"),
                    ("color", "black"),
                    ("font-weight", "bold"),
                    ("border", "1px solid black"),
                    ("text-align", "center")
                ]
            },
            {
                "selector": "td",
                "props": [
                    ("border", "1px solid black")
                ]
            }
        ])
    )

    st.write(styled_obs.to_html(), unsafe_allow_html=True)


    # --------------------------------------------------
    # Confusion Matrix – Model Evaluation
    # --------------------------------------------------
    st.subheader("Confusion Matrix – Model Evaluation")

    st.write("""
    The confusion matrix below is generated during the **model evaluation phase**.
    After training, each model is evaluated on the unseen test dataset and the
    predicted labels are compared with the actual churn labels.
    """)

    cm_model = st.selectbox(
        "Select Model for Confusion Matrix",
        list(models.keys()),
        index=list(models.keys()).index(BEST_MODEL)
    )

    # Load original dataset
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

    # Preprocessing (same as training)
    df.drop("customerID", axis=1, inplace=True)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    for col, le in label_encoders.items():
        df[col] = le.transform(df[col])

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    if cm_model in ["Logistic Regression", "KNN", "Naive Bayes"]:
        X_test_scaled = scaler.transform(X_test)
        y_pred = models[cm_model].predict(X_test_scaled)
    else:
        y_pred = models[cm_model].predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(4, 3), dpi=80)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix – {cm_model}")

    st.pyplot(fig, use_container_width=False)

    accuracy_cm = (cm[0, 0] + cm[1, 1]) / cm.sum()
    st.caption(f"Accuracy from Confusion Matrix: {accuracy_cm:.4f}")

    # --------------------------------------------------
    # Final Model Selection & Justification
    # --------------------------------------------------
    st.subheader("Final Model Selection & Justification")
    st.success("Selected Best Model: Logistic Regression")

    st.write("""
    The final model was selected based on a comprehensive comparison of all
    evaluation metrics including Accuracy, AUC, Precision, Recall, F1-score,
    and Matthews Correlation Coefficient (MCC).

    **Logistic Regression** achieved the best overall balance across metrics:
    - **AUC:** 0.8609 – strong class separation capability  
    - **F1-score:** 0.6062 – good balance between precision and recall  
    - **MCC:** 0.4887 – robust performance on imbalanced data  

    Compared to other models, Logistic Regression demonstrated consistent
    performance without overfitting and provided better interpretability,
    making it the most reliable choice for this churn prediction task.
    """)

# ==================================================
# ==================================================
# PAGE 2: ML PREDICTION
# ==================================================
elif page == "ML Prediction":

    st.subheader("Machine Learning Prediction")

    # ------------------ MODEL SELECTION ------------------
    model_name = st.selectbox(
        "Select Model - Predcition will be genrated for selected model",
        list(models.keys()),
        index=list(models.keys()).index(BEST_MODEL)
    )

    # ------------------ SAMPLE FILE DOWNLOAD ------------------
    # ------------------ SAMPLE FILE DOWNLOAD ------------------
    st.subheader("Download Test Data / Sample Input File")
    st.write(
        "Download the Test Data or Sample file format for download - Input data need to be provided in same format with their respective column values."
    )    

    with open("sample_input_template.csv", "rb") as f:
        st.download_button(
            label="Download Sample CSV Template",
            data=f,
            file_name="sample_input_template.csv",
            mime="text/csv"
        )




    # ------------------ FILE UPLOAD ------------------
    st.subheader("Upload Data for ML Predcition")
    uploaded_file = st.file_uploader(
        "Upload the file to genratet the prediction, no Churn required)",
        type=["csv"]
    )

    if uploaded_file is not None:

        df_test = pd.read_csv(uploaded_file)

        # ----------- INPUT PREVIEW (STYLED) -----------
        st.subheader("Input Data Preview (First 5 Records)")

        styled_input = (
            df_test.head()
            .style
            .hide(axis="index")
            .set_table_styles([
                {
                    "selector": "th",
                    "props": [
                        ("background-color", "#D6EAF8"),
                        ("color", "black"),
                        ("font-weight", "bold"),
                        ("border", "1px solid black"),
                        ("text-align", "center")
                    ]
                },
                {
                    "selector": "td",
                    "props": [("border", "1px solid black")]
                }
            ])
        )

        st.write(styled_input.to_html(), unsafe_allow_html=True)

        # ------------------ PREPROCESSING ------------------
        df_features = df_test.drop(
            columns=["customerID", "Churn"], errors="ignore"
        )
        df_features = df_features[feature_names]

        for col, le in label_encoders.items():
            if col in df_features.columns:
                df_features[col] = le.transform(df_features[col])

        X_scaled = scaler.transform(df_features)

        X_input = (
            X_scaled
            if model_name in ["Logistic Regression", "KNN", "Naive Bayes"]
            else df_features
        )

        # ------------------ PREDICTION ------------------
        predictions = models[model_name].predict(X_input)

        output_df = pd.DataFrame({
            "Record No": [f"Customer {i}" for i in range(1, len(predictions) + 1)],
            "Predicted Churn": predictions
        })

        # Convert 0/1 to Yes/No
        output_df["Predicted Churn"] = output_df["Predicted Churn"].map({
            0: "No",
            1: "Yes"
        })

        output_df.insert(0, "Sr.No", range(1, len(output_df) + 1))

        # ----------- PREDICTION OUTPUT (STYLED) -----------
        st.subheader("Prediction Output")

        styled_pred = (
            output_df.style
            .hide(axis="index")
            .set_properties(**{"text-align": "center"})
            .set_table_styles([
                {
                    "selector": "th",
                    "props": [
                        ("background-color", "#D6EAF8"),
                        ("color", "black"),
                        ("font-weight", "bold"),
                        ("border", "1px solid black"),
                        ("text-align", "center")
                    ]
                },
                {
                    "selector": "td",
                    "props": [("border", "1px solid black")]
                }
            ])
        )

        st.write(styled_pred.to_html(), unsafe_allow_html=True)

        # ----------- DOWNLOAD PREDICTIONS -----------
        st.download_button(
            "Download Predictions",
            output_df.to_csv(index=False),
            "predictions.csv",
            "text/csv"
        )

        # ----------- METRICS TABLE (STYLED) -----------
        st.subheader("Model Performance Metrics")

        metrics_row = results.loc[model_name].to_frame().T
        metrics_row.insert(0, "Model", model_name)

        styled_metrics = (
            metrics_row.style
            .hide(axis="index")
            .set_properties(**{"text-align": "center"})
            .set_table_styles([
                {
                    "selector": "th",
                    "props": [
                        ("background-color", "#D6EAF8"),
                        ("color", "black"),
                        ("font-weight", "bold"),
                        ("border", "1px solid black"),
                        ("text-align", "center")
                    ]
                },
                {
                    "selector": "td",
                    "props": [("border", "1px solid black")]
                }
            ])
        )

        st.write(styled_metrics.to_html(), unsafe_allow_html=True)


# ==================================================
# PAGE 3: DATASET INFO
# ==================================================
elif page == "Dataset Info":

    st.subheader("📂 About the Dataset")

    st.markdown("""
    **Dataset Name:** Telco Customer Churn  
    **Source:** Kaggle  
    **Link:** https://www.kaggle.com/datasets/blastchar/telco-customer-churn  
    """)

    st.subheader("⬇️ Download Dataset File")

    with open("WA_Fn-UseC_-Telco-Customer-Churn.csv", "rb") as f:
        st.download_button(
            label="Download Telco Churn Dataset",
            data=f,
            file_name="WA_Fn-UseC_-Telco-Customer-Churn.csv",
            mime="text/csv"
        )
        

    st.subheader("📊 Dataset Overview")
    st.write("""
    - **Total rows (customers):** 7,043  
    - **Total columns (features):** 21  
    - **Target variable:** `Churn` (Yes/No)  
    - Each row represents one customer  
    - Each column represents a customer attribute  
    """)

    st.subheader("🧾 Context")

    st.write("""
    **1) Customer Status**
    - Customers who left within the last month — stored in the column **Churn**

    **2) Services Signed Up**
    - Phone service  
    - Multiple lines  
    - Internet service  
    - Online security  
    - Online backup  
    - Device protection  
    - Tech support  
    - Streaming TV  
    - Streaming movies  

    **3) Customer Account Information**
    - Tenure (how long they’ve been a customer)  
    - Contract type  
    - Payment method  
    - Paperless billing  
    - Monthly charges  
    - Total charges  

    **4) Demographics**
    - Gender  
    - Senior citizen (0/1)  
    - Partner (Yes/No)  
    - Dependents (Yes/No)
    """)

    st.subheader("📑 Feature Types")

    st.write("""
    - **Categorical features:** gender, contract, payment method, internet service  
    - **Binary features:** partner, dependents, paperless billing, senior citizen  
    - **Numerical features:** tenure, monthly charges, total charges  
    """)

    # ---------------- COLUMN SUMMARY (NEW ADDITION) ----------------
    st.subheader("📋 Column Summary & Sample Data")

    try:
        df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

        st.write(f"**Total Rows:** {df.shape[0]} | **Total Columns:** {df.shape[1]}")

        st.write("### First 5 rows of the dataset")
        st.dataframe(df.head())

        st.write("### Column list with data types")
        col_info = pd.DataFrame({
            "Column Name": df.columns,
            "Data Type": df.dtypes.astype(str)
        })
        st.dataframe(col_info)

    except Exception as e:
        st.error(f"Could not load dataset for preview: {e}")

    # ---------------------------------------------------------------

    st.subheader("🎯 Why this dataset is suitable for Machine Learning")

    st.write("""
    This dataset is highly suitable for machine learning, particularly for customer churn
    prediction, because it provides a rich and diverse set of features that capture
    multiple dimensions of customer behavior.

   - It includes **behavioral data**, such as tenure, service usage patterns,
    and subscription choices, which help the model understand how long customers stay
    with the company and how actively they use different services.

    - It contains detailed **billing information**, including monthly charges,
    total charges, contract type, and payment method. These financial attributes are
    strong indicators of customer satisfaction and likelihood of churn, making them
    valuable predictors for classification models.

    - The dataset captures **service usage patterns**, such as phone service,
    internet service, online security, streaming services, and technical support.
    These features allow machine learning models to learn relationships between
    specific services and customer retention or churn.

    - It includes important **demographic attributes** such as gender,
    senior citizen status, partner, and dependents. These variables help analyze how
    customer characteristics influence churn behavior.

    Because the dataset combines behavioral, financial, service-related, and
    demographic information in a structured format, it is well-balanced, interpretable,
    and highly appropriate for supervised learning tasks such as binary classification,
    model comparison, and customer retention analysis.
    """)



# --------------------------------------------------
# Footer (centered)
# --------------------------------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center;'>Assignment-2 Machine Learning | Student ID : 2025AA05373 | Student Name: Rajesh Dubey</p>",
    unsafe_allow_html=True
)

