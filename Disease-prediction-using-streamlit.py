import streamlit as st
import pandas as pd
import numpy as np
import psycopg2
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import hashlib

# Database connection
def create_connection():
    try:
        conn = psycopg2.connect("postgresql://postgres:1234@localhost:5432/Telemedicine")
        return conn
    except Exception as e:
        st.error(f"Error connecting to database: {e}")
        return None

# User authentication functions
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def login_user(username, password):
    conn = create_connection()
    if conn:
        cur = conn.cursor()
        hashed_password = hash_password(password)
        cur.execute("SELECT * FROM login WHERE username=%s AND pswd=%s", (username, hashed_password))
        result = cur.fetchone()
        conn.close()
        return result
    return None

def create_user(username, password):
    conn = create_connection()
    if conn:
        cur = conn.cursor()
        hashed_password = hash_password(password)
        cur.execute("INSERT INTO login (username, pswd) VALUES (%s, %s)", (username, hashed_password))
        conn.commit()
        conn.close()

def reset_password(username, new_password):
    conn = create_connection()
    if conn:
        cur = conn.cursor()
        hashed_password = hash_password(new_password)
        cur.execute("UPDATE login SET pswd=%s WHERE username=%s", (hashed_password, username))
        conn.commit()
        conn.close()

# User authentication UI
def user_authentication():
    st.sidebar.title("User Authentication")
    auth_mode = st.sidebar.radio("Choose Action", ["Login", "Sign Up", "Forgot Password"])

    if auth_mode == "Login":
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")
        if st.sidebar.button("Login"):
            if login_user(username, password):
                st.session_state['logged_in'] = True
                st.session_state['username'] = username
                st.sidebar.success("Logged in successfully!")
            else:
                st.sidebar.error("Invalid username or password.")
    elif auth_mode == "Sign Up":
        username = st.sidebar.text_input("New Username")
        password = st.sidebar.text_input("New Password", type="password")
        if st.sidebar.button("Sign Up"):
            create_user(username, password)
            st.sidebar.success("User created successfully! Please login.")
    elif auth_mode == "Forgot Password":
        username = st.sidebar.text_input("Username")
        new_password = st.sidebar.text_input("New Password", type="password")
        if st.sidebar.button("Reset Password"):
            reset_password(username, new_password)
            st.sidebar.success("Password reset successfully! Please login.")

# Load datasets
@st.cache_data
def load_datasets():
    conn = create_connection()
    if conn is None:
        return None, None, None, None
    try:
        anemia_df = pd.read_sql("SELECT * FROM anemia_saved", conn)
        ckd_df = pd.read_sql("SELECT * FROM chronic_kidney_disease_saved", conn)
        diabetes_df = pd.read_sql("SELECT * FROM diabetes_saved", conn)
        stroke_df = pd.read_sql("SELECT * FROM stroke_prediction_saved", conn)
        return anemia_df, ckd_df, diabetes_df, stroke_df
    except Exception as e:
        st.error(f"Error loading datasets: {e}")
        return None, None, None, None
    finally:
        conn.close()

# Define classifiers
def get_classifiers():
    svm = SVC(probability=True, kernel='rbf', gamma='scale')
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    lr = LogisticRegression(max_iter=100, solver='lbfgs', random_state=42)
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    return svm, rf, lr, xgb

# Parallel prediction function for each classifier
def predict_model(model, X_train, X_test, y_train):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    return y_pred, y_pred_proba

# Train and evaluate hybrid classifier
def train_and_predict_ensemble_parallel(X_train, X_test, y_train, y_test, models):
    with ThreadPoolExecutor() as executor:
        results = executor.map(lambda model: predict_model(model, X_train, X_test, y_train), models)

    y_preds, y_pred_probas = [], []
    for y_pred, y_pred_proba in results:
        y_preds.append(y_pred)
        y_pred_probas.append(y_pred_proba)

    avg_pred_proba = np.mean(y_pred_probas, axis=0)
    avg_pred = (avg_pred_proba >= 0.5).astype(int)

    accuracy = accuracy_score(y_test, avg_pred)
    roc_auc = roc_auc_score(y_test, avg_pred_proba)
    precision = precision_score(y_test, avg_pred)
    recall = recall_score(y_test, avg_pred)
    f1 = f1_score(y_test, avg_pred)

    return accuracy, roc_auc, precision, recall, f1, avg_pred, avg_pred_proba

# Plot ROC curve
def plot_roc_curve(y_test, y_pred_proba):
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    st.pyplot(plt)

# Function to plot correlation matrix
def plot_correlation_matrix(df):
    correlation = df.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation, annot=True, fmt=".2f", cmap='coolwarm', center=0)
    plt.title("Correlation Matrix Heatmap")
    st.pyplot(plt)

# Feature importance plot
def plot_feature_importance(df, feature_cols, target_col):
    X = df[feature_cols]
    y = df[target_col]
    model = ExtraTreesClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    feature_importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': feature_cols, 'Importance': feature_importances}).sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
    plt.title("Feature Importance using ExtraTreesClassifier")
    st.pyplot(plt)

# Categorical encoding
def encode_features(df):
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df

# Impute missing values
def impute_missing_values(df):
    imputer = SimpleImputer(strategy='mean')
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    df[numerical_cols] = imputer.fit_transform(df[numerical_cols])
    return df

# Disease prediction
def disease_prediction_with_extras(disease_name, feature_cols, target_col, df, categorical_cols=None, dropdown_options=None):
    input_values = {}
    for feature in feature_cols:
        if categorical_cols and feature in categorical_cols:
            options = dropdown_options.get(feature, []) if dropdown_options else []
            input_values[feature] = st.selectbox(f"{feature}", options)
        else:
            input_values[feature] = st.number_input(f"{feature}", value=0.0)

    if st.button(f"Predict {disease_name}"):
        try:
            df_encoded = encode_features(df)
            df_imputed = impute_missing_values(df_encoded)
            X = df_imputed[feature_cols]
            y = df_imputed[target_col]
            if len(y.unique()) < 2:
                st.error("Insufficient classes in the training data.")
                return
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            models = get_classifiers()
            accuracy, roc_auc, precision, recall, f1, _, avg_pred_proba = train_and_predict_ensemble_parallel(X_train, X_test, y_train, y_test, models)
            
            user_input = pd.DataFrame([input_values], columns=feature_cols)
            user_input_encoded = encode_features(user_input)
            user_input_imputed = impute_missing_values(user_input_encoded)
            user_prediction = (np.mean([model.predict_proba(user_input_imputed)[:, 1] for model in models], axis=0) >= 0.5).astype(int)
            
            st.success(f"Prediction Result: {'Positive' if user_prediction else 'Negative'}")
            st.write(f"Accuracy: {accuracy*100:.2f}%, ROC AUC: {roc_auc*100:.2f}%, Precision: {precision*100:.2f}%, Recall: {recall*100:.2f}%, F1 Score: {f1*100:.2f}%")
            plot_roc_curve(y_test, avg_pred_proba)
            plot_correlation_matrix(df_imputed)
            plot_feature_importance(df_imputed, feature_cols, target_col)
        except Exception as e:
            st.error(f"Prediction failed due to error: {e}")

# Main app function
def main():
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    if not st.session_state['logged_in']:
        user_authentication()
    else:
        st.sidebar.title("Welcome")
        if st.sidebar.button("Logout"):
            st.session_state['logged_in'] = False
            st.session_state['username'] = ""
            st.sidebar.success("Logged out successfully!")
            return

        st.sidebar.title("Multiple Disease Prediction System")
        disease_option = st.sidebar.radio("Choose Disease for Prediction", ["Anemia Prediction", "Chronic Kidney Disease Prediction", "Diabetes Prediction", "Stroke Prediction"])
        anemia_df, ckd_df, diabetes_df, stroke_df = load_datasets()
        
        if disease_option == "Anemia Prediction":
            feature_cols = anemia_df.columns[:-1]
            disease_prediction_with_extras("Anemia", feature_cols, "Result", anemia_df)
        
        elif disease_option == "Chronic Kidney Disease Prediction":
            feature_cols = ckd_df.columns[:-1]
            categorical_cols = ["rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"]
            dropdown_options = {
                "rbc": ["normal", "abnormal", "null"],
                "pc": ["normal", "abnormal", "null"],
                "pcc": ["present", "notpresent"],
                "ba": ["present", "notpresent"],
                "htn": ["yes", "no"],
                "dm": ["yes", "no"],
                "cad": ["yes", "no"],
                "appet": ["good", "poor"],
                "pe": ["yes", "no"],
                "ane": ["yes", "no"]
            }
            disease_prediction_with_extras("Chronic Kidney Disease", feature_cols, "class", ckd_df, categorical_cols, dropdown_options)

        elif disease_option == "Diabetes Prediction":
            feature_cols = diabetes_df.columns[:-1]
            disease_prediction_with_extras("Diabetes", feature_cols, "Outcome", diabetes_df)

        elif disease_option == "Stroke Prediction":
            feature_cols = stroke_df.columns[:-1]
            categorical_cols = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
            dropdown_options = {
                "gender": ["Male", "Female"],
                "ever_married": ["yes", "no"],
                "work_type": ["private", "Govt_job", "self-employed"],
                "Residence_type": ["Rural", "Urban"],
                "smoking_status": ["formerly smoked", "never smoked", "smokes", "Unknown"]
            }
            disease_prediction_with_extras("Stroke", feature_cols, "stroke", stroke_df, categorical_cols, dropdown_options)

        st.write("Note: Please choose a disease option from the sidebar to begin prediction.")

# Run the main app
if __name__ == "__main__":
    main()
