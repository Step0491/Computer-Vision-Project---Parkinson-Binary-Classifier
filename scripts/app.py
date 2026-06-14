import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

st.set_page_config(page_title="Parkinson's Classification Dashboard", layout="wide")

st.title("Parkinson's Disease Classification Dashboard")
st.markdown("""
This dashboard presents the baseline classification model distinguishing **Parkinson's disease (PD)** patients from **Healthy (H)** subjects using statistical features extracted from the PaHaW handwriting dataset.
""")

DATA_FILE = 'dataset_features.csv'

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)
    # The true test set for the challenge has NaN labels, we only use the labeled data here
    df_labeled = df[df['Label'].notna()]
    return df_labeled

df = load_data()

st.header("1. Dataset Overview")
st.write("Below is a sample of the extracted statistical features from the handwriting time-series:")
st.dataframe(df.head(10))

# Separate features and labels
X = df.drop(columns=['Subject_ID', 'Label'])
y = df['Label']
subject_ids = df['Subject_ID']

# 80-20 train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Model with optimized hyperparameters (regularization to prevent overfitting)
model = RandomForestClassifier(
    n_estimators=200, 
    max_depth=7, 
    min_samples_split=5, 
    min_samples_leaf=2, 
    max_features='sqrt',
    random_state=42, 
    class_weight='balanced'
)
model.fit(X_train_scaled, y_train)

# Predict
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# Metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
auc = roc_auc_score(y_test, y_prob)

st.header("2. Baseline Model Performance")
st.write("Hold-out Test Set (20%) Evaluation using a Random Forest Classifier:")

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Accuracy", f"{acc*100:.1f}%")
col2.metric("Precision", f"{prec*100:.1f}%")
col3.metric("Recall", f"{rec*100:.1f}%")
col4.metric("Specificity", f"{spec*100:.1f}%")
col5.metric("F1-Score", f"{f1*100:.1f}%")
col6.metric("AUC", f"{auc:.2f}")

st.header("3. Visualizations")

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Feature Importances")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_n = 10
    top_indices = indices[:top_n]
    top_features = X.columns[top_indices]
    top_importances = importances[top_indices]
    
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    sns.barplot(x=top_importances, y=top_features, ax=ax1, palette="viridis")
    ax1.set_xlabel("Relative Importance")
    st.pyplot(fig1)

with col_b:
    st.subheader("Confusion Matrix")
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax2, cbar=False, 
                xticklabels=['Healthy (0)', 'PD (1)'], yticklabels=['Healthy (0)', 'PD (1)'])
    ax2.set_ylabel("True Label")
    ax2.set_xlabel("Predicted Label")
    st.pyplot(fig2)

col_c, col_d = st.columns(2)

with col_c:
    st.subheader("ROC Curve")
    fig3, ax3 = plt.subplots(figsize=(6, 5))
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    ax3.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
    ax3.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax3.set_xlim([0.0, 1.0])
    ax3.set_ylim([0.0, 1.05])
    ax3.set_xlabel('False Positive Rate')
    ax3.set_ylabel('True Positive Rate')
    ax3.legend(loc="lower right")
    st.pyplot(fig3)
