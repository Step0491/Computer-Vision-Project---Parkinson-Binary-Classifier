import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

DATA_FILE = 'dataset_features.csv'
SUBMISSION_FILE = 'submission.csv'

def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0

def train_and_evaluate():
    print("Loading data...")
    df = pd.read_csv(DATA_FILE)
    
    # Separate features and labels
    # Subject_ID is column 0, Label is column 1
    X = df.drop(columns=['Subject_ID', 'Label'])
    y = df['Label']
    subject_ids = df['Subject_ID']
    
    print(f"Dataset shape: {X.shape}")
    
    # Let's do an 80-20 train-test split
    # Stratified to ensure equal proportion of PD/Healthy in train and test
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, subject_ids, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Model: Random Forest with optimized regularization to prevent overfitting and improve precision
    model = RandomForestClassifier(
        n_estimators=200, 
        max_depth=7, 
        min_samples_split=5, 
        min_samples_leaf=2, 
        max_features='sqrt',
        random_state=42, 
        class_weight='balanced'
    )
    
    # 1. Evaluate using Stratified 5-Fold Cross Validation on the training set
    print("\n--- 5-Fold Cross Validation on Training Set ---")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(
        model, X_train_scaled, y_train, cv=cv,
        scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    )
    
    print(f"CV Accuracy:  {np.mean(cv_results['test_accuracy']):.4f} ± {np.std(cv_results['test_accuracy']):.4f}")
    print(f"CV Precision: {np.mean(cv_results['test_precision']):.4f} ± {np.std(cv_results['test_precision']):.4f}")
    print(f"CV Recall:    {np.mean(cv_results['test_recall']):.4f} ± {np.std(cv_results['test_recall']):.4f}")
    print(f"CV F1-Score:  {np.mean(cv_results['test_f1']):.4f} ± {np.std(cv_results['test_f1']):.4f}")
    print(f"CV AUC:       {np.mean(cv_results['test_roc_auc']):.4f} ± {np.std(cv_results['test_roc_auc']):.4f}")
    
    # 2. Train on the full training set and evaluate on the hold-out test set
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    spec = specificity_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    print("\n--- Hold-Out Test Set Evaluation ---")
    print(f"Accuracy:    {acc:.4f}")
    print(f"Precision:   {prec:.4f}")
    print(f"Recall:      {rec:.4f}")
    print(f"Specificity: {spec:.4f}")
    print(f"F1-Score:    {f1:.4f}")
    print(f"AUC:         {auc:.4f}")
    
    # 3. Generate Submission CSV for the test set
    submission_df = pd.DataFrame({
        'ID': ids_test.astype(str).str.zfill(5),
        'PD status': y_pred.astype(int)
    })
    submission_df.to_csv(SUBMISSION_FILE, index=False)
    print(f"\nSubmission file saved to '{SUBMISSION_FILE}'.")
    
    # Optional: print feature importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    print("\nTop 5 Important Features:")
    for i in range(5):
        print(f"{i+1}. {X.columns[indices[i]]} ({importances[indices[i]]:.4f})")

if __name__ == '__main__':
    train_and_evaluate()
