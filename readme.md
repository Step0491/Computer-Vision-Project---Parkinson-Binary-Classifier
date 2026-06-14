# Parkinson's Classification Project Report

## How to Start the Project
To run this project and reproduce the results, the main entry point is the Jupyter Notebook.
1. Activate your virtual environment (e.g., `venv\Scripts\activate` on Windows).
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Open and run the `main.ipynb` notebook from top to bottom. It will sequentially execute data processing, model training, and generate the evaluation visualizations.
4. Alternatively, you can run the individual python scripts (`scripts/prepare_data.py` and `scripts/train_model.py`) from your terminal, or launch the interactive dashboard using `streamlit run scripts/app.py`.

### Classes Explanation
This is a binary classification problem focusing on detecting Parkinson's disease from handwriting time-series data. The model predicts the `Disease` status mapped as follows:
- **Class `1` (PD)**: Represents patients diagnosed with Parkinson's Disease.
- **Class `0` (H)**: Represents Healthy control subjects.

---

## 1. Introduction
This project aims to build a binary classification model to distinguish between Parkinson's disease (PD) patients and healthy control (H) subjects using the **PaHaW Handwriting Dataset**. The dataset consists of time-series data captured from a digitizing tablet, measuring variables like X and Y coordinates, pen pressure, altitude, azimuth, and button state (indicating whether the pen touches the surface).

## 2. Project Architecture and Scripts

### 2.1. `prepare_data.py` (Data Extraction and Feature Engineering)
This script handles the raw dataset and transforms it into a machine-learning-ready tabular format (`dataset_features.csv`).
- **Label Mapping**: It parses `corpus_PaHaW.xlsx` to extract labels. We identified that the `Disease` column accurately separates subjects into 'PD' (labeled `1`) and 'H' (labeled `0`).
- **Exclusions**: It safely removes subjects `00061`, `00080`, and `00089` from the pipeline as requested in the project guidelines.
- **Feature Engineering**: For each patient's handwriting task files (`.svc`), it extracts:
  - **Statistical metrics**: The mean, standard deviation, max, min, and median for physical properties (X, Y, Altitude, Azimuth, Pressure).
  - **Temporal features**: Total duration of the writing task.
  - **Dynamic features**: Re-calculates continuous velocity by dividing spatial displacement (derived from X and Y coordinates) by time deltas. It then computes velocity mean, standard deviation, and maximum.
  - **Behavioral markers**: Calculates the proportion of time the pen is pressed (`Button_mean`) and counts the instances of pen lifts.
- **Aggregation**: It averages these features across all handwriting tasks per subject, resulting in a single high-dimensional feature vector per person.

### 2.2. `train_model.py` (Model Training and Evaluation)
This script is responsible for the predictive modeling.
- **Validation Strategy**: Since the dataset is relatively small (72 subjects), it employs an 80/20 Train-Test split. For robust model validation, it applies **Stratified 5-Fold Cross Validation** on the 80% training set to ensure generalization before predicting on the hold-out set.
- **Modeling**: It uses a `RandomForestClassifier` with optimized hyper-parameters (such as `n_estimators=200`, `max_depth=7`, `min_samples_split=5`) to prevent overfitting and improve precision on the small dataset. Random Forests naturally resist overfitting in high-dimensional datasets and don't strictly require feature scaling.
- **Evaluation**: The script evaluates and outputs all required challenge metrics: *Accuracy, Precision, Recall, Specificity, F1-Score, and AUC*.
- **Submission Output**: It automatically builds the final `submission.csv` containing the required 5-digit zero-padded `ID` and `PD status` prediction columns for the test split, fully complying with the project guidelines.

### 2.3. `main.ipynb` (End-to-End Visual Pipeline)
The Jupyter notebook consolidates the operations from `prepare_data.py` and `train_model.py` into one streamlined environment, augmented with visual analytics.
- **Data Execution**: Re-runs data processing and model training sequentially within reproducible cells.
- **Visual Analytics**: 
  - Generates a **Feature Importances Bar Chart** ranking the top 10 attributes the Random Forest found most discriminative (e.g., median X-coordinate, average Y-coordinate, etc.).
  - Plots a **Confusion Matrix Heatmap** to visualize the raw numbers of True Positives, True Negatives, False Positives, and False Negatives, giving immediate insight into model Specificity.
  - Displays the **Receiver Operating Characteristic (ROC) Curve**, plotting the tradeoff between Sensitivity and Specificity alongside the calculated AUC score.

### 2.4. `app.py` (Interactive Dashboard)
We implemented a Streamlit-powered dashboard that visually summarizes the notebook's end-to-end capabilities dynamically on a webpage.
- **Live Retraining**: To simplify deployment, it extracts the dataset and instantly re-trains the model in memory.
- **Dynamic Views**: Presents the live evaluation metrics and seamlessly replicates the Confusion Matrix, ROC Curve, and Feature Importances natively in an accessible web UI using Streamlit's internal layout functions.

## 3. Results Summary
The optimized baseline model achieves an accuracy of approximately **66.7%** with an AUC of **0.71**, which is a solid baseline considering we are relying purely on global statistical time-series aggregates without deep learning techniques. The model demonstrates a good balance between identifying Parkinson's patients (Recall: **71.4%**) and healthy control subjects. Future improvements—such as employing Convolutional Neural Networks on Spectrogram representations of the time series as suggested in the project hint—are recommended to further improve overall F1-Score and AUC.
