import os
import glob
import pandas as pd
import numpy as np

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = os.path.join('PaHaW', 'PaHaW_public')
LABELS_FILE = os.path.join('PaHaW', 'PaHaW_files', 'corpus_PaHaW.xlsx')
OUTPUT_FILE = 'dataset_features.csv'

EXCLUDE_SUBJECTS = ['00061', '00080', '00089']

def extract_features(file_path):
    # Columns: Y coordinate, X coordinate, time stamp, button state, azimuth, altitude, pressure
    # 1st line is number of samples, so skip it
    try:
        df = pd.read_csv(file_path, sep=' ', header=None, skiprows=1, 
                         names=['Y', 'X', 'Time', 'Button', 'Azimuth', 'Altitude', 'Pressure'])
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

    if len(df) < 2:
        return None
        
    features = {}
    
    # Basic stats for main columns
    for col in ['X', 'Y', 'Pressure', 'Altitude', 'Azimuth']:
        features[f'{col}_mean'] = df[col].mean()
        features[f'{col}_std'] = df[col].std()
        features[f'{col}_min'] = df[col].min()
        features[f'{col}_max'] = df[col].max()
        features[f'{col}_median'] = df[col].median()
        
    # Button State features
    features['Button_mean'] = df['Button'].mean()  # Proportion of time pen is on surface
    features['Pen_lifts'] = (df['Button'].diff() == -1).sum()  # Count of 1 -> 0 transitions
    
    # Time features
    features['Total_Duration'] = df['Time'].iloc[-1] - df['Time'].iloc[0]
    
    # Velocity features
    dx = df['X'].diff().fillna(0)
    dy = df['Y'].diff().fillna(0)
    dt = df['Time'].diff().fillna(0)
    
    # Avoid division by zero
    dt_safe = dt.replace(0, 1e-6)
    v = np.sqrt(dx**2 + dy**2) / dt_safe
    
    # Remove the first velocity point as diff is 0
    v = v[1:]
    if len(v) > 0:
        features['Velocity_mean'] = v.mean()
        features['Velocity_std'] = v.std()
        features['Velocity_max'] = v.max()
    else:
        features['Velocity_mean'] = 0
        features['Velocity_std'] = 0
        features['Velocity_max'] = 0
        
    return features

def prepare_dataset():
    print("Loading labels...")
    df_labels = pd.read_excel(LABELS_FILE)
    df_labels['ID'] = df_labels['ID'].astype(str).str.zfill(5)
    labels_dict = dict(zip(df_labels['ID'], (df_labels['Disease'] == 'PD').astype(int)))
    
    dataset = []
    
    subject_dirs = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    
    print(f"Found {len(subject_dirs)} subject directories.")
    
    for subject_id in subject_dirs:
        if subject_id in EXCLUDE_SUBJECTS:
            continue
            
        if subject_id not in labels_dict:
            print(f"Warning: Subject {subject_id} not found in labels. Skipping.")
            continue
            
        subject_path = os.path.join(DATA_DIR, subject_id)
        svc_files = glob.glob(os.path.join(subject_path, "*.svc"))
        
        if not svc_files:
            continue
            
        subject_features_list = []
        for svc_file in svc_files:
            feats = extract_features(svc_file)
            if feats is not None:
                subject_features_list.append(feats)
                
        if subject_features_list:
            # Average features across all tasks for this subject
            df_subj = pd.DataFrame(subject_features_list)
            subj_avg_features = df_subj.mean().to_dict()
            
            subj_avg_features['Subject_ID'] = subject_id
            subj_avg_features['Label'] = labels_dict[subject_id]
            
            dataset.append(subj_avg_features)
            
    print(f"Extracted features for {len(dataset)} subjects.")
    
    final_df = pd.DataFrame(dataset)
    # Move Subject_ID and Label to front
    cols = ['Subject_ID', 'Label'] + [c for c in final_df.columns if c not in ['Subject_ID', 'Label']]
    final_df = final_df[cols]
    
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Dataset saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    prepare_dataset()
