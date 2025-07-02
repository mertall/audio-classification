import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from collections import Counter
from typing import Optional
import os

# --- Feature Extraction ---
def extract_features(fpath: str) -> np.ndarray:
    df = pd.read_csv(fpath)
    freqs = df.iloc[:, 0].values
    matrix = df.iloc[:, 1:].values

    log_matrix = np.log1p(matrix)
    denom = matrix.sum(axis=0)
    denom = np.where(denom == 0, 1e-8, denom)

    centroid = (freqs[:, None] * matrix).sum(axis=0) / denom
    centroid_mean = np.mean(np.nan_to_num(centroid))

    spread = ((freqs[:, None] - centroid_mean) ** 2 * matrix).sum(axis=0) / denom
    bandwidth_mean = np.sqrt(np.mean(np.nan_to_num(spread)))

    energy_per_time = matrix.sum(axis=0)
    cumsum = np.cumsum(matrix, axis=0)
    rolloff_idx = (cumsum >= 0.85 * energy_per_time).argmax(axis=0)
    rolloff_85 = np.nanmean(freqs[rolloff_idx])

    freq_energy = matrix.sum(axis=1)
    peak_freq = freqs[np.argmax(freq_energy)]

    return np.array([centroid_mean, bandwidth_mean, rolloff_85, peak_freq])

# --- Load Training Data (exclude labeled) ---
def load_training_data(data_dir: str):
    features = []
    files = []
    for fname in os.listdir(data_dir):
        if not fname.endswith(".csv"):
            continue
        if fname in ("top.csv", "bottom.csv"):
            continue
        fpath = os.path.join(data_dir, fname)
        features.append(extract_features(fpath))
        files.append(fname)
    return np.array(features), files

# --- Main classification function ---
def classify_preprocessed_audio(fpath: str) -> Optional[int]:
    # We assume a 'data' folder in current working directory holds all CSVs
    data_dir = os.path.abspath(os.path.join(os.getcwd(), 'data'))

    # Load and scale training features (exclude labeled)
    X_train, _ = load_training_data(data_dir)
    if X_train.size == 0:
        return None
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Fit KMeans with 3 clusters
    kmeans = KMeans(n_clusters=3, random_state=0)
    kmeans.fit(X_train_scaled)

    # Map clusters to "top" / "bottom" using the two labeled examples
    label_map = {}
    for label in ("top", "bottom"):
        label_path = os.path.join(data_dir, f"{label}.csv")
        if not os.path.exists(label_path):
            return None
        vec = extract_features(label_path)
        vec_scaled = scaler.transform([vec])[0]
        cluster = kmeans.predict([vec_scaled])[0]
        label_map[cluster] = label

    # Classify the new file
    if not os.path.exists(fpath):
        return None
    vec = extract_features(fpath)
    vec_scaled = scaler.transform([vec])[0]
    cluster = kmeans.predict([vec_scaled])[0]
    pred = label_map.get(cluster)
    if pred == "top":
        return 0
    if pred == "bottom":
        return 1
    return None
