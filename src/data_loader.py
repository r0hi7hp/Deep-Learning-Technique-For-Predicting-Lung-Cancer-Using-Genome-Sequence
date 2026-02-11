import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import sys

def get_kmers(sequence, k=3):
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]

def load_and_preprocess_data(file_path, k=3):
    if not os.path.exists(file_path):
        print(f"Error: Dataset file not found at {file_path}")
        print("Please place the 'Final.csv' file in the specified path or run with --data_path argument.")
        sys.exit(1)

    print(f"Loading data from {file_path}...")
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)

    if 'sequence' not in df.columns or 'label' not in df.columns:
        print("Error: Dataset must contain 'sequence' and 'label' columns.")
        sys.exit(1)

    X = df['sequence']
    y = df['label']

    print("Encoding sequences with K-mers...")
    # Build k-mer vocabulary
    all_kmers = set()
    for seq in X:
        all_kmers.update(get_kmers(seq, k))
    
    # 0 reserved for padding
    kmer2idx = {kmer: idx + 1 for idx, kmer in enumerate(sorted(all_kmers))}
    
    # Encode sequences
    X_kmer_encoded = [[kmer2idx[kmer] for kmer in get_kmers(seq, k)] for seq in X]
    maxlen = max(len(seq) for seq in X_kmer_encoded)
    
    print(f"Max sequence length: {maxlen}")
    print(f"Vocabulary size: {len(kmer2idx)}")

    X_padded = pad_sequences(X_kmer_encoded, maxlen=maxlen, padding='post')

    print("Balancing dataset using SMOTE...")
    # Original logic used sampling_strategy={1: 50000}, but this might fail if class 0 has < 50000.
    # To make it robust, we'll auto-balance or use the logic if applicable.
    # For closer fidelity to original: 
    try:
        smote = SMOTE(sampling_strategy={1: 50000}, random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_padded, y)
    except ValueError as e:
        print(f"SMOTE warning (adjustment might be needed): {e}")
        print("Falling back to auto strategy if specific strategy fails.")
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_padded, y)

    print("Splitting data...")
    # Split: Train (70%), Temp (30%) -> Val (10%), Test (20%)
    X_train, X_temp, y_train, y_temp = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1/3, random_state=42)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), maxlen, len(kmer2idx)
