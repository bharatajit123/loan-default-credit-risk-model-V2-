"""
Data Preprocessing Module

Functions for loading, cleaning, and preparing data for model training.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def load_data(filepath):
    """
    Load loan dataset from CSV file.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
    
    Returns:
    --------
    pd.DataFrame
        Loaded dataset
    """
    df = pd.read_csv(filepath)
    return df


def check_data_quality(df):
    """
    Check data quality (missing values, duplicates, etc.)
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to check
    
    Returns:
    --------
    dict
        Dictionary with quality metrics
    """
    quality = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicates': df.duplicated().sum(),
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # MB
    }
    return quality


def separate_features_target(df):
    """
    Separate features and target variable.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset with features and target
    
    Returns:
    --------
    tuple
        (X, y) - Features and target
    """
    X = df.drop(['LoanID', 'Default'], axis=1)
    y = df['Default']
    return X, y


def encode_categorical(X):
    """
    Encode categorical variables using LabelEncoder.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Features with categorical columns
    
    Returns:
    --------
    tuple
        (X_encoded, label_encoders) - Encoded features and encoder objects
    """
    X_encoded = X.copy()
    label_encoders = {}
    
    categorical_cols = X_encoded.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col])
        label_encoders[col] = le
    
    return X_encoded, label_encoders


def train_test_split_stratified(X, y, test_size=0.2, random_state=42):
    """
    Split data into train and test sets with stratification.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Features
    y : pd.Series
        Target variable
    test_size : float
        Proportion of test set (default: 0.2)
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y
    )
    return X_train, X_test, y_train, y_test


def preprocess_pipeline(filepath):
    """
    Complete preprocessing pipeline.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
    
    Returns:
    --------
    dict
        Dictionary containing all preprocessing outputs
    """
    # Load
    df = load_data(filepath)
    
    # Check quality
    quality = check_data_quality(df)
    
    # Separate
    X, y = separate_features_target(df)
    
    # Encode
    X_encoded, label_encoders = encode_categorical(X)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split_stratified(X_encoded, y)
    
    return {
        'data': df,
        'X': X_encoded,
        'y': y,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'label_encoders': label_encoders,
        'quality': quality
    }
