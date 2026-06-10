"""
Utility Functions

Helper functions for common tasks.
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime


def save_config(config, filepath):
    """
    Save configuration to JSON file.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary
    filepath : str
        Path to save file
    """
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✓ Config saved to {filepath}")


def load_config(filepath):
    """
    Load configuration from JSON file.
    
    Parameters:
    -----------
    filepath : str
        Path to config file
    
    Returns:
    --------
    dict
        Configuration dictionary
    """
    with open(filepath, 'r') as f:
        config = json.load(f)
    return config


def print_metrics_table(metrics_dict):
    """
    Print metrics in formatted table.
    
    Parameters:
    -----------
    metrics_dict : dict
        Dictionary of metric names and values
    """
    print("\nMetrics:")
    print("-" * 50)
    for name, value in metrics_dict.items():
        if isinstance(value, float):
            print(f"  {name:.<40} {value:.4f}")
        else:
            print(f"  {name:.<40} {value}")
    print("-" * 50)


def get_timestamp():
    """
    Get current timestamp.
    
    Returns:
    --------
    str
        ISO format timestamp
    """
    return datetime.now().isoformat()


def create_report_header(title, description=""):
    """
    Create formatted report header.
    
    Parameters:
    -----------
    title : str
        Report title
    description : str, optional
        Report description
    
    Returns:
    --------
    str
        Formatted header
    """
    header = f"\n{'='*80}\n{title}\n{'='*80}\n"
    if description:
        header += f"{description}\n\n"
    return header


def summary_statistics(df):
    """
    Get summary statistics for dataframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    
    Returns:
    --------
    dict
        Summary statistics
    """
    return {
        'rows': len(df),
        'columns': len(df.columns),
        'memory_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'missing_values': df.isnull().sum().sum(),
        'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': len(df.select_dtypes(include=['object']).columns)
    }


def validate_input(data, required_fields):
    """
    Validate input has required fields.
    
    Parameters:
    -----------
    data : dict
        Input data
    required_fields : list
        Required field names
    
    Returns:
    --------
    tuple
        (is_valid, missing_fields)
    """
    missing = [field for field in required_fields if field not in data]
    return len(missing) == 0, missing


def scale_values(values, min_val=0, max_val=100):
    """
    Scale values to range [min_val, max_val].
    
    Parameters:
    -----------
    values : array-like
        Values to scale
    min_val : float
        Minimum value (default: 0)
    max_val : float
        Maximum value (default: 100)
    
    Returns:
    --------
    array
        Scaled values
    """
    values = np.array(values)
    return (values - values.min()) / (values.max() - values.min()) * (max_val - min_val) + min_val


def format_percentage(value, decimals=2):
    """
    Format value as percentage string.
    
    Parameters:
    -----------
    value : float
        Value (0-1)
    decimals : int
        Number of decimal places
    
    Returns:
    --------
    str
        Formatted percentage
    """
    return f"{value*100:.{decimals}f}%"


def format_currency(value, currency='$'):
    """
    Format value as currency string.
    
    Parameters:
    -----------
    value : float
        Value
    currency : str
        Currency symbol
    
    Returns:
    --------
    str
        Formatted currency
    """
    return f"{currency}{value:,.2f}"


def get_risk_category(probability, thresholds=None):
    """
    Categorize risk based on probability.
    
    Parameters:
    -----------
    probability : float
        Default probability (0-1)
    thresholds : dict, optional
        Custom thresholds for categories
    
    Returns:
    --------
    str
        Risk category
    """
    if thresholds is None:
        thresholds = {
            'low': 0.15,
            'medium': 0.30,
            'high': 0.50
        }
    
    if probability < thresholds['low']:
        return 'Very Low'
    elif probability < thresholds['medium']:
        return 'Low'
    elif probability < thresholds['high']:
        return 'Medium'
    else:
        return 'High'


def generate_summary_report(y_true, y_pred, y_pred_proba):
    """
    Generate comprehensive summary report.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_pred_proba : array-like
        Predicted probabilities
    
    Returns:
    --------
    str
        Formatted report
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, 
        f1_score, roc_auc_score, confusion_matrix
    )
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    cm = confusion_matrix(y_true, y_pred)
    
    report = f"""
{'='*80}
MODEL PERFORMANCE SUMMARY
{'='*80}

Metrics:
  Accuracy:  {accuracy:.4f}
  Precision: {precision:.4f}
  Recall:    {recall:.4f}
  F1-Score:  {f1:.4f}
  ROC-AUC:   {roc_auc:.4f}

Confusion Matrix:
  True Negatives:   {cm[0,0]:>6}
  False Positives:  {cm[0,1]:>6}
  False Negatives:  {cm[1,0]:>6}
  True Positives:   {cm[1,1]:>6}

{'='*80}
    """
    return report
