"""
Model Evaluation Module

Functions for evaluating model performance and metrics.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, auc,
    precision_recall_curve, classification_report
)


def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    """
    Calculate evaluation metrics.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_pred_proba : array-like, optional
        Predicted probabilities
    
    Returns:
    --------
    dict
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
    }
    
    if y_pred_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
    
    return metrics


def get_confusion_matrix(y_true, y_pred):
    """
    Get confusion matrix and derived metrics.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    
    Returns:
    --------
    dict
        Confusion matrix components
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'true_positives': tp,
        'specificity': specificity,
        'confusion_matrix': cm
    }


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model on test set.
    
    Parameters:
    -----------
    model : sklearn model
        Trained model
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test target
    
    Returns:
    --------
    dict
        Evaluation results
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
    cm_info = get_confusion_matrix(y_test, y_pred)
    
    return {
        'metrics': metrics,
        'confusion_matrix': cm_info,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }


def evaluate_models(models, X_test, y_test):
    """
    Evaluate multiple models.
    
    Parameters:
    -----------
    models : dict
        Dictionary of trained models
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test target
    
    Returns:
    --------
    dict
        Evaluation results for all models
    """
    results = {}
    
    for name, model in models.items():
        results[name] = evaluate_model(model, X_test, y_test)
    
    return results


def apply_threshold(y_pred_proba, threshold=0.5):
    """
    Apply custom threshold to probability predictions.
    
    Parameters:
    -----------
    y_pred_proba : array-like
        Predicted probabilities
    threshold : float
        Threshold value (0-1)
    
    Returns:
    --------
    array
        Binary predictions
    """
    return (y_pred_proba >= threshold).astype(int)


def analyze_threshold(y_true, y_pred_proba, thresholds=[0.5, 0.3, 0.2, 0.15, 0.1]):
    """
    Analyze performance at different thresholds.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred_proba : array-like
        Predicted probabilities
    thresholds : list
        Thresholds to evaluate
    
    Returns:
    --------
    dict
        Performance metrics at each threshold
    """
    results = []
    
    for threshold in thresholds:
        y_pred = apply_threshold(y_pred_proba, threshold)
        metrics = calculate_metrics(y_true, y_pred)
        cm_info = get_confusion_matrix(y_true, y_pred)
        
        results.append({
            'threshold': threshold,
            **metrics,
            **cm_info,
            'flagged_count': (y_pred == 1).sum(),
            'flagged_pct': (y_pred == 1).sum() / len(y_pred) * 100
        })
    
    return results


def get_classification_report(y_true, y_pred):
    """
    Get detailed classification report.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    
    Returns:
    --------
    str
        Formatted classification report
    """
    return classification_report(y_true, y_pred, 
                                target_names=['No Default', 'Default'])


def get_roc_curve(y_true, y_pred_proba):
    """
    Get ROC curve coordinates.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred_proba : array-like
        Predicted probabilities
    
    Returns:
    --------
    tuple
        (fpr, tpr, thresholds)
    """
    return roc_curve(y_true, y_pred_proba)
