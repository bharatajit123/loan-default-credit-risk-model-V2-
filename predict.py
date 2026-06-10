"""
Prediction Module

Functions for making predictions with trained models.
"""

import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder


class LoanDefaultPredictor:
    """
    Class for making loan default predictions.
    """
    
    def __init__(self, model_path, encoders_path, metadata_path=None):
        """
        Initialize predictor.
        
        Parameters:
        -----------
        model_path : str
            Path to trained model
        encoders_path : str
            Path to label encoders
        metadata_path : str, optional
            Path to metadata JSON
        """
        self.model = joblib.load(model_path)
        self.label_encoders = joblib.load(encoders_path)
        self.threshold = 0.20  # Default threshold
        
        if metadata_path:
            import json
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
    
    def preprocess_input(self, data):
        """
        Preprocess input data.
        
        Parameters:
        -----------
        data : dict or pd.DataFrame
            Input data for prediction
        
        Returns:
        --------
        pd.DataFrame
            Preprocessed data
        """
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        data_processed = data.copy()
        
        # Encode categorical columns
        for col in self.label_encoders:
            if col in data_processed.columns:
                data_processed[col] = self.label_encoders[col].transform(data_processed[col])
        
        return data_processed
    
    def predict(self, data):
        """
        Make prediction on single loan.
        
        Parameters:
        -----------
        data : dict or pd.DataFrame
            Single loan features
        
        Returns:
        --------
        dict
            Prediction results
        """
        data_processed = self.preprocess_input(data)
        
        probability = self.model.predict_proba(data_processed)[0, 1]
        flagged = probability > self.threshold
        
        return {
            'default_probability': float(probability),
            'risk_level': 'HIGH' if flagged else 'LOW',
            'flagged': bool(flagged),
            'threshold': self.threshold,
            'recommendation': 'Manual Review Required' if flagged else 'Auto-Approve'
        }
    
    def predict_batch(self, data):
        """
        Make predictions on multiple loans.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Batch of loan features
        
        Returns:
        --------
        pd.DataFrame
            Predictions for each loan
        """
        data_processed = self.preprocess_input(data)
        
        probabilities = self.model.predict_proba(data_processed)[:, 1]
        
        results = pd.DataFrame({
            'default_probability': probabilities,
            'risk_level': ['HIGH' if p > self.threshold else 'LOW' for p in probabilities],
            'flagged': probabilities > self.threshold
        })
        
        return results
    
    def set_threshold(self, threshold):
        """
        Set prediction threshold.
        
        Parameters:
        -----------
        threshold : float
            Threshold value (0-1)
        """
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        self.threshold = threshold


def make_single_prediction(model_path, encoders_path, data, threshold=0.20):
    """
    Quick prediction for single loan.
    
    Parameters:
    -----------
    model_path : str
        Path to model
    encoders_path : str
        Path to encoders
    data : dict
        Loan features
    threshold : float
        Decision threshold
    
    Returns:
    --------
    dict
        Prediction result
    """
    predictor = LoanDefaultPredictor(model_path, encoders_path)
    predictor.set_threshold(threshold)
    return predictor.predict(data)


def make_batch_prediction(model_path, encoders_path, data, threshold=0.20):
    """
    Quick prediction for batch of loans.
    
    Parameters:
    -----------
    model_path : str
        Path to model
    encoders_path : str
        Path to encoders
    data : pd.DataFrame
        Batch of loans
    threshold : float
        Decision threshold
    
    Returns:
    --------
    pd.DataFrame
        Predictions
    """
    predictor = LoanDefaultPredictor(model_path, encoders_path)
    predictor.set_threshold(threshold)
    return predictor.predict_batch(data)
