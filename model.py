"""
Model Training Module

Functions for training and saving machine learning models.
"""

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB


def create_logistic_regression(max_iter=1000, random_state=42):
    """Create Logistic Regression model."""
    return LogisticRegression(max_iter=max_iter, random_state=random_state, n_jobs=-1)


def create_decision_tree(max_depth=15, random_state=42):
    """Create Decision Tree model."""
    return DecisionTreeClassifier(random_state=random_state, max_depth=max_depth)


def create_random_forest(n_estimators=100, max_depth=15, random_state=42):
    """Create Random Forest model."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
    )


def create_naive_bayes():
    """Create Naive Bayes model."""
    return GaussianNB()


def create_gradient_boosting(n_estimators=100, max_depth=6, random_state=42):
    """Create Gradient Boosting model."""
    return GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        learning_rate=0.1
    )


def train_model(model, X_train, y_train):
    """
    Train a model.
    
    Parameters:
    -----------
    model : sklearn model
        Untrained model
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    
    Returns:
    --------
    sklearn model
        Trained model
    """
    model.fit(X_train, y_train)
    return model


def save_model(model, filepath):
    """
    Save trained model to disk.
    
    Parameters:
    -----------
    model : sklearn model
        Trained model
    filepath : str
        Path to save model
    """
    joblib.dump(model, filepath)
    print(f"✓ Model saved to {filepath}")


def load_model(filepath):
    """
    Load trained model from disk.
    
    Parameters:
    -----------
    filepath : str
        Path to saved model
    
    Returns:
    --------
    sklearn model
        Loaded model
    """
    return joblib.load(filepath)


def train_all_models(X_train, y_train):
    """
    Train all models.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    
    Returns:
    --------
    dict
        Dictionary of trained models
    """
    models = {}
    
    model_creators = {
        'Logistic Regression': create_logistic_regression,
        'Decision Tree': create_decision_tree,
        'Random Forest': create_random_forest,
        'Naive Bayes': create_naive_bayes,
        'Gradient Boosting': create_gradient_boosting
    }
    
    for name, creator in model_creators.items():
        print(f"Training {name}...")
        model = creator()
        model = train_model(model, X_train, y_train)
        models[name] = model
        print(f"  ✓ {name} trained")
    
    return models
