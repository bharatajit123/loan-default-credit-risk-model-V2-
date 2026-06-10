# Production Deployment Guide

## 🚀 Deploying Gradient Boosting Model to Production

---

## Phase 1: Model Preparation

### Step 1: Export Trained Model

```python
import joblib
from sklearn.ensemble import GradientBoostingClassifier

# Save the trained model
model = GradientBoostingClassifier(...)
model.fit(X_train, y_train)

# Export model
joblib.dump(model, 'models/gradient_boosting_production.pkl')

# Export feature names for validation
import json
feature_names = list(X_train.columns)
with open('models/feature_metadata.json', 'w') as f:
    json.dump({
        'features': feature_names,
        'n_features': len(feature_names),
        'threshold': 0.20,
        'model_type': 'GradientBoostingClassifier'
    }, f)
```

### Step 2: Prepare Preprocessing Pipeline

```python
from sklearn.preprocessing import LabelEncoder
import pickle

# Save label encoders for categorical features
label_encoders = {
    'Education': LabelEncoder(),
    'EmploymentType': LabelEncoder(),
    'MaritalStatus': LabelEncoder(),
    # ... etc
}

for col, encoder in label_encoders.items():
    encoder.fit(X_train[col])

# Save all encoders
with open('models/label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)
```

---

## Phase 2: API Development

### Flask API Example

```python
# app.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load model and metadata
model = joblib.load('models/gradient_boosting_production.pkl')
with open('models/feature_metadata.json', 'r') as f:
    metadata = json.load(f)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'model': 'Gradient Boosting v1.0'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data
        data = request.json
        
        # Validate input
        required_features = metadata['features']
        missing_features = [f for f in required_features if f not in data]
        if missing_features:
            return jsonify({
                'error': f'Missing features: {missing_features}'
            }), 400
        
        # Create DataFrame
        df = pd.DataFrame([{f: data[f] for f in required_features}])
        
        # Preprocess (encode categoricals)
        for col in label_encoders:
            if col in df.columns:
                df[col] = label_encoders[col].transform(df[col])
        
        # Make prediction
        probability = model.predict_proba(df)[0, 1]
        risk_level = probability > metadata['threshold']
        
        return jsonify({
            'default_probability': float(probability),
            'risk_flag': bool(risk_level),
            'risk_level': 'HIGH' if risk_level else 'LOW',
            'threshold': metadata['threshold'],
            'recommendation': 'Manual Review Required' if risk_level else 'Auto-Approve'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Handle batch predictions"""
    try:
        data = request.json['loans']
        results = []
        
        for loan in data:
            # Predict for each loan
            df = pd.DataFrame([loan])
            probability = model.predict_proba(df)[0, 1]
            results.append({
                'loan_id': loan.get('id'),
                'probability': float(probability),
                'flag': probability > metadata['threshold']
            })
        
        return jsonify({
            'total': len(results),
            'flagged': sum(1 for r in results if r['flag']),
            'predictions': results
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
```

### Docker Containerization

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy app
COPY app.py .
COPY models/ models/

# Expose port
EXPOSE 5000

# Run
CMD ["python", "app.py"]
```

Build and run:
```bash
docker build -t loan-default-api .
docker run -p 5000:5000 loan-default-api
```

---

## Phase 3: Model Monitoring

### Performance Tracking

```python
# monitoring.py
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
import json
from datetime import datetime

class ModelMonitor:
    def __init__(self, model_name, threshold=0.20):
        self.model_name = model_name
        self.threshold = threshold
        self.metrics_history = []
    
    def evaluate_batch(self, y_true, y_pred_proba):
        """Evaluate on new data"""
        y_pred = (y_pred_proba >= self.threshold).astype(int)
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'roc_auc': float(roc_auc_score(y_true, y_pred_proba)),
            'total_samples': len(y_true),
            'flagged_count': int((y_pred_proba >= self.threshold).sum())
        }
        
        self.metrics_history.append(metrics)
        return metrics
    
    def check_drift(self, threshold=0.05):
        """Check for model performance degradation"""
        if len(self.metrics_history) < 2:
            return None
        
        baseline = self.metrics_history[0]['accuracy']
        current = self.metrics_history[-1]['accuracy']
        drift = (baseline - current) / baseline
        
        if drift > threshold:
            return {
                'drift_detected': True,
                'drift_percentage': drift * 100,
                'recommendation': 'Retrain model with recent data'
            }
        return {'drift_detected': False}
    
    def save_metrics(self, filename):
        """Save metrics for dashboard"""
        with open(filename, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)

# Usage
monitor = ModelMonitor('GradientBoosting')
metrics = monitor.evaluate_batch(y_test, y_pred_proba)
print(metrics)

drift_status = monitor.check_drift()
if drift_status['drift_detected']:
    print("⚠️  Model drift detected! Initiate retraining.")
```

---

## Phase 4: Deployment Checklist

### Pre-Deployment
- [ ] Model tested on holdout test set
- [ ] Performance metrics documented
- [ ] Feature validation implemented
- [ ] Error handling tested
- [ ] API endpoint tested locally
- [ ] Docker image built and tested
- [ ] Monitoring script ready
- [ ] Fallback plan documented

### Deployment
- [ ] Deploy to staging environment
- [ ] Run smoke tests
- [ ] Monitor for 1 week
- [ ] Get stakeholder approval
- [ ] Deploy to production
- [ ] Monitor predictions in real-time
- [ ] Set up alerts for model failures

### Post-Deployment
- [ ] Daily performance checks
- [ ] Weekly metrics review
- [ ] Monthly drift analysis
- [ ] Quarterly retraining evaluation
- [ ] Collect feedback from users

---

## Phase 5: Retraining Schedule

### Monthly Review
```python
# monthly_review.py
def monthly_review():
    """Run at end of each month"""
    # 1. Load new data
    new_data = load_production_data(last_30_days=True)
    
    # 2. Get actual outcomes
    actual_defaults = get_actual_defaults(last_30_days=True)
    
    # 3. Compare predictions vs actual
    accuracy = evaluate_predictions(new_data, actual_defaults)
    
    # 4. If accuracy drops > 5%, flag for retraining
    if accuracy < baseline_accuracy - 0.05:
        log_alert("Model accuracy degraded. Retraining recommended.")
        return True
    
    return False
```

### Quarterly Retraining
```python
# retrain.py
def retrain_model():
    """Retrain with latest 3 months of data"""
    
    # 1. Combine old and new data
    training_data = combine_datasets(
        current_data='data/loans.csv',
        new_data=load_production_data(last_90_days=True)
    )
    
    # 2. Preprocess
    X, y = preprocess(training_data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # 3. Train new model
    model_new = GradientBoostingClassifier(...)
    model_new.fit(X_train, y_train)
    
    # 4. Compare performance
    old_auc = evaluate_model(model_old, X_test, y_test)
    new_auc = evaluate_model(model_new, X_test, y_test)
    
    if new_auc > old_auc:
        # 5. Deploy new model
        save_model(model_new, 'models/gradient_boosting_production.pkl')
        log_info(f"Model updated. New AUC: {new_auc:.4f}")
    else:
        log_warning("New model not better. Keeping current model.")
```

---

## Phase 6: Monitoring Dashboard

### Key Metrics to Track

```python
# dashboard_metrics.py
DAILY_METRICS = {
    'Total Loans Processed': None,
    'Flagged for Review': None,
    'Approval Rate': None,
    'Model Prediction Time': None,
    'API Availability': None
}

WEEKLY_METRICS = {
    'Average Default Probability': None,
    'Model Accuracy (if outcomes available)': None,
    'False Positive Rate': None,
    'False Negative Rate': None
}

MONTHLY_METRICS = {
    'ROC-AUC': None,
    'Precision': None,
    'Recall': None,
    'Model Drift': None,
    'Data Distribution Changes': None
}
```

---

## Phase 7: Troubleshooting

### Common Issues

| Problem | Solution |
|---------|----------|
| **Model predictions too high/low** | Check for data drift; may need retraining |
| **API response slow** | Optimize model or use model quantization |
| **Feature validation fails** | Check for new categorical values in production data |
| **Accuracy drops suddenly** | Check if recent data differs significantly; consider retraining |

---

## Production URLs & Configuration

```yaml
# config.yaml
ENVIRONMENT: production
API_HOST: 0.0.0.0
API_PORT: 5000
MODEL_PATH: models/gradient_boosting_production.pkl
THRESHOLD: 0.20
LOG_LEVEL: INFO
ALERT_EMAIL: ml-alerts@company.com
RETRAINING_SCHEDULE: "0 0 1 * *"  # First day of month
```

---

## Contact & Support

- **Model Owner:** Data Science Team
- **Emergency:** On-call data scientist
- **Status Page:** metrics.company.com

---

**Last Updated:** June 10, 2026
