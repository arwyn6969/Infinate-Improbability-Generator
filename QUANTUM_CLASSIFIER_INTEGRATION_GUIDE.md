# 🔗 Quantum XGBoost Classifier - Integration Guide

## Overview

This guide provides step-by-step instructions for integrating the new QuantumXGBoostClassifier with the existing IIG quantum trading system.

## 🎯 Integration Points

### 1. QAOA Momentum Detector Integration

**File to Modify:** `packages/quantum_finance/src/quantum_finance/vwap/qaoa_momentum.py`

**Current Code (Rule-Based):**
```python
def _analyze_quantum_state(self, quantum_state):
    # Extract quantum features
    bullish_prob = quantum_state.get('bullish_prob', 0.5)
    bearish_prob = quantum_state.get('bearish_prob', 0.5)
    total_momentum = max(bullish_prob, bearish_prob)

    # Rule-based classification (REPLACE THIS)
    if total_momentum > self.momentum_threshold:
        direction = 'BULLISH' if bullish_prob > bearish_prob else 'BEARISH'
        strength = total_momentum
    else:
        direction = 'NEUTRAL'
        strength = 0.0

    return MomentumSignal(
        direction=direction,
        strength=strength,
        quantum_features=quantum_state
    )
```

**New Code (ML-Based):**
```python
def _analyze_quantum_state(self, quantum_state):
    # Extract quantum features
    features = self._extract_quantum_features(quantum_state)

    # Use ML classifier instead of rules
    try:
        predictions, probabilities = self.ml_classifier.predict(features)

        # Convert to momentum signal format
        direction = 'BULLISH' if predictions[0] == 1 else 'BEARISH'
        confidence = probabilities[0] if predictions[0] == 1 else (1 - probabilities[0])
        strength = confidence  # Use ML confidence as strength

    except Exception as e:
        # Fallback to rule-based if ML fails
        logger.warning(f"ML classification failed, using rule-based fallback: {e}")
        direction, strength = self._rule_based_classification(quantum_state)

    return MomentumSignal(
        direction=direction,
        strength=strength,
        quantum_features=quantum_state,
        ml_confidence=confidence if 'confidence' in locals() else None
    )
```

### 2. Initialize ML Classifier

**Add to QAOA Momentum Detector `__init__`:**
```python
from quantum_finance.ml.quantum_classifier import QuantumXGBoostClassifier

def __init__(self, ...):
    # Existing initialization...

    # Initialize ML classifier
    self.ml_classifier = None
    try:
        self.ml_classifier = QuantumXGBoostClassifier()
        self.ml_classifier.load_model("models/quantum_xgb_v1.json")
        logger.info("Quantum XGBoost classifier loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load ML classifier, using rule-based only: {e}")
```

### 3. Feature Extraction Method

**Add to QAOA Momentum Detector:**
```python
def _extract_quantum_features(self, quantum_state):
    """Extract features for ML classifier."""
    # Create DataFrame with single row
    features = pd.DataFrame({
        'bullish_prob': [quantum_state.get('bullish_prob', 0.5)],
        'bearish_prob': [quantum_state.get('bearish_prob', 0.5)],
        'directional_confidence': [quantum_state.get('directional_confidence', 0.5)],
        'quantum_advantage': [quantum_state.get('quantum_advantage', 1.0)],
        'total_momentum': [quantum_state.get('total_momentum', 0.5)]
    })

    # Apply feature engineering (lagged features, rolling stats)
    features = self._apply_feature_engineering(features)

    return features

def _apply_feature_engineering(self, df):
    """Apply feature engineering to match training pipeline."""
    # This should match the feature_engineering function from train_quantum_classifier.py
    # For single predictions, we need historical context from previous signals
    # Implementation depends on how historical data is stored
    return df  # Placeholder - implement based on your historical data storage
```

## 🚀 Training Pipeline Setup

### 1. Initial Model Training

```bash
# Train the classifier with mock data (for initial testing)
cd packages/quantum_finance
python src/quantum_finance/scripts/train_quantum_classifier.py --mock --save-path models/quantum_xgb_v1.json

# Train with real data (once you have quantum feature logs)
python src/quantum_finance/scripts/train_quantum_classifier.py --save-path models/quantum_xgb_v1.json
```

### 2. Feature Logging Setup

**Add to Quantum Feature Logging:**
```python
# In your quantum feature logging code
from quantum_finance.questdb_qaoa_features_manager import QuestDBQAOAFeaturesManager

features_manager = QuestDBQAOAFeaturesManager()

# Log features after each quantum analysis
feature_record = {
    'timestamp': datetime.now(),
    'bullish_prob': quantum_state['bullish_prob'],
    'bearish_prob': quantum_state['bearish_prob'],
    'directional_confidence': quantum_state['directional_confidence'],
    'quantum_advantage': quantum_state['quantum_advantage'],
    'total_momentum': quantum_state['total_momentum'],
    'price_change_5min': actual_price_change,  # Target variable
    'symbol': 'BTCUSDT'
}

features_manager.log_features(feature_record)
```

## 🧪 A/B Testing Framework

### 1. Parallel Classification

```python
class ABMomentumDetector:
    def __init__(self):
        self.rule_based = QAOAUptrendDetector(use_ml=False)
        self.ml_based = QAOAUptrendDetector(use_ml=True)
        self.results_logger = ResultsLogger()

    def detect_momentum(self, quantum_state, actual_return=None):
        # Get both predictions
        rule_signal = self.rule_based.detect_momentum(quantum_state)
        ml_signal = self.ml_based.detect_momentum(quantum_state)

        # Log for comparison
        if actual_return is not None:
            self.results_logger.log_comparison({
                'rule_direction': rule_signal.direction,
                'rule_strength': rule_signal.strength,
                'ml_direction': ml_signal.direction,
                'ml_strength': ml_signal.strength,
                'actual_return': actual_return,
                'quantum_features': quantum_state
            })

        return ml_signal  # Use ML by default
```

### 2. Performance Comparison

```python
from sklearn.metrics import accuracy_score, classification_report

def compare_classifiers(results_df):
    # Convert directions to binary
    rule_pred = (results_df['rule_direction'] == 'BULLISH').astype(int)
    ml_pred = (results_df['ml_direction'] == 'BULLISH').astype(int)
    actual = (results_df['actual_return'] > 0).astype(int)

    print("Rule-Based Performance:")
    print(classification_report(actual, rule_pred))

    print("\nML-Based Performance:")
    print(classification_report(actual, ml_pred))

    print(f"\nAccuracy Improvement: {accuracy_score(actual, ml_pred) - accuracy_score(actual, rule_pred):.3f}")
```

## 📊 Monitoring & Maintenance

### 1. Model Performance Monitoring

```python
class ModelMonitor:
    def __init__(self):
        self.performance_history = []

    def log_prediction(self, features, prediction, actual_return, confidence):
        self.performance_history.append({
            'timestamp': datetime.now(),
            'prediction': prediction,
            'actual': 1 if actual_return > 0 else 0,
            'confidence': confidence,
            'features': features
        })

    def get_performance_metrics(self):
        if len(self.performance_history) < 100:
            return None

        recent = self.performance_history[-100:]
        predictions = [r['prediction'] for r in recent]
        actuals = [r['actual'] for r in recent]

        accuracy = accuracy_score(actuals, predictions)
        return {
            'accuracy': accuracy,
            'sample_size': len(recent),
            'drift_detected': accuracy < 0.55  # Alert if performance drops
        }
```

### 2. Model Retraining Pipeline

```bash
# Automated retraining script
#!/bin/bash
cd packages/quantum_finance

# Check if we have enough new data
NEW_SAMPLES=$(python -c "from questdb_qaoa_features_manager import QuestDBQAOAFeaturesManager; print(len(QuestDBQAOAFeaturesManager().export_training_data()[0] or []))")

if [ "$NEW_SAMPLES" -gt 1000 ]; then
    echo "Retraining model with $NEW_SAMPLES samples..."
    python src/quantum_finance/scripts/train_quantum_classifier.py --save-path models/quantum_xgb_$(date +%Y%m%d).json

    # Validate new model
    python validate_quantum_classifier.py --skip-training

    # Deploy if validation passes
    if [ $? -eq 0 ]; then
        cp models/quantum_xgb_$(date +%Y%m%d).json models/quantum_xgb_v1.json
        echo "New model deployed successfully"
    fi
else
    echo "Not enough new samples ($NEW_SAMPLES), skipping retraining"
fi
```

## ⚠️ Error Handling & Fallbacks

### 1. Graceful Degradation

```python
def safe_predict(self, features):
    """Safe prediction with automatic fallback."""
    try:
        if self.ml_classifier is None:
            raise ValueError("ML classifier not initialized")

        predictions, probabilities = self.ml_classifier.predict(features)
        return predictions, probabilities, "ml"

    except Exception as e:
        logger.warning(f"ML prediction failed: {e}, using rule-based fallback")
        try:
            # Implement rule-based fallback
            rule_prediction = self._rule_based_predict(features)
            return rule_prediction, 0.5, "rule_based"
        except Exception as e2:
            logger.error(f"Rule-based fallback also failed: {e2}")
            # Ultimate fallback - neutral signal
            return np.array([0]), 0.5, "neutral_fallback"
```

### 2. Health Checks

```python
def health_check(self):
    """Comprehensive health check for the classifier."""
    issues = []

    # Check model loading
    if self.ml_classifier is None:
        issues.append("ML classifier not initialized")

    # Check model file exists
    if not Path("models/quantum_xgb_v1.json").exists():
        issues.append("Model file missing")

    # Test prediction with dummy data
    try:
        dummy_features = pd.DataFrame({
            'bullish_prob': [0.6], 'bearish_prob': [0.4],
            'directional_confidence': [0.5], 'quantum_advantage': [2.0],
            'total_momentum': [0.6]
        })
        pred, prob = self.ml_classifier.predict(dummy_features)
        if len(pred) != 1 or not (0 <= prob[0] <= 1):
            issues.append("Model prediction test failed")
    except Exception as e:
        issues.append(f"Model prediction error: {e}")

    return {
        'healthy': len(issues) == 0,
        'issues': issues,
        'model_loaded': self.ml_classifier is not None
    }
```

## 🔧 Configuration

### 1. Model Configuration

```yaml
# config/quantum_classifier.yaml
model:
  path: "models/quantum_xgb_v1.json"
  fallback_enabled: true
  confidence_threshold: 0.6  # Minimum confidence for trading signals
  retrain_interval_days: 7
  min_samples_for_retrain: 1000

monitoring:
  enabled: true
  metrics_interval_minutes: 60
  alert_on_accuracy_drop: 0.05  # Alert if accuracy drops by 5%
```

### 2. Feature Engineering Configuration

```python
FEATURE_CONFIG = {
    'lags': [1, 2, 3, 5, 10],  # Lag periods to create
    'rolling_windows': [5, 10, 20],  # Rolling window sizes
    'base_features': [
        'bullish_prob', 'bearish_prob', 'total_momentum',
        'directional_confidence', 'quantum_advantage'
    ]
}
```

## 📈 Performance Optimization

### 1. Prediction Caching

```python
from functools import lru_cache
import hashlib

class CachedClassifier:
    def __init__(self, classifier, cache_size=1000):
        self.classifier = classifier
        self.cache = {}

    def predict(self, features):
        # Create cache key from features
        feature_hash = hashlib.md5(str(features.values.tobytes()).encode()).hexdigest()

        if feature_hash in self.cache:
            return self.cache[feature_hash]

        # Compute prediction
        result = self.classifier.predict(features)
        self.cache[feature_hash] = result

        return result
```

### 2. Batch Predictions

```python
def batch_predict(self, feature_list):
    """Efficient batch prediction for multiple signals."""
    if not feature_list:
        return [], []

    # Combine into single DataFrame
    batch_features = pd.concat(feature_list, ignore_index=True)

    # Single batch prediction
    predictions, probabilities = self.ml_classifier.predict(batch_features)

    # Split back into individual results
    results = []
    batch_size = len(feature_list[0])

    for i in range(len(feature_list)):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        results.append((
            predictions[start_idx:end_idx],
            probabilities[start_idx:end_idx]
        ))

    return results
```

## 🎯 Success Metrics

### 1. Classification Performance
- **Accuracy**: >65% on validation set
- **Precision**: >0.65 for bullish/bearish predictions
- **Recall**: >0.60 for capturing true signals
- **F1-Score**: >0.62 balanced metric

### 2. Trading Performance
- **Sharpe Ratio**: 10-20% improvement over rule-based
- **Win Rate**: Increase in profitable trades
- **False Positive Rate**: Reduction in spurious signals
- **Drawdown**: No increase in maximum drawdown

### 3. Operational Metrics
- **Prediction Latency**: <10ms per prediction
- **Memory Usage**: <100MB additional RAM
- **CPU Usage**: <5% additional CPU during predictions
- **Uptime**: >99.9% availability

## 🚀 Deployment Checklist

- [ ] ML model trained and validated
- [ ] QAOA momentum detector updated with ML integration
- [ ] A/B testing framework implemented
- [ ] Monitoring and alerting configured
- [ ] Rollback procedures documented
- [ ] Team trained on new system
- [ ] Gradual rollout plan executed
- [ ] Performance benchmarks established

---

**Integration Status: Ready for Implementation** ✅

This guide provides everything needed to successfully integrate the Quantum XGBoost Classifier into the IIG quantum trading system.