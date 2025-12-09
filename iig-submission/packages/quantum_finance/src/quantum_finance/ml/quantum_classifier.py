
"""
Quantum XGBoost Classifier

This module implements a wrapper around XGBoost for classifying market direction
based on quantum features extracted from QAOA/VQE algorithms.

It manages:
1. Model initialization and configuration
2. Feature preprocessing (handling quantum probabilities)
3. Training and validation
4. Model persistence
5. Prediction with confidence scoring
"""

import json
import logging
import os
import pickle
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit

from ..utils.logger_factory import get_logger

logger = get_logger(__name__)


class QuantumXGBoostClassifier:
    """
    XGBoost classifier optimized for quantum feature sets.
    
    Attributes:
        model: Underlying XGBoost model
        features: List of feature names used for training
        params: XGBoost hyperparameters
    """

    DEFAULT_PARAMS = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "eta": 0.05,
        "max_depth": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_estimators": 100,
        "random_state": 42,
        "tree_method": "hist",  # Faster training
    }

    QUANTUM_FEATURES = [
        "bullish_prob",
        "bearish_prob", 
        "directional_confidence",
        "quantum_advantage",
        "total_momentum"
    ]

    def __init__(self, params: Optional[Dict[str, Any]] = None, model_path: Optional[str] = None):
        """
        Initialize the classifier.

        Args:
            params: XGBoost hyperparameters (overrides defaults)
            model_path: Path to load a pre-trained model from
        """
        self.params = self.DEFAULT_PARAMS.copy()
        if params:
            self.params.update(params)
        
        self.features = self.QUANTUM_FEATURES.copy()
        self.model = None
        self.metadata = {}
        
        if model_path:
            self.load_model(model_path)
        else:
            self.model = xgb.XGBClassifier(**self.params)

    def train(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series, 
        X_val: Optional[pd.DataFrame] = None, 
        y_val: Optional[pd.Series] = None,
        early_stopping_rounds: int = 10,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Train the classifier.

        Args:
            X_train: Training features
            y_train: Training labels (0=Bearish, 1=Bullish)
            X_val: Validation features
            y_val: Validation labels
            early_stopping_rounds: Rounds for early stopping
            verbose: Print training progress

        Returns:
            Dictionary identifying training metrics
        """
        logger.info(f"Starting training with {len(X_train)} samples")
        
        # Ensure only relevant features are used
        train_cols = [col for col in X_train.columns if col in self.features or "lag_" in col or "roll_" in col]
        self.features = train_cols # Update features list to include engineered features if present
        
        X_train_filtered = X_train[self.features]
        
        eval_set = []
        if X_val is not None and y_val is not None:
            X_val_filtered = X_val[self.features]
            eval_set = [(X_train_filtered, y_train), (X_val_filtered, y_val)]
        
        try:
            self.model.fit(
                X_train_filtered, 
                y_train,
                eval_set=eval_set,
                verbose=verbose
            )
            
            # Record training metadata
            self.metadata = {
                "train_date": datetime.now().isoformat(),
                "n_samples": len(X_train),
                "features": self.features,
                "best_iteration": self.model.best_iteration if hasattr(self.model, "best_iteration") else -1,
                "feature_importances": dict(zip(self.features, [float(x) for x in self.model.feature_importances_]))
            }
            
            logger.info("Training completed successfully")
            return self.metadata
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions and confidence scores.

        Args:
            X: Feature dataframe

        Returns:
            Tuple containing:
            - predictions: Array of class labels (0/1)
            - probabilities: Array of probabilities for the positive class (1)
        """
        if not self.model:
            raise ValueError("Model not trained or loaded")
            
        # Ensure features match training features
        # Handle missing columns safely if possible, or raise error
        missing_cols = set(self.features) - set(X.columns)
        if missing_cols:
            raise ValueError(f"Input features missing required columns: {missing_cols}")
            
        X_filtered = X[self.features]
        
        probs = self.model.predict_proba(X_filtered)[:, 1]
        preds = self.model.predict(X_filtered)
        
        return preds, probs

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary of metrics
        """
        preds, probs = self.predict(X_test)
        
        metrics = {
            "accuracy": float(accuracy_score(y_test, preds)),
            "precision": float(precision_score(y_test, preds, zero_division=0)),
            "recall": float(recall_score(y_test, preds, zero_division=0)),
            "f1": float(f1_score(y_test, preds, zero_division=0)),
        }
        
        logger.info(f"Evaluation Metrics: {metrics}")
        return metrics

    def save_model(self, filepath: str) -> None:
        """
        Save model to disk.

        Args:
            filepath: Path to save the model (should be .json or .pkl)
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save using native XGBoost save if json, else pickle for full wrapper persistence
            if filepath.endswith(".json"):
                 self.model.save_model(filepath)
                 # Save metadata separately
                 meta_path = filepath + ".meta"
                 with open(meta_path, "w") as f:
                     json.dump(self.metadata, f, indent=2)
            else:
                with open(filepath, "wb") as f:
                    pickle.dump(self, f)
            
            logger.info(f"Model saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise

    def load_model(self, filepath: str) -> None:
        """
        Load model from disk.
        
        Args:
            filepath: Path to model file
        """
        try:
            if filepath.endswith(".json"):
                self.model = xgb.XGBClassifier()
                self.model.load_model(filepath)
                # Try load metadata
                meta_path = filepath + ".meta"
                if os.path.exists(meta_path):
                     with open(meta_path, "r") as f:
                         self.metadata = json.load(f)
                     self.features = self.metadata.get("features", self.QUANTUM_FEATURES)
            else:
                with open(filepath, "rb") as f:
                    obj = pickle.load(f)
                    self.__dict__ = obj.__dict__
                    
            logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
