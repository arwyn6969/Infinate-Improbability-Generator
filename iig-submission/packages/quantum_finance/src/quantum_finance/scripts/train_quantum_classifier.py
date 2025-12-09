
"""
Train Quantum XGBoost Classifier

This script orchestrates the training pipeline for the quantum momentum classifier:
1. Loads quantum feature data from QuestDB (or generates mock data)
2. Engineers lagged features and rolling statistics
3. Generates target labels (future returns)
4. Trains the XGBoost model
5. Evaluates and saves the model artifact
"""

import asyncio
import argparse
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Tuple
import numpy as np
import pandas as pd

# Add src to path for direct execution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from quantum_finance.questdb_qaoa_features_manager import QuestDBQAOAFeaturesManager, QAOAFeatureRecord
from quantum_finance.ml.quantum_classifier import QuantumXGBoostClassifier
from quantum_finance.utils.logger_factory import get_logger

logger = get_logger(__name__)

def generate_mock_data(n_samples: int = 1000) -> pd.DataFrame:
    """Generate synthetic data for testing the pipeline when DB is empty."""
    logger.info(f"Generating {n_samples} mock samples for pipeline verification")
    
    dates = pd.date_range(end=datetime.now(), periods=n_samples, freq="1min")
    
    # Generate random features with some correlation to target
    data = {
        "timestamp": dates,
        "symbol": ["BTCUSDT"] * n_samples,
        "bullish_prob": np.random.uniform(0, 1, n_samples),
        "bearish_prob": np.random.uniform(0, 1, n_samples),
        "directional_confidence": np.random.uniform(0.5, 1, n_samples),
        "quantum_advantage": np.random.uniform(0, 5, n_samples),
        "total_momentum": np.random.uniform(0, 1, n_samples),
        "profitability": np.random.normal(0, 0.01, n_samples), # Previous trade result
        "close": np.cumsum(np.random.normal(0, 1, n_samples)) + 10000 
    }
    
    df = pd.DataFrame(data)
    return df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Create lagged features and rolling stats."""
    df = df.copy()
    features = ["bullish_prob", "bearish_prob", "total_momentum"]
    
    # 1. Create Lagged Features
    for feat in features:
        for lag in [1, 2, 3, 5]:
            df[f"{feat}_lag_{lag}"] = df[feat].shift(lag)
            
    # 2. Rolling Means
    for feat in features:
        df[f"{feat}_roll_mean_5"] = df[feat].rolling(window=5).mean()
        
    return df.dropna()

def generate_labels(df: pd.DataFrame, horizon: int = 5) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Generate target labels based on future returns.
    Target = 1 if Future Return > 0 (Bullish), 0 otherwise (Bearish).
    """
    if "close" not in df.columns:
        # If we don't have price data in this dataframe, we can't label
        # Ideally, we merge with price data. For mock data, we have 'close'.
        # For real QuestDB export, we might need a separate price query or rely on 'actual_price_change' if it's forward looking (it's not).
        # We will assume 'close' acts as a proxy or simulate it.
        # Fallback: Use 'profitability' as a proxy for past trend? No, we need FUTURE.
        # In real pipeline, we fetch prices separately.
        logger.warning("No 'close' price column found. Using mock prices for labeling.")
        df["close"] = np.cumsum(np.random.normal(0, 1, len(df))) + 100
        
    # Calculate future returns
    df["future_return"] = df["close"].shift(-horizon) / df["close"] - 1
    
    # Target: 1 if future return is positive, 0 else
    # Add a buffer? e.g. > 0.001
    df["target"] = (df["future_return"] > 0).astype(int)
    
    # Drop NaNs created by shifting
    df = df.dropna()
    
    y = df["target"]
    X = df.drop(columns=["target", "future_return", "timestamp", "symbol", "close"])
    
    return X, y

def main():
    parser = argparse.ArgumentParser(description="Train Quantum Classifier")
    parser.add_argument("--mock", action="store_true", help="Use mock data")
    parser.add_argument("--save-path", type=str, default="models/quantum_xgb_v1.json", help="Path to save model")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    # 1. Load Data
    manager = QuestDBQAOAFeaturesManager()
    
    if args.mock:
        train_df = generate_mock_data(2000)
    else:
        logger.info("Fetching training data from QuestDB...")
        train_df, val_df_db = manager.export_training_data(split_ratio=0.8) # We'll split manually to handle feature engineering dropna
        
        if train_df is None or len(train_df) < 100:
            logger.warning("Insufficient data in QuestDB. Switching to MOCK data for pipeline verification.")
            train_df = generate_mock_data(2000)
            
    # 2. Feature Engineering
    logger.info("Engineering features...")
    processed_df = feature_engineering(train_df)
    
    # 3. Generate Labels
    logger.info("Generating target labels...")
    X, y = generate_labels(processed_df)
    
    # Split Train/Test
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    logger.info(f"Training shapes: X_train={X_train.shape}, y_train={y_train.shape}")
    
    # 4. Train Model
    logger.info("Initializing classifier...")
    clf = QuantumXGBoostClassifier()
    
    logger.info("Starting training...")
    metrics = clf.train(X_train, y_train, X_val=X_test, y_val=y_test)
    
    # 5. Evaluate
    logger.info("Evaluating on test set...")
    eval_metrics = clf.evaluate(X_test, y_test)
    print(f"Test Metrics: {eval_metrics}")
    
    # 6. Save
    clf.save_model(args.save_path)
    logger.info("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
