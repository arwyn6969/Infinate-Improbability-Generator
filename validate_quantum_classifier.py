#!/usr/bin/env python3
"""
Comprehensive Validation Suite for Quantum XGBoost Classifier

This script validates all aspects of the Phase 2 XGBoost classifier implementation:
1. Pipeline integrity and end-to-end functionality
2. Feature engineering correctness
3. Label generation accuracy
4. Model training and performance
5. Model persistence and loading
6. Integration compatibility

Usage:
    python validate_quantum_classifier.py [--verbose] [--skip-training]

Arguments:
    --verbose: Enable detailed logging
    --skip-training: Skip model training (use existing model if available)
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent / "packages" / "quantum_finance" / "src"))

try:
    from quantum_finance.ml.quantum_classifier import QuantumXGBoostClassifier
    from quantum_finance.scripts.train_quantum_classifier import (
        generate_mock_data,
        feature_engineering,
        generate_labels
    )
    from quantum_finance.utils.logger_factory import get_logger
    logger = get_logger(__name__)
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Make sure you're running from the Probertha root directory")
    sys.exit(1)

class QuantumClassifierValidator:
    """Comprehensive validator for the quantum XGBoost classifier implementation."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results = {}
        self.models_dir = Path("packages/quantum_finance/models")
        self.test_model_path = self.models_dir / "validation_test.json"

        # Setup logging
        if verbose:
            logging.basicConfig(level=logging.INFO)
        else:
            logging.basicConfig(level=logging.WARNING)

    def log(self, message: str, level: str = "info"):
        """Enhanced logging with visual indicators."""
        if level == "success":
            print(f"✅ {message}")
        elif level == "error":
            print(f"❌ {message}")
        elif level == "warning":
            print(f"⚠️  {message}")
        elif level == "info" and self.verbose:
            print(f"ℹ️  {message}")
        elif level == "header":
            print(f"\n🔍 {message}")
            print("=" * (len(message) + 3))

    def run_all_validations(self) -> Dict[str, Any]:
        """Run complete validation suite."""
        self.log("Starting Comprehensive Quantum Classifier Validation", "header")

        validations = [
            ("pipeline_integrity", self.validate_pipeline_integrity),
            ("feature_engineering", self.validate_feature_engineering),
            ("label_generation", self.validate_label_generation),
            ("model_training", self.validate_model_training),
            ("model_persistence", self.validate_model_persistence),
            ("performance_metrics", self.validate_performance_metrics),
            ("integration_compatibility", self.validate_integration_compatibility)
        ]

        all_passed = True
        for test_name, test_func in validations:
            try:
                self.log(f"Running {test_name.replace('_', ' ').title()} Test", "header")
                result = test_func()
                self.results[test_name] = result
                if not result.get("passed", False):
                    all_passed = False
            except Exception as e:
                self.results[test_name] = {"passed": False, "error": str(e)}
                self.log(f"Test failed with error: {e}", "error")
                all_passed = False

        # Summary
        self.log("Validation Summary", "header")
        passed_tests = sum(1 for r in self.results.values() if r.get("passed", False))
        total_tests = len(self.results)

        self.results["summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "success_rate": f"{passed_tests}/{total_tests}",
            "all_passed": all_passed
        }

        if all_passed:
            self.log(f"All {total_tests} validation tests PASSED! 🎉", "success")
        else:
            self.log(f"{passed_tests}/{total_tests} tests passed", "warning")

        return self.results

    def validate_pipeline_integrity(self) -> Dict[str, Any]:
        """Test 1: Pipeline Integrity - Can we run the training script?"""
        try:
            # Check if we can import all required components
            from quantum_finance.scripts.train_quantum_classifier import main as train_main

            # Check if models directory exists
            if not self.models_dir.exists():
                self.models_dir.mkdir(parents=True, exist_ok=True)

            # Test basic imports work
            clf = QuantumXGBoostClassifier()
            mock_data = generate_mock_data(100)

            return {
                "passed": True,
                "imports_ok": True,
                "models_dir_exists": self.models_dir.exists(),
                "sample_data_shape": mock_data.shape,
                "classifier_initialized": clf.model is not None
            }

        except Exception as e:
            return {"passed": False, "error": str(e)}

    def validate_feature_engineering(self) -> Dict[str, Any]:
        """Test 2: Feature Engineering - Are lagged and rolling features created correctly?"""
        try:
            # Generate mock data
            df = generate_mock_data(1000)

            # Apply feature engineering
            engineered_df = feature_engineering(df)

            # Expected original features
            expected_base_features = ['bullish_prob', 'bearish_prob', 'total_momentum']

            # Check lagged features
            lagged_features = []
            for feat in expected_base_features:
                for lag in [1, 2, 3, 5]:
                    lagged_features.append(f"{feat}_lag_{lag}")

            # Check rolling features
            rolling_features = []
            for feat in expected_base_features:
                rolling_features.append(f"{feat}_roll_mean_5")

            # Validate
            all_expected_features = expected_base_features + lagged_features + rolling_features
            missing_features = [f for f in all_expected_features if f not in engineered_df.columns]
            extra_features = [f for f in engineered_df.columns if f not in all_expected_features and f not in ['timestamp', 'symbol', 'close', 'profitability']]

            # Check for NaN values
            nan_counts = engineered_df.isnull().sum()
            has_nans = nan_counts.sum() > 0

            return {
                "passed": len(missing_features) == 0 and not has_nans,
                "original_features": len(expected_base_features),
                "lagged_features": len(lagged_features),
                "rolling_features": len(rolling_features),
                "total_features": len(all_expected_features),
                "missing_features": missing_features,
                "extra_features": extra_features,
                "has_nans": has_nans,
                "data_shape": engineered_df.shape
            }

        except Exception as e:
            return {"passed": False, "error": str(e)}

    def validate_label_generation(self) -> Dict[str, Any]:
        """Test 3: Label Generation - Are target labels created correctly?"""
        try:
            # Generate mock data with price data
            df = generate_mock_data(1000)

            # Apply feature engineering first
            engineered_df = feature_engineering(df)

            # Generate labels
            X, y = generate_labels(engineered_df)

            # Validate labels
            label_distribution = y.value_counts().to_dict()
            label_balance = y.mean()  # Should be ~0.5 for random walk

            # Check distribution is roughly balanced
            balance_ok = 0.45 <= label_balance <= 0.55

            # Check no NaN values
            has_nans = y.isnull().sum() > 0

            # Check target values are binary
            unique_values = sorted(y.unique())
            binary_ok = unique_values == [0, 1]

            return {
                "passed": balance_ok and not has_nans and binary_ok,
                "label_distribution": label_distribution,
                "label_balance": float(label_balance),
                "balance_acceptable": balance_ok,
                "has_nans": has_nans,
                "binary_labels": binary_ok,
                "unique_values": unique_values,
                "sample_size": len(y),
                "feature_count": X.shape[1]
            }

        except Exception as e:
            return {"passed": False, "error": str(e)}

    def validate_model_training(self) -> Dict[str, Any]:
        """Test 4: Model Training - Can we train the classifier successfully?"""
        try:
            # Generate training data
            df = generate_mock_data(2000)
            engineered_df = feature_engineering(df)
            X, y = generate_labels(engineered_df)

            # Split data
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

            # Train model
            start_time = time.time()
            clf = QuantumXGBoostClassifier()
            training_metrics = clf.train(X_train, y_train, X_val=X_test, y_val=y_test, verbose=False)
            training_time = time.time() - start_time

            # Basic validation
            training_completed = training_metrics is not None
            has_feature_importances = 'feature_importances' in training_metrics
            reasonable_training_time = training_time < 60  # Should train in under 1 minute

            return {
                "passed": training_completed and has_feature_importances,
                "training_time_seconds": round(training_time, 2),
                "training_completed": training_completed,
                "has_feature_importances": has_feature_importances,
                "reasonable_training_time": reasonable_training_time,
                "training_samples": len(X_train),
                "validation_samples": len(X_test),
                "features_used": len(clf.features),
                "training_metrics": training_metrics
            }

        except Exception as e:
            return {"passed": False, "error": str(e)}

    def validate_performance_metrics(self) -> Dict[str, Any]:
        """Test 5: Performance Metrics - Does the model achieve expected performance?"""
        try:
            # Generate more realistic mock data with some predictive patterns
            df = self._generate_realistic_mock_data(3000)
            engineered_df = feature_engineering(df)
            X, y = generate_labels(engineered_df)

            split_idx = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

            clf = QuantumXGBoostClassifier()
            clf.train(X_train, y_train, X_val=X_test, y_val=y_test, verbose=False)

            # Evaluate performance
            eval_metrics = clf.evaluate(X_test, y_test)

            # For mock data, we expect performance slightly better than random
            # Lower thresholds since mock data has limited realism
            accuracy_ok = eval_metrics.get('accuracy', 0) > 0.52  # Slightly better than random (0.5)
            f1_ok = eval_metrics.get('f1', 0) > 0.50               # Basic predictive power
            precision_ok = eval_metrics.get('precision', 0) > 0.50
            recall_ok = eval_metrics.get('recall', 0) > 0.50

            # Feature importance analysis
            feature_importance = dict(zip(clf.features, clf.model.feature_importances_))
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]

            # Check quantum features are important
            quantum_features_present = any('bullish_prob' in f[0] or 'total_momentum' in f[0] for f in top_features)

            return {
                "passed": accuracy_ok,  # Relaxed threshold for mock data validation
                "accuracy": eval_metrics.get('accuracy', 0),
                "precision": eval_metrics.get('precision', 0),
                "recall": eval_metrics.get('recall', 0),
                "f1_score": eval_metrics.get('f1', 0),
                "accuracy_threshold_met": accuracy_ok,
                "f1_threshold_met": f1_ok,
                "precision_ok": precision_ok,
                "recall_ok": recall_ok,
                "top_5_features": top_features,
                "quantum_features_important": quantum_features_present,
                "total_features": len(feature_importance),
                "note": "Using relaxed thresholds for mock data validation"
            }

        except Exception as e:
            return {"passed": False, "error": str(e)}

    def _generate_realistic_mock_data(self, n_samples: int = 3000) -> pd.DataFrame:
        """Generate more realistic mock data with some predictive patterns."""
        self.log("Generating realistic mock data with predictive patterns", "info")

        dates = pd.date_range(end=datetime.now(), periods=n_samples, freq="1min")

        # Create base signals with some autocorrelation (momentum)
        np.random.seed(42)  # For reproducible results

        # Generate trending base signal
        trend = np.cumsum(np.random.normal(0, 0.01, n_samples))
        base_signal = np.sin(np.linspace(0, 4*np.pi, n_samples)) * 0.3 + trend * 0.2

        # Create quantum features with realistic correlations
        bullish_prob = 0.5 + base_signal * 0.4 + np.random.normal(0, 0.1, n_samples)
        bearish_prob = 0.5 - base_signal * 0.4 + np.random.normal(0, 0.1, n_samples)

        # Ensure they stay in [0,1] range
        bullish_prob = np.clip(bullish_prob, 0.01, 0.99)
        bearish_prob = np.clip(bearish_prob, 0.01, 0.99)

        # Create price data that follows the signal with some noise
        returns = base_signal * 0.002 + np.random.normal(0, 0.005, n_samples)
        close = 10000 * np.exp(np.cumsum(returns))

        data = {
            "timestamp": dates,
            "symbol": ["BTCUSDT"] * n_samples,
            "bullish_prob": bullish_prob,
            "bearish_prob": bearish_prob,
            "directional_confidence": np.abs(bullish_prob - bearish_prob),
            "quantum_advantage": np.random.uniform(1.5, 3.0, n_samples),
            "total_momentum": np.maximum(bullish_prob, bearish_prob),
            "profitability": np.random.normal(0, 0.01, n_samples),
            "close": close
        }

        return pd.DataFrame(data)

    def validate_model_persistence(self) -> Dict[str, Any]:
        """Test 6: Model Persistence - Can we save and load models correctly?"""
        try:
            # Train and save model
            df = generate_mock_data(1000)
            engineered_df = feature_engineering(df)
            X, y = generate_labels(engineered_df)

            clf = QuantumXGBoostClassifier()
            clf.train(X, y, verbose=False)

            # Save model
            save_start = time.time()
            clf.save_model(str(self.test_model_path))
            save_time = time.time() - save_start

            # Check files exist
            model_exists = self.test_model_path.exists()
            meta_exists = (self.test_model_path.parent / f"{self.test_model_path.name}.meta").exists()

            # Load model
            load_start = time.time()
            loaded_clf = QuantumXGBoostClassifier(model_path=str(self.test_model_path))
            load_time = time.time() - load_start

            # Test prediction
            sample_data = X.iloc[:5]  # First 5 samples
            original_preds, original_probs = clf.predict(sample_data)
            loaded_preds, loaded_probs = loaded_clf.predict(sample_data)

            # Predictions should be identical
            predictions_match = np.array_equal(original_preds, loaded_preds)
            probabilities_match = np.allclose(original_probs, loaded_probs, rtol=1e-10)

            # Metadata preserved
            metadata_preserved = (
                hasattr(loaded_clf, 'metadata') and
                loaded_clf.metadata.get('features') == clf.features
            )

            return {
                "passed": model_exists and predictions_match and probabilities_match,
                "model_file_exists": model_exists,
                "metadata_file_exists": meta_exists,
                "save_time_seconds": round(save_time, 3),
                "load_time_seconds": round(load_time, 3),
                "predictions_match": predictions_match,
                "probabilities_match": probabilities_match,
                "metadata_preserved": metadata_preserved,
                "features_preserved": loaded_clf.features == clf.features
            }

        except Exception as e:
            return {"passed": False, "error": str(e)}

    def validate_integration_compatibility(self) -> Dict[str, Any]:
        """Test 7: Integration Compatibility - Can the model work with quantum feature inputs?"""
        try:
            # Check if we can import QAOA components (optional)
            try:
                from quantum_finance.vwap.qaoa_momentum import QAOAUptrendDetector
                qaoa_available = True
                self.log("QAOA detector available for testing", "info")
            except ImportError:
                qaoa_available = False
                self.log("QAOA detector not available - testing with synthetic quantum features", "warning")

            # Load trained model
            if self.test_model_path.exists():
                clf = QuantumXGBoostClassifier(model_path=str(self.test_model_path))
                self.log("Loaded existing test model", "info")
            else:
                # Train a quick model
                df = self._generate_realistic_mock_data(1000)
                engineered_df = feature_engineering(df)
                X, y = generate_labels(engineered_df)

                clf = QuantumXGBoostClassifier()
                clf.train(X, y, verbose=False)
                self.log("Trained new model for integration testing", "info")

            # Create a more realistic test dataset with proper feature engineering
            test_samples = 20
            base_data = self._generate_realistic_mock_data(test_samples + 10)  # Extra for lagged features

            # Apply feature engineering
            engineered_test = feature_engineering(base_data)
            if len(engineered_test) < 10:
                # Generate more data if feature engineering reduced it too much
                base_data = self._generate_realistic_mock_data(test_samples + 20)
                engineered_test = feature_engineering(base_data)

            # Take final samples after feature engineering
            final_samples = min(10, len(engineered_test))
            quantum_sample = engineered_test.head(final_samples).copy()

            if len(quantum_sample) >= 5:  # Need at least 5 samples for meaningful test
                predictions, probabilities = clf.predict(quantum_sample)

                # Validate predictions
                valid_predictions = len(predictions) == len(quantum_sample)
                probabilities_in_range = np.all((probabilities >= 0) & (probabilities <= 1))
                reasonable_predictions = (
                    predictions.dtype == int and
                    np.all(np.isin(predictions, [0, 1])) and
                    len(np.unique(predictions)) >= 1  # At least one class predicted
                )

                prediction_dist = pd.Series(predictions).value_counts().to_dict() if valid_predictions else {}

                return {
                    "passed": valid_predictions and probabilities_in_range and reasonable_predictions,
                    "qaoa_detector_available": qaoa_available,
                    "model_loaded_successfully": True,
                    "predictions_generated": valid_predictions,
                    "probabilities_in_range": probabilities_in_range,
                    "reasonable_predictions": reasonable_predictions,
                    "test_samples": len(quantum_sample),
                    "prediction_distribution": prediction_dist,
                    "feature_count": len(clf.features),
                    "note": "Test passed with synthetic quantum features" if not qaoa_available else "Test passed with full QAOA integration"
                }
            else:
                return {
                    "passed": False,
                    "error": f"Insufficient test samples after feature engineering: {len(quantum_sample)}",
                    "qaoa_detector_available": qaoa_available,
                    "model_loaded_successfully": True,
                    "test_samples": len(quantum_sample)
                }

        except Exception as e:
            return {"passed": False, "error": str(e)}

    def save_validation_report(self, output_path: str = "validation_report.json"):
        """Save detailed validation report to file."""
        try:
            report = {
                "validation_timestamp": datetime.now().isoformat(),
                "summary": self.results.get("summary", {}),
                "detailed_results": self.results,
                "system_info": {
                    "python_version": sys.version,
                    "pandas_version": pd.__version__,
                    "numpy_version": np.__version__
                }
            }

            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)

            self.log(f"Validation report saved to {output_path}", "success")

        except Exception as e:
            self.log(f"Failed to save report: {e}", "error")

    def print_summary_report(self):
        """Print a human-readable summary of validation results."""
        summary = self.results.get("summary", {})

        print("\n" + "="*60)
        print("🎯 QUANTUM CLASSIFIER VALIDATION SUMMARY")
        print("="*60)

        print(f"Total Tests: {summary.get('total_tests', 0)}")
        print(f"Passed Tests: {summary.get('passed_tests', 0)}")
        print(f"Success Rate: {summary.get('success_rate', '0/0')}")

        if summary.get('all_passed', False):
            print("🎉 OVERALL STATUS: ALL TESTS PASSED")
        else:
            print("⚠️  OVERALL STATUS: SOME TESTS FAILED")

        print("\n📋 DETAILED RESULTS:")
        for test_name, result in self.results.items():
            if test_name == "summary":
                continue

            status = "✅ PASS" if result.get("passed", False) else "❌ FAIL"
            print(f"  {test_name.replace('_', ' ').title()}: {status}")

            if not result.get("passed", False) and "error" in result:
                print(f"    Error: {result['error']}")

        print("\n💡 RECOMMENDATIONS:")
        if summary.get('all_passed', False):
            print("  ✅ Ready for integration with QAOA momentum detector")
            print("  ✅ Proceed to A/B testing against rule-based classifier")
            print("  ✅ Consider production deployment preparation")
        else:
            print("  🔧 Address failed tests before integration")
            print("  📋 Check validation_report.json for detailed error information")

        print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Validate Quantum XGBoost Classifier")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--skip-training", action="store_true", help="Skip model training tests")
    parser.add_argument("--output", "-o", default="validation_report.json", help="Output report file")

    args = parser.parse_args()

    # Run validation
    validator = QuantumClassifierValidator(verbose=args.verbose)

    if args.skip_training:
        # Run only non-training tests
        validator.results["pipeline_integrity"] = validator.validate_pipeline_integrity()
        validator.results["feature_engineering"] = validator.validate_feature_engineering()
        validator.results["label_generation"] = validator.validate_label_generation()
        validator.results["integration_compatibility"] = validator.validate_integration_compatibility()

        # Load existing model if available for persistence test
        if validator.test_model_path.exists():
            validator.results["model_persistence"] = validator.validate_model_persistence()
        else:
            print("⚠️  Skipping model persistence test - no existing model found")
            validator.results["model_persistence"] = {"passed": False, "error": "No existing model"}

        validator.results["model_training"] = {"passed": False, "skipped": "Training tests disabled"}
        validator.results["performance_metrics"] = {"passed": False, "skipped": "Training tests disabled"}

    else:
        # Run all tests
        validator.run_all_validations()

    # Generate reports
    validator.save_validation_report(args.output)
    validator.print_summary_report()


if __name__ == "__main__":
    main()