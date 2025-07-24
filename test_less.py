#!/usr/bin/env python3
"""
Test script for LESS Regressor and Classifier
"""

import numpy as np
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from less import LESSRegressor, LESSClassifier

def test_regressor():
    """Test LESSRegressor"""
    print("Testing LESSRegressor...")
    
    # Generate synthetic regression data
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Test different local estimators
    estimators = ["linear", "ridge", "tree"]
    
    for est_type in estimators:
        print(f"\nTesting with local_estimator='{est_type}'")
        
        # Create and fit model
        model = LESSRegressor(
            n_subsets=10,
            n_estimators=50,
            learning_rate=0.1,
            local_estimator=est_type,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        print(f"MSE: {mse:.4f}")
        
        # Test with n_rounds parameter
        y_pred_limited = model.predict(X_test, n_rounds=25)
        mse_limited = mean_squared_error(y_test, y_pred_limited)
        print(f"MSE (25 rounds): {mse_limited:.4f}")

def test_classifier():
    """Test LESSClassifier"""
    print("\n" + "="*50)
    print("Testing LESSClassifier...")
    
    # Generate synthetic classification data
    X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, 
                              n_informative=8, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Test different local estimators
    estimators = ["logistic", "tree"]
    
    for est_type in estimators:
        print(f"\nTesting with local_estimator='{est_type}'")
        
        # Create and fit model
        model = LESSClassifier(
            n_subsets=10,
            n_estimators=50,
            learning_rate=0.1,
            local_estimator=est_type,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Probability shape: {y_proba.shape}")
        print(f"Probability range: [{y_proba.min():.4f}, {y_proba.max():.4f}]")
        
        # Test with n_rounds parameter
        y_pred_limited = model.predict(X_test, n_rounds=25)
        accuracy_limited = accuracy_score(y_test, y_pred_limited)
        print(f"Accuracy (25 rounds): {accuracy_limited:.4f}")

if __name__ == "__main__":
    test_regressor()
    test_classifier()
    print("\nAll tests completed!") 