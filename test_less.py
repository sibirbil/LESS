import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from less import LESSGBRegressor, LESSAVRegressor

def run_tests():
    """Runs a series of tests for the LESS regressors."""
    print('--- Generating Synthetic Regression Data ---')
    X, y = make_regression(n_samples=10000, n_features=15, noise=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print('Data generated successfully.')
    print(f'Train set size: {X_train.shape}')
    print(f'Test set size: {X_test.shape}')

    models_to_test = {
        'LESSGBRegressor': LESSGBRegressor,
        'LESSAVRegressor': LESSAVRegressor
    }

    for model_name, model_class in models_to_test.items():
        print(f'\n--- Testing {model_name} ---')

        # Test 1: Default parameters (cluster_method='tree')
        try:
            print('\n1. Testing with default parameters (cluster_method=\'tree\')...')
            model_default = model_class(random_state=42, kernel_coeff=None)
            model_default.fit(X_train, y_train)
            predictions_default = model_default.predict(X_test)
            mse_default = mean_squared_error(y_test, predictions_default)
            print(f'  -> SUCCESS: Model trained and predicted. MSE: {mse_default:.4f}')
        except Exception as e:
            print(f'  -> FAILED: An error occurred: {e}')

        # Test 2: KMeans clustering
        try:
            print("\n2. Testing with cluster_method='kmeans'...")
            model_kmeans = model_class(cluster_method='kmeans', n_subsets=10, random_state=42)
            model_kmeans.fit(X_train, y_train)
            predictions_kmeans = model_kmeans.predict(X_test)
            mse_kmeans = mean_squared_error(y_test, predictions_kmeans)
            print(f'  -> SUCCESS: Model trained and predicted. MSE: {mse_kmeans:.4f}')
        except Exception as e:
            print(f'  -> FAILED: An error occurred: {e}')

        # Test 3: Train/validation split with val_size
        try:
            print('\n3. Testing with validation set (val_size=0.5)...')
            model_val = model_class(val_size=0.5, n_subsets=10, random_state=42)
            model_val.fit(X_train, y_train)
            predictions_val = model_val.predict(X_test)
            mse_val = mean_squared_error(y_test, predictions_val)
            print(f'  -> SUCCESS: Model trained and predicted. MSE: {mse_val:.4f}')
        except Exception as e:
            print(f'  -> FAILED: An error occurred: {e}')

    print('\n--- All Tests Completed ---')

if __name__ == '__main__':
    run_tests()