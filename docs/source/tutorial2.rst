LESS is More: A Comprehensive Tutorial
=======================================

**LESS (Learning with Subset Stacking)** is a scalable and versatile ensemble learning framework. It constructs an ensemble of local models trained on subsets of data and combines their predictions using a global meta-estimator.

In this tutorial, we will explore the two main variants of LESS:

1.  **LESS-A (Averaging):** Trains multiple iterations of local/global models and averages their predictions.
2.  **LESS-B (Boosting):** Trains models sequentially, where each stage learns to correct the residuals of the previous stage.

We will also dive deep into the critical parameters that control the behavior of LESS, such as ``n_subsets``, ``n_estimators``, ``min_neighbors``, and the choice of estimators.

.. code-block:: python

    import numpy as np
    import pandas as pd

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.datasets import fetch_openml
    from sklearn.metrics import mean_squared_error
    from sklearn.tree import DecisionTreeRegressor

    from less import LESSARegressor, LESSBRegressor

Data Preparation
----------------

We will use the **Abalone** dataset for this tutorial. It contains physical measurements of abalones, and the goal is to predict the age (number of rings).

.. code-block:: python

    abalone = fetch_openml(name="abalone", version=1, as_frame=True)

    X = pd.get_dummies(abalone.data, drop_first=True, dtype=np.float32)
    y = abalone.target.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")

Output:

.. code-block:: text

    Training set size: (3341, 9)
    Test set size: (836, 9)

1. LESS-A (Averaging)
---------------------

**LESS-A** (``LESSARegressor``) is the averaging variant. It performs multiple independent iterations. In each iteration, it selects subsets of data, trains local models, and optionally trains a global model to combine them. The final prediction is the average of predictions from all iterations.

Key Parameter: ``n_estimators``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For LESS-A, ``n_estimators`` controls the number of averaging iterations.

*   **Default:** 100
*   **Effect:** Generally, more estimators lead to more stable predictions (lower variance) but increase training time linearly.

.. code-block:: python

    # LESS-A with default parameters (n_estimators=100)
    less_a = LESSARegressor(n_estimators=100, random_state=42)
    less_a.fit(X_train, y_train)
    y_pred_a = less_a.predict(X_test)
    print(f'LESS-A Test MSE: {mean_squared_error(y_test, y_pred_a):.4f}')

Output:

.. code-block:: text

    LESS-A Test MSE: 4.3957

2. LESS-B (Boosting)
--------------------

**LESS-B** (``LESSBRegressor``) applies a boosting strategy. Instead of independent iterations, it trains models sequentially. Each stage learns to correct the residuals (errors) of the previous stage.

Key Parameters: ``n_estimators`` and ``learning_rate``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*   **``n_estimators``**: The number of boosting stages (default: 100).
*   **``learning_rate``**: Shrinks the contribution of each estimator (default: 0.1). There is a trade-off between ``learning_rate`` and ``n_estimators``.

.. code-block:: python

    less_b = LESSBRegressor(
        n_estimators=50,       # 50 boosting stages
        learning_rate=0.1,     # Shrinkage parameter
        random_state=42
    )
    less_b.fit(X_train, y_train)
    y_pred_b = less_b.predict(X_test)
    print(f'LESS-B Test MSE: {mean_squared_error(y_test, y_pred_b):.4f}')

Output:

.. code-block:: text

    LESS-B Test MSE: 4.3576

3. Critical Parameters Deep Dive
--------------------------------

To get the most out of LESS, it's essential to understand its key parameters.

Number of Subsets (``n_subsets``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is perhaps the most critical parameter. It determines how many local subsets are created in each iteration (or stage).

*   **Default:** 20
*   **Effect:** A higher ``n_subsets`` means more local models, which can capture more local details but increases computational cost. It also affects the number of neighbors used for training each local model (see ``min_neighbors`` below).

.. code-block:: python

    # Experimenting with n_subsets
    for n in [5, 20, 50]:
        model = LESSARegressor(n_subsets=n, random_state=42)
        model.fit(X_train, y_train)
        mse = mean_squared_error(y_test, model.predict(X_test))
        print(f'n_subsets={n}: MSE={mse:.4f}')

Output:

.. code-block:: text

    n_subsets=5: MSE=4.4578
    n_subsets=20: MSE=4.3957
    n_subsets=50: MSE=4.3848

Minimum Neighbors (``min_neighbors``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This parameter ensures that each local model has enough data to train on.

*   **Default:** 10
*   **Internal Logic:** LESS automatically calculates the number of neighbors (``n_neighbors``) for each subset based on the dataset size (``n_samples``) and ``n_subsets``.

.. math::

    \text{suggested_neighbors} = \max(\text{min_neighbors}, \lfloor \frac{\text{n_samples}}{\text{n_subsets}} \rfloor)

    \text{n_neighbors} = \min(\text{suggested_neighbors}, \text{n_samples})

This logic ensures that even if you have many subsets, each local model will see at least ``min_neighbors`` samples (overlapping if necessary). If ``n_subsets`` is small, the local models will see more data (``n_samples / n_subsets``).

.. code-block:: python

    # Experimenting with min_neighbors
    # This allows min_neighbors to actually control the subset size.
    for n in [50, 500, 1000]:
        model = LESSARegressor(n_subsets=10, min_neighbors=n, random_state=42)
        model.fit(X_train, y_train)
        mse = mean_squared_error(y_test, model.predict(X_test))
        print(f'min_neighbors={n}: MSE={mse:.4f}')

Output:

.. code-block:: text

    min_neighbors=50: MSE=4.4130
    min_neighbors=500: MSE=4.4186
    min_neighbors=1000: MSE=4.4224

Local Estimator (``local_estimator``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This defines the model used for each local subset.

*   **Default:** ``'linear'`` (uses ``LinearRegression``)
*   **Options:**
    *   ``'linear'``: Standard Linear Regression.
    *   ``'tree'``: A ``DecisionTreeRegressor`` with specific parameters (max_leaf_nodes=31, etc.).
    *   **Custom:** You can pass any callable that returns a scikit-learn compatible regressor (e.g., ``lambda: SVR()``).

.. code-block:: python

    # Using 'tree' as local estimator
    less_tree_local = LESSARegressor(local_estimator='tree', random_state=42)
    less_tree_local.fit(X_train, y_train)
    print(f'Local=Tree MSE: {mean_squared_error(y_test, less_tree_local.predict(X_test)):.4f}')

    # Using a custom local estimator
    less_custom_local = LESSARegressor(local_estimator=lambda: DecisionTreeRegressor(max_depth=5), random_state=42)
    less_custom_local.fit(X_train, y_train)
    print(f'Local=CustomTree MSE: {mean_squared_error(y_test, less_custom_local.predict(X_test)):.4f}')

Output:

.. code-block:: text

    Local=Tree MSE: 5.0080
    Local=CustomTree MSE: 4.9919

Global Estimator (``global_estimator``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The global estimator combines the predictions of the local models.

*   **Default:** ``'xgboost'`` (uses ``XGBRFRegressor``)
*   **Options:**
    *   ``'xgboost'``: Random Forest regressor from XGBoost.
    *   ``None``: Removes the global estimator. The final prediction becomes a weighted average of local predictions.
    *   **Custom:** Any callable returning a regressor (e.g., ``lambda: RandomForestRegressor()``).

.. code-block:: python

    # Using default (XGBoost)
    less_default = LESSARegressor(random_state=42)
    less_default.fit(X_train, y_train)
    print(f'Global=XGBoost MSE: {mean_squared_error(y_test, less_default.predict(X_test)):.4f}')

    # Removing global estimator (Weighted Average)
    less_no_global = LESSARegressor(global_estimator=None, random_state=42)
    less_no_global.fit(X_train, y_train)
    print(f'No Global MSE: {mean_squared_error(y_test, less_no_global.predict(X_test)):.4f}')

Output:

.. code-block:: text

    Global=XGBoost MSE: 4.3957
    No Global MSE: 6.1603

Validation Split (``val_size``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can split the dataset into training and validation sets within LESS.

*   **Purpose:** The training set is used to train the **local estimators**, while the validation set is used to train the **global estimator**. This can help prevent overfitting, especially when the global estimator is powerful.
*   **Usage:** Set ``val_size`` to a float between 0 and 1 (e.g., ``0.2`` for 20% validation data).

.. code-block:: python

    less_val = LESSARegressor(val_size=0.2, random_state=42)
    less_val.fit(X_train, y_train)
    y_pred_val = less_val.predict(X_test)
    print(f'Test error (val_size=0.2): {mean_squared_error(y_test, y_pred_val):.4f}')

Output:

.. code-block:: text

    Test error (val_size=0.2): 4.3987

Clustering Method (``cluster_method``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This parameter controls how the centers of the subsets are selected.

*   **Default:** ``'tree'`` (Random sampling). It selects ``n_subsets`` centers randomly from the data.
*   **Options:**
    *   ``'tree'``: Random sampling.
    *   ``'kmeans'``: Uses K-Means clustering. **Crucially, the number of clusters is set equal to** ``n_subsets``. The cluster centers found by K-Means become the centers of the subsets.

.. code-block:: python

    # Using K-Means for clustering
    # Here, n_subsets=20 means K-Means will find 20 cluster centers
    less_kmeans = LESSARegressor(cluster_method='kmeans', n_subsets=20, random_state=42)
    less_kmeans.fit(X_train, y_train)
    print(f'Cluster=KMeans MSE: {mean_squared_error(y_test, less_kmeans.predict(X_test)):.4f}')

Output:

.. code-block:: text

    Cluster=KMeans MSE: 4.3866

Random State (``random_state``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Controls the randomness of the algorithm (subset selection, local estimator initialization, global estimator initialization). Setting this ensures reproducibility of your results.
