# Learning with Subset Stacking (LESS)

LESS is a new supervised learning algorithm that is based on training many local estimators on subsets of a given dataset, and then passing their predictions to a global estimator.

## Installation

`pip install git+https://github.com/sibirbil/LESS.git`

## Testing

Here is how you can use LESS for regression (we are working on classification):

```python
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from less import LESSRegressor

# Synthetic dataset (X, y)
xvals = np.arange(-10, 10, 0.1) # domain
num_of_samples = 200
X = np.zeros((num_of_samples, 1))
y = np.zeros(num_of_samples)
for i in range(num_of_samples):
    xran = -10 + 20*np.random.rand()
    X[i] = xran
    y[i] = 10*np.sin(xran) + 2.5*np.random.randn()

# Train and test split
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.3)

# LESS fit() & predict()
LESS_model = LESSRegressor()
LESS_model.fit(X_train, y_train)
y_pred = LESS_model.predict(X_test)
print('Test error of LESS: {0:.2f}'.format(mean_squared_error(y_pred, y_test)))
```

## Tutorials

Our **two-part** [tutorial](https://colab.research.google.com/drive/183MRHH-i4XT3-HepHbIKVRPiwH7uMzrw?usp=sharing) aims at getting you familiar with LESS. If you want to try the tutorials on your own computer, then you also need to install the following additional packages: `pandas`, `matplotlib`, and `seaborn`.
