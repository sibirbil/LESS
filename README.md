# Learning with Subset Stacking (LESS)

LESS is a supervised learning algorithm that is based on training many local estimators on subsets of a given dataset, and then passing their predictions to a global estimator. You can find the details about LESS in our [manuscript](https://arxiv.org/abs/2112.06251).

![LESS](./img/LESS1Level.png)

## Installation

`pip install less-learn`

or

``conda install -c conda-forge less-learn``

(see also [conda-smithy repository](https://github.com/conda-forge/less-learn-feedstock))

## Testing

Here is how you can use LESS:

```python
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from less import LESSRegressor, LESSClassifier

### CLASSIFICATION ###

X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, \
                           n_clusters_per_class=2, n_informative=10, random_state=42)

# Train and test split
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.3, random_state=42)

# LESS fit() & predict()
LESS_model = LESSClassifier(random_state=42)
LESS_model.fit(X_train, y_train)
y_pred = LESS_model.predict(X_test)
print('Test accuracy of LESS: {0:.2f}'.format(accuracy_score(y_pred, y_test)))


### REGRESSION ###

X, y = make_regression(n_samples=1000, n_features=20, random_state=42)

# Train and test split
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.3, random_state=42)

# LESS fit() & predict()
LESS_model = LESSRegressor(random_state=42)
LESS_model.fit(X_train, y_train)
y_pred = LESS_model.predict(X_test)
print('Test error of LESS: {0:.2f}'.format(mean_squared_error(y_pred, y_test)))

```

## Tutorials

Our **two-part** [tutorial on Colab](https://colab.research.google.com/drive/183MRHH-i4XT3-HepHbIKVRPiwH7uMzrw?usp=sharing) aims at getting you familiar with LESS **regression**. If you want to try the tutorials on your own computer, then you also need to install the following additional packages: `pandas`, `matplotlib`, and `seaborn`.

## Recommendation

Default implementation of LESS uses Euclidean distances with radial basis function. Therefore, it is a good idea to scale the input data before fitting. This can be done by setting the parameter `scaling` in `LESSRegressor` or `LESSClassifier` to `True` (this is the default value) or by preprocessing the data as follows:

```python
from sklearn.preprocessing import StandardScaler

SC = StandardarScaler()
X_train = SC.fit_transform(X_train)
X_test = SC.transform(X_test)
```

## Citation
Our software can be cited as:
````
  @misc{LESS,
    author = "Ilker Birbil",
    title = "LESS: LEarning with Subset Stacking",
    year = 2021,
    url = "https://github.com/sibirbil/LESS/"
  }
````

## Parallel Version

An `openmpi` implementation of LESS is also available in [another repository](https://github.com/sibirbil/LESS-MPI).

## Changes in v.0.2.0

* Classification is added (`LESSClassifier`)
* Scaling is automatically done as default (`scaling = True`)
* The default global estimator for regression is now `DecisionTreeRegressor` instead of `LinearRegression` (`global_estimator=DecisionTreeRegressor`)
* Warnings can be turned on or off with a flag (`warnings = True`)

## Changes in v.0.3.0

* Typos are corrected
* The hidden class for the binary classifier is now separate
* Local subsets with a single class are handled (the case of `ConstantPredictor`)

---

#### Acknowledgments

We thank Oguz Albayrak for his help with structuring our Python scripts.
