

import numpy as np
from numpy.linalg import inv

from sklearn import datasets 
from sklearn.decomposition import PCA
from sklearn.feature_extraction. text import TfidfVectorizer 
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler

class LinearRegression:
    @staticmethod
    def loss (predictions, y):
        return ((predictions - y) ** 2).sum() / len(y)

    def fit(self, X, y):
        n_features = np.size(X, 1)
        y = np.expand_dims(y, axis=1)
        self.weights = inv(X.T @ X) @ X.T @ y 
        return self
        
    def predict(self, X):
        return (X @ self.weights).sum(axis=1)


def train_model(X,Y):
    model = make_pipeline(
        TfidfVectorizer(analyzer="char"),
        Normalizer(),
        LinearRegression()
    )
    model.fit(X, y) 
    return model

def test_model(model, X, y):
    predictions = model.predict(X) 
    return LinearRegression.loss(predictions, y)


newsgroup_data = datasets.fetch_20newsgroups(subset="train")
X = newsgroup_data.data
y = newsgroup_data.target

model = train_model(X, y)
test_result = test_model(model, X, y)

print(test_result)
