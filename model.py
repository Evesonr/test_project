

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

# Notes
# A particular topic? This is linear regression, what are you trying to do exactly?
#   Use logistic regression instead, and consider changing your y to one-hot encoded labels for each topic
#   Consider using Logistic regression instead, or NB, or RF
# evaluating on train set!
# Not using PCA or standard scalar
# Do we need to reduce dimensions? What are dimensions of the dataset?
# Do we need to apply standardisation? Already applying normalisation
# Is pipeline in correct order?
# Numpy imports are inefficient
# from sklearn.datasets import dtaaset...
# returning loss?
# Is calculation of loss correct? Could it be more efficient?
# Fit method could all be looked at.
#   Why calculating n_features and not using?
#   Check matrix multiplication is correct

# For the analyzer, "word" is likely a better choice than "char" for text classification. Analyzing whole words instead of character n-grams will lead to features that capture meaning better.
    # Some other analyzer options to consider:
    # "word" - Split text into words using whitespace and punctuation
    # "char_wb" - Character n-grams but keep words intact
    # "ngram" - Tokenize into n-grams of words instead of individual words