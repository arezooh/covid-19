import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn import linear_model
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn import svm
# import warnings
# warnings.filterwarnings('ignore')
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer
from pexecute.process import ProcessLoom
import time
from sys import argv


####################################################### GBM: Gradient Boosting Regressor
def GBM(X_train, X_test, y_train):

    parameters = {'n_estimators': 10000, 'max_depth': 4, 'min_samples_split': 2,
                  'learning_rate': 0.01, 'loss': 'ls'}
    GradientBoostingRegressorObject = GradientBoostingRegressor(random_state=0, **parameters)

    GradientBoostingRegressorObject.fit(X_train, y_train)
    y_prediction = GradientBoostingRegressorObject.predict(X_test)

    return y_prediction


###################################################### GLM: Generalized Linear Model, we use Lasso
def GLM(X_train, X_test, y_train):

    GLM_Model = ElasticNetCV(random_state=0, tol=0.01, cv=5, max_iter=20000)
    GLM_Model.fit(X_train, y_train)
    y_prediction = GLM_Model.predict(X_test)

    return y_prediction


####################################################### KNN: K-Nearest Neighbors
def KNN(X_train, X_test, y_train):

    KNeighborsRegressorObject = KNeighborsRegressor()
    # Grid search over different Ks to choose the best one
    parameters = {'n_neighbors': [5, 10, 20, 25, 30, 40, 50, 60, 70, 80]}
    GridSearchOnKs = GridSearchCV(KNeighborsRegressorObject, parameters, cv=5)
    GridSearchOnKs.fit(X_train, y_train)
    best_K = GridSearchOnKs.best_params_
    # train KNN with the best K
    print('best k:', best_K['n_neighbors'])
    KNN_Model = KNeighborsRegressor(n_neighbors=best_K['n_neighbors'])
    KNN_Model.fit(X_train, y_train)
    y_prediction = KNN_Model.predict(X_test)

    return y_prediction


####################################################### NN: Neural Network
def NN(X_train, X_test, y_train, y_test):

    scaler = MinMaxScaler()  # For normalizing dataset
    X_train_scaled = scaler.fit_transform(X_train)
    y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1))
    X_test_scaled = scaler.fit_transform(X_test)
    #y_test_scaled = scaler.fit_transform(y_test.reshape(-1, 1))

    def denormalize(main_data, normal_data):

        main_data = main_data.reshape(-1, 1)
        normal_data = normal_data.reshape(-1, 1)
        scaleObject = MinMaxScaler()
        scaleMain = scaleObject.fit_transform(main_data)
        denormalizedData = scaleObject.inverse_transform(normal_data)

        return denormalizedData

    neurons = (X_train_scaled.shape[1]) // 2 + 2
    NeuralNetworkObject = MLPRegressor(random_state=0, hidden_layer_sizes=(neurons,), alpha=0.1)
    NeuralNetworkObject.fit(X_train_scaled, y_train_scaled.ravel())
    y_prediction = NeuralNetworkObject.predict(X_test_scaled)
    y_prediction = denormalize(y_test, y_prediction)

    return y_prediction


########################################################## MM-LR: Mixed Model with Linear Regression
def MM_LR(X_train, X_test, y_train):

    # fit a linear regression model on the outputs of the other models
    regressionModelObject = linear_model.LinearRegression()
    regressionModelObject.fit(X_train, y_train)
    y_prediction = regressionModelObject.predict(X_test)

    return y_prediction









