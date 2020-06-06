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
from sklearn.linear_model import ElasticNet, ElasticNetCV, Lasso
import statsmodels.api as sm
from sklearn import svm
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer
from pexecute.process import ProcessLoom
import time
from sys import argv
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


####################################################### GBM: Gradient Boosting Regressor
def GBM(X_train, X_test, y_train):

    parameters = {'n_estimators': 5000, 'max_depth': 4, 'min_samples_split': 2,
                  'learning_rate': 0.01, 'loss': 'ls'}
    GradientBoostingRegressorObject = GradientBoostingRegressor(random_state=1, **parameters)

    GradientBoostingRegressorObject.fit(X_train, y_train)
    y_prediction = GradientBoostingRegressorObject.predict(X_test)
    y_prediction_train = GradientBoostingRegressorObject.predict(X_train)

    return y_prediction, y_prediction_train


###################################################### GLM: Generalized Linear Model, we use Lasso
# def GLM(X_train, X_test, y_train):
#
#     GLM_Model = ElasticNet(random_state=1)
#     GLM_Model.fit(X_train, y_train)
#     y_prediction = GLM_Model.predict(X_test)
#     y_prediction_train = GLM_Model.predict(X_train)
#     print('GLM coef: ', GLM_Model.coef_)
#
#     return np.array(y_prediction).ravel(), np.array(y_prediction_train).ravel()

def GLM(X_train, X_test, y_train):

    Xtrain = sm.add_constant(X_train.values)
    Xtest = sm.add_constant(X_test.values)
    poisson_training_results = sm.ZeroInflatedPoisson(endog=y_train, exog=Xtrain,
                                                      exog_infl=Xtrain, inflation='logit').fit(maxiter=1000)
    y_prediction_train = np.array(np.round(poisson_training_results.predict(Xtrain, exog_infl=Xtrain)))
    y_prediction = np.array(np.round(poisson_training_results.predict(Xtest, exog_infl=Xtest)))

    # poisson_training_results = sm.GLM(y_train, Xtrain, family=sm.families.Gaussian()).fit()
    # y_prediction_train = np.array(np.round(poisson_training_results.predict(Xtrain)))
    # y_prediction = np.array(np.round(poisson_training_results.predict(Xtest)))

    return y_prediction, y_prediction_train


# ####################################################### KNN: K-Nearest Neighbors
def KNN(X_train, X_test, y_train):

    KNeighborsRegressorObject = KNeighborsRegressor()
    # Grid search over different Ks to choose the best one
    parameters = {'n_neighbors': [10, 20, 30, 40, 50, 60, 70, 80]}
    GridSearchOnKs = GridSearchCV(KNeighborsRegressorObject, parameters, cv=5)
    GridSearchOnKs.fit(X_train, y_train)
    best_K = GridSearchOnKs.best_params_
    # train KNN with the best K
    print('best k:', best_K['n_neighbors'])
    KNN_Model = KNeighborsRegressor(n_neighbors=best_K['n_neighbors'])
    KNN_Model.fit(X_train, y_train)
    y_prediction = KNN_Model.predict(X_test)
    y_prediction_train = KNN_Model.predict(X_train)

    return y_prediction, y_prediction_train


####################################################### NN: Neural Network
def NN(X_train, X_test, y_train, y_test):

    # prepare dataset with input and output scalers, can be none
    def get_dataset(input_scaler, output_scaler):

        trainX, testX = X_train, X_test
        trainy, testy = y_train, y_test
        # scale inputs
        if input_scaler is not None:
            # fit scaler
            input_scaler.fit(trainX)
            # transform training dataset
            trainX = input_scaler.transform(trainX)
            # fit scaler
            input_scaler.fit(testX)
            # transform test dataset
            testX = input_scaler.transform(testX)
        if output_scaler is not None:
            # reshape 1d arrays to 2d arrays
            trainy = trainy.reshape(len(trainy), 1)
            testy = testy.reshape(len(testy), 1)
            # fit scaler on training dataset
            output_scaler.fit(trainy)
            # transform training dataset
            trainy = output_scaler.transform(trainy)
            # fit scaler on testing dataset
            output_scaler.fit(testy)
            # transform test dataset
            testy = output_scaler.transform(testy)
        return trainX, trainy, testX, testy

    def denormalize(main_data, normal_data, scaler):

        main_data = main_data.reshape(-1, 1)
        normal_data = normal_data.reshape(-1, 1)
        # scaleObject = StandardScaler()
        scaler.fit_transform(main_data)
        denormalizedData = scaler.inverse_transform(normal_data)

        return denormalizedData

    trainX, trainy, testX, testy = get_dataset(MinMaxScaler(), MinMaxScaler())
    neurons = (trainX.shape[1]) // 2 + 1
    # print(neurons)
    # NeuralNetworkObject = MLPRegressor(hidden_layer_sizes=(neurons,), max_iter=2000, random_state=1, solver='sgd')
    # NeuralNetworkObject.fit(trainX, trainy.ravel())

    NeuralNetworkObject = keras.Sequential([
        keras.Input(shape=(trainX.shape[1],)),
        layers.Dense(neurons),
        layers.Dense(1,activation=tf.exp)
    ])
    # Compile the model
    NeuralNetworkObject.compile(
        loss='poisson',
        optimizer=keras.optimizers.RMSprop(),
        metrics=['mean_squared_error'])

    early_stop = EarlyStopping(monitor='val_loss', patience=30)

    NeuralNetworkObject.fit(trainX, trainy.ravel(),
                   callbacks=[early_stop],
                   batch_size=128,
                   validation_split=0.2,
                   epochs=2000, verbose=0)

    test_mse = NeuralNetworkObject.evaluate(testX, testy)[1]
    print('NN mse test: ', test_mse)
    train_mse = NeuralNetworkObject.evaluate(trainX, trainy)[1]
    print('NN mse train: ', train_mse)
    y_prediction = NeuralNetworkObject.predict(testX)
    y_prediction = denormalize(testy, y_prediction, MinMaxScaler())
    y_prediction_train = NeuralNetworkObject.predict(trainX)
    y_prediction_train = denormalize(trainy, y_prediction_train, MinMaxScaler())

    return np.array(y_prediction).ravel(), np.array(y_prediction_train).ravel()

########################################################## MM-LR: Mixed Model with Linear Regression
def MM_LR(X_train, X_test, y_train):

    # fit a linear regression model on the outputs of the other models
    regressionModelObject = linear_model.LinearRegression()
    regressionModelObject.fit(X_train, y_train)
    y_prediction = regressionModelObject.predict(X_test)
    y_prediction_train = regressionModelObject.predict(X_train)

    return y_prediction, y_prediction_train









