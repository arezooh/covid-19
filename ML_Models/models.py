from featureSelection import correlatinosWithTarget
from makeHistoricalData import makeHistoricalData
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


########################################################## SVM-poly
def SVM(X_train, X_test, y_train):

    svmModelObject = svm.SVC(kernel = 'poly')
    svmModelObject.fit(X_train, y_train)
    y_prediction = svmModelObject.predict(X_test)

    return y_prediction


####################################################### GBM: Gradient Boosting Regressor
def GBM(X_train, X_test, y_train):

    parameters = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
                  'learning_rate': 0.01, 'loss': 'ls'}
    GradientBoostingRegressorObject = GradientBoostingRegressor(**parameters)

    GradientBoostingRegressorObject.fit(X_train, y_train)
    y_prediction = GradientBoostingRegressorObject.predict(X_test)

    return y_prediction


###################################################### GLM: Generalized Linear Model, we use Lasso
def GLM(X_train, X_test, y_train):

    GLM_Model = linear_model.Lasso()
    parameters = {'alpha': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
    GridSearch_alpha = GridSearchCV(GLM_Model, parameters, cv=5)  # alpha is l2 regularization term
    GridSearch_alpha.fit(X_train, y_train)
    best_alpha = GridSearch_alpha.best_params_
    GLM_Model = linear_model.Lasso(alpha=best_alpha['alpha'])
    print('best alpha for GLM: ', best_alpha['alpha'])
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
def NN(X_train, X_test, y_train):

    # scaler = MinMaxScaler()  # For normalizing dataset
    # X_train_scaled = scaler.fit_transform(X_train)
    # y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1))
    # X_test_scaled = scaler.fit_transform(X_test)
    # y_test_scaled = scaler.fit_transform(y_test.reshape(-1, 1))
    #
    # def denormalize(main_data, normal_data):
    #
    #     main_data = main_data.reshape(-1, 1)
    #     normal_data = normal_data.reshape(-1, 1)
    #     scaleObject = MinMaxScaler()
    #     scaleMain = scaleObject.fit_transform(main_data)
    #     denormalizedData = scaleObject.inverse_transform(normal_data)
    #
    #     return denormalizedData
    #print(X_train.shape)
    neurons = (X_train.shape[1]) // 2 + 2
    #print(neurons)
    NeuralNetworkObject = MLPRegressor(hidden_layer_sizes=(neurons,))  # , activation='tanh'
    # parameters = {'learning_rate_init': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1]}
    # GridSearch_learningRate= GridSearchCV(NeuralNetworkObject, parameters, cv=5)
    # GridSearch_learningRate.fit(X_train, y_train)
    # best_learning_rate = GridSearch_learningRate.best_params_
    # NeuralNetworkObject = MLPRegressor(hidden_layer_sizes=(neurons,), learning_rate_init=best_learning_rate['learning_rate_init'])
    # print('best learning rate init: ', best_learning_rate['learning_rate_init'])
    #
    # parameters = {'alpha': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1]}
    # GridSearch_alpha = GridSearchCV(NeuralNetworkObject, parameters, cv=5)  # alpha is l2 regularization term
    # GridSearch_alpha.fit(X_train, y_train)
    # best_alpha = GridSearch_alpha.best_params_
    # NeuralNetworkObject = MLPRegressor(hidden_layer_sizes=(neurons,), learning_rate_init=best_learning_rate['learning_rate_init'],
    #                                    alpha=best_alpha['alpha'])
    # print('best alpha: ', best_alpha['alpha'])

    NeuralNetworkObject.fit(X_train, y_train)
    y_prediction = NeuralNetworkObject.predict(X_test)
    #y_prediction = denormalize(y_test, y_prediction)

    return y_prediction
    # scaler = PowerTransformer()  # For normalizing dataset
    # X_train_scaled = scaler.fit_transform(X_train)
    # y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1))
    # X_test_scaled = scaler.fit_transform(X_test)
    # y_test_scaled = scaler.fit_transform(y_test.reshape(-1, 1))
    #
    # def denormalize(main_data, normal_data):
    #
    #     main_data = main_data.reshape(-1, 1)
    #     normal_data = normal_data.reshape(-1, 1)
    #     scaleObject = PowerTransformer()
    #     scaleMain = scaleObject.fit_transform(main_data)
    #     denormalizedData = scaleObject.inverse_transform(normal_data)
    #
    #     return denormalizedData
    #
    # def neural_net_model(X_data, input_dim):  # a 1-layer NN
    #
    #     neurons = int(input_dim/2)
    #     # layer 1 multiplying and adding bias and activation function
    #     W_1 = tf.Variable(tf.random_uniform([input_dim, neurons]))
    #     b_1 = tf.Variable(tf.zeros([neurons]))
    #     layer_1 = tf.add(tf.matmul(X_data, W_1), b_1)
    #     layer_1 = tf.nn.relu(layer_1)
    #     # output layer
    #     W_O = tf.Variable(tf.random_uniform([neurons, 1]))
    #     b_O = tf.Variable(tf.zeros([1]))
    #     output = tf.add(tf.matmul(layer_1, W_O), b_O)
    #     # output layer multiplying and adding bias then activation function
    #     # notice output layer has one node only since performing #regression
    #     return output
    #
    # xPlaceHolder = tf.placeholder("float")
    # yPlaceHolder = tf.placeholder("float")
    # output = neural_net_model(xPlaceHolder, X_train.shape[1])
    # # our mean squared error cost function
    # cost = tf.reduce_mean(abs(output - yPlaceHolder))
    # # Gradinent Descent optimiztion for updating weights and biases
    # train = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
    #
    # with tf.Session() as sess:
    #     # Initiate session and initialize all vaiables
    #     sess.run(tf.global_variables_initializer())
    #     results_train = []
    #     results_test = []
    #     epochs = 20
    #     for i in range(epochs):
    #         for j in range(X_train_scaled.shape[0]):
    #             sess.run([cost, train],
    #                      feed_dict={xPlaceHolder: X_train_scaled[j, :].reshape(1, X_train.shape[1]), yPlaceHolder: y_train_scaled[j]})
    #             # Run cost and train with each sample
    #         results_train.append(sess.run(cost, feed_dict={xPlaceHolder: X_train_scaled, yPlaceHolder: y_train_scaled}))
    #         results_test.append(sess.run(cost, feed_dict={xPlaceHolder: X_test_scaled, yPlaceHolder: y_test_scaled}))
    #         print('Epoch :', i, 'Cost :', results_train[i])
    #
    #     # predict output of test data after training
    #     y_prediction = sess.run(output, feed_dict={xPlaceHolder: X_test_scaled})
    #
    #     print('Cost :', sess.run(cost, feed_dict={xPlaceHolder: X_test_scaled, yPlaceHolder: y_test_scaled}))
    #     # Denormalize data
    #     # y_test_denormalized = denormalize(y_test, y_test_scaled)
    #     y_prediction = denormalize(y_test, y_prediction)

    # return y_prediction


########################################################## MM-LR: Mixed Model with Linear Regression
def MM_LR(X_train, X_test, y_train):

    # fit a linear regression model on the outputs of the other models
    regressionModelObject = linear_model.LinearRegression()
    regressionModelObject.fit(X_train, y_train)
    y_prediction = regressionModelObject.predict(X_test)

    return y_prediction









