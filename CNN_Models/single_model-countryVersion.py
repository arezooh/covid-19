################################################################ Imports

import cnn_search

import os
import datetime
import matplotlib.pyplot as plt
import progressbar

from math import log2, floor, ceil, sqrt
from numpy import array, zeros, save, load, copyto, append
from os import getpid

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Dense, BatchNormalization, MaxPooling2D, Dropout

from numpy.random import seed
from tensorflow.random import set_seed

from normalizers import normal_x, normal_y, inverse_normal_y

################################################################ Globals

_RESULTS_DIR_ = './results/'
_PROGRESS_BAR_WIDGET_ = [progressbar.Percentage(), ' ', progressbar.Bar('=', '[', ']'), ' ']
_NUMPY_SEED_ = 580
_TENSORFLOW_SEED_ = 870

single_model_parameters = (3, 0.5, 0.8, 2, 0, 0.1, 16)
target_counties = [36061, 40117, 51059]
image_size = 300

################################################################

# Use this function to log states of code, helps to find bugs
def log(str):
    t = datetime.datetime.now().isoformat()
    with open('log_sd', 'a') as logFile:
        logFile.write('[{0}][{1}] {2}\n'.format(t, getpid(), str))

def pad_subImage(data, input_size, minX, minY, maxX, maxY):
    data_shape = data.shape
    padded_data = []

    for i in range(data_shape[0]):
        padded_data.append(cnn_search.pad_data(data[i], input_size))

    n = input_size // 2
    padded_data = array(padded_data)
    return padded_data[0:data_shape[0], minX:maxX + 2 * n, minY:maxY + 2 * n, 0:data_shape[3]] 

def create_model(inputSize, hiddenDropout, visibleDropout, noBlocks, noDenseLayer, increaseFilters, learning_rate):
    # Set random seeds to make situation equal for all models 
    seed(_NUMPY_SEED_)
    set_seed(_TENSORFLOW_SEED_)

    noFilters = 64
    model = keras.Sequential()

    # Layers before first block
    model.add(tf.keras.layers.Conv2D(filters=noFilters, kernel_size = (3,3), padding='same', activation='relu', input_shape=(inputSize, inputSize, 62)))
    if (visibleDropout != 0):
        model.add(Dropout(visibleDropout))

    # layers in Blocks
    for i in range(noBlocks):
        if (increaseFilters == 1):
            noFilters = 64 * pow(2, i)
        model.add(Conv2D(filters=noFilters, kernel_size = (3,3), padding='same', activation="relu"))
        model.add(Conv2D(filters=noFilters, kernel_size = (3,3), padding='same', activation="relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())
        if (hiddenDropout != 0):
            model.add(Dropout(hiddenDropout))

    # Layers after last block
    for i in range(noDenseLayer - 1):
        model.add(Dense(512,activation="relu"))
    # Last layer
    model.add(Dense(1,activation="relu"))

    model_optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='mean_squared_error', optimizer=model_optimizer)
    return model

# This function extract windows with "input_size" size from image, train model with the windows data
def train_data(model, x_train, y_train, x_validation, y_validation, NO_epochs, input_size, batch_size):
    data_shape = x_train.shape
    y_shape = y_train.shape
    no_validation = x_validation.shape[0]
    
    padded_x = []
    padded_y = []

    for i in range(data_shape[0]):
        padded_x.append(cnn_search.pad_data(x_train[i], input_size))
        padded_y.append(cnn_search.pad_data(y_train[i], input_size))

    x_train = array(padded_x)
    y_train = array(padded_y)
    
    padded_x = []
    padded_y = []

    for i in range(no_validation):
        padded_x.append(cnn_search.pad_data(x_validation[i], input_size))
        padded_y.append(cnn_search.pad_data(y_validation[i], input_size))

    x_validation = array(padded_x)
    y_validation = array(padded_y)

    # clear memory
    del padded_x, padded_y

    # progressbar
    progressBar = progressbar.ProgressBar(maxval=data_shape[1]*data_shape[2], widgets=_PROGRESS_BAR_WIDGET_)
    progressBar.start()

    for i in range(data_shape[1]):
        for j in range(data_shape[2]):
            subX_trian = x_train[0:data_shape[0], i:i+input_size, j:j+input_size, 0:data_shape[3]]
            subY_train = y_train[0:data_shape[0], i:i+input_size, j:j+input_size, 0:y_shape[3]]

            subX_validation = x_validation[0:data_shape[0], i:i+input_size, j:j+input_size, 0:data_shape[3]]
            subY_validation = y_validation[0:data_shape[0], i:i+input_size, j:j+input_size, 0:y_shape[3]]

            model.fit(subX_trian, subY_train, batch_size=batch_size, epochs=NO_epochs, verbose=0, validation_data=(subX_validation, subY_validation))
            
            progressBar.update((i * data_shape[2]) + j)

    progressBar.finish()

# This function extract windows with "input_size" size from image, evaluate model with the windows data
# Note: the data in here is padded.
def evaluate_data_sd(model, x_data, y_data, input_size):
    data_shape = x_data.shape
    y_data_org = inverse_normal_y(y_data)

    sum_org = 0
    sum_predict = 0
    sum_simple = 0
    sum_MAE = 0
    sum_MAPE = 0
    sum_MASE = 0

    country_org = zeros(data_shape[0])
    country_pred = zeros(data_shape[0])
    country_simple = zeros(data_shape[0])
    
    sum_MAE_country = 0
    sum_MAPE_country = 0
    sum_MASE_country = 0

    _debug_no_pixcels = 0

    # progressbar
    progressBar = progressbar.ProgressBar(maxval=(data_shape[1] - (input_size // 2 * 2))*(data_shape[2] - (input_size // 2 * 2)), widgets=_PROGRESS_BAR_WIDGET_)
    progressBar.start()

    for i in range(data_shape[1] - (input_size // 2 * 2)):
        for j in range(data_shape[2] - (input_size // 2 * 2)):
            subX = x_data[0:data_shape[0], i:i+input_size, j:j+input_size, 0:data_shape[3]]

            subY_predict_normal = model.predict(subX)
            pred_shape = subY_predict_normal.shape
            subY_predict = inverse_normal_y(subY_predict_normal)

            for k in range(pred_shape[0]):
                if (k >= pred_shape[0] - 21):
                    sum_org += y_data_org[k][i][j][0]
                    sum_predict += subY_predict[k][0][0][0]
                    sum_simple += x_data[k][i][j][-4]
                    sum_MAE += abs(y_data_org[k][i][j][0] - subY_predict[k][0][0][0])
                    sum_MAPE += abs(y_data_org[k][i][j][0] - subY_predict[k][0][0][0])
                    sum_MASE += abs(y_data_org[k][i][j][0] - x_data[k][i][j][-4])

                country_org[k] += y_data_org[k][i][j][0]
                country_pred[k] += subY_predict[k][0][0][0]
                country_simple[k] += x_data[k][i][j][-4]

            _debug_no_pixcels += 1
            
            progressBar.update((i * (data_shape[2] - (input_size // 2 * 2))) + j)

    progressBar.finish()

    if (sum_org == 0):
        log('sum_org is zero, _debug_no_pixcels = {0}'.format(_debug_no_pixcels))

    MAE_pixel = sum_MAE / (21 * data_shape[1] * data_shape[2])
    MAPE_pixel = sum_MAPE / sum_org
    MASE_pixel = MAE_pixel / (sum_MASE / (21 * data_shape[1] * data_shape[2]))

    # calculating country errors
    for k in range(21):
        sum_MAE_country += abs(country_org[k - 21] - country_pred[k - 21])
        sum_MASE_country += abs(country_org[k - 21] - country_simple[k - 21])
        
        if (country_org[k - 21] != 0):
            sum_MAPE_country += abs(country_org[k - 21] - country_pred[k - 21]) / country_org[k - 21]
        else:
            log('sum_org_country[{0}] is zero'.format(k))

    MAE_country = sum_MAE_country / 21
    MAPE_country = sum_MAPE_country / 21
    MASE_country = MAE_country / (sum_MASE_country / 21)

    results = (MAE_pixel, MAPE_pixel, MASE_pixel, MAE_country, MAPE_country, MASE_country)

    return (results, country_org, country_pred)

def plot_chart(fips, prediction, original):
    start_day = (cnn_search.startDay + datetime.timedelta(days=14))
    days = []
    for i in range(len(prediction)):
        days.append((start_day + datetime.timedelta(days=i)).strftime('%m-%d'))

    fig, ax = plt.subplots()

    ax.tick_params(axis = 'x', labelrotation = 60, labelsize=8)
    ax.plot(days, original, color='red', linestyle='-', label='original')
    ax.plot(days[:-20], prediction[:-20], color='blue', linestyle='-', label='train')
    ax.plot(days[-21:], prediction[-21:], color='green', linestyle='-', label='test')
    ax.set(ylabel='death')

    fig.set_figheight(5)
    fig.set_figwidth(25)
    plt.legend()

    fig.savefig(_RESULTS_DIR_ + 'result_{0}.png'.format(fips))

def predict_counties_result(counties_fips, model, normal_x_data, normal_y_data, input_size):
    min_x = 0
    min_y = 0
    max_x = 300
    max_y = 300

    log('MIN x:{0}, y:{1} | MAX x:{2}, y:{3}'.format(min_x, min_y, max_x, max_y))

    result, country_org, country_pred = evaluate_data_sd(model, 
        pad_subImage(normal_x_data, input_size, min_x, min_y, max_x, max_y),
        pad_subImage(normal_y_data, input_size, min_x, min_y, max_x, max_y),
        input_size,)

    # create results directory if it's missing
    if (os.path.exists('results') == False):
        os.mkdir('results')
        
    # save results in a file
    save_results('Pixels', result[0], result[1], result[2])
    save_results('Country', result[3], result[4], result[5])

    plot_chart('country', country_pred, country_org)

def save_results(results_type, MAE, MAPE, MASE):
    output = 'results for {0}, MAE:{1}, MAPE:{2}, MASE:{3}'.format(results_type, MAE, MAPE, MASE)
    with open(_RESULTS_DIR_ + 'single_model_results.txt', 'a') as logFile:
        logFile.write('{0}\n'.format(output))

################################################################ main

if __name__ == "__main__":
    cnn_search.init_hashCounties()
    cnn_search.init_days()

    # Check if instances are ready
    if (os.path.exists('x_' + cnn_search._INSTANCES_FILENAME_) and os.path.exists('y_' + cnn_search._INSTANCES_FILENAME_)):
        log('instances found')
    else:
        log('creating instances')
        cnn_search.create_instances()

    x_instances = load('x_' + cnn_search._INSTANCES_FILENAME_)
    y_instances = load('y_' + cnn_search._INSTANCES_FILENAME_)

    ################################################################ split imageArray into train, validation and test

    log('spliting data into train, validation and test')

    x_dataTrain = x_instances[:-42]
    y_dataTrain = y_instances[:-42]

    x_dataValidation = x_instances[-42:-21]
    y_dataValidation = y_instances[-42:-21]

    x_dataTest = x_instances[-21:]
    y_dataTest = y_instances[-21:]

    ################################################################ normalize data

    log('normalizing data')

    normal_x_dataTrain, normal_x_dataValidation, normal_x_dataTest = normal_x(x_dataTrain, x_dataValidation, x_dataTest)
    normal_y_dataTrain, normal_y_dataValidation, normal_y_dataTest = normal_y(y_dataTrain, y_dataValidation, y_dataTest)

    data_shape = normal_x_dataTrain.shape
    no_train = normal_x_dataTrain.shape[0]
    no_validation = normal_x_dataValidation.shape[0]
    no_test = normal_x_dataTest.shape[0]

    ################################################################ clearing memory

    del x_instances, x_dataTrain, x_dataValidation, x_dataTest
    del y_instances, y_dataTrain, y_dataValidation, y_dataTest

    log('Phase of testing the model started')

    ################################################################ model execution

    input_size = single_model_parameters[0]
    hidden_dropout = single_model_parameters[1]
    visible_dropout = single_model_parameters[2]
    NO_dense_layer = single_model_parameters[3]
    increase_filters = single_model_parameters[4]
    learning_rate = single_model_parameters[5]
    batch_size = single_model_parameters[6]

    log('Model testing with parameters {0}'.format(single_model_parameters))
    NO_blocks = floor(log2(input_size))
    model = create_model(input_size, hidden_dropout, visible_dropout, NO_blocks, NO_dense_layer, increase_filters, learning_rate)
    train_data(model, normal_x_dataTrain, normal_y_dataTrain, normal_x_dataValidation, normal_y_dataValidation, 2, input_size, batch_size)
    predict_counties_result(target_counties, 
        model, 
        append(append(normal_x_dataTrain, normal_x_dataValidation).reshape(no_train + no_validation, data_shape[1], data_shape[2], data_shape[3]), normal_x_dataTest).reshape(no_train + no_validation + no_test, data_shape[1], data_shape[2], data_shape[3]), 
        append(append(normal_y_dataTrain, normal_y_dataValidation).reshape(no_train + no_validation, data_shape[1], data_shape[2], 1), normal_y_dataTest).reshape(no_train + no_validation + no_test, data_shape[1], data_shape[2], 1), 
        input_size)


        