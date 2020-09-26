################################################################ Imports

import cnn_search

from math import log2, floor, ceil, sqrt
from numpy import array, zeros, save, load, copyto, append
from os import getpid
import os
import datetime
import matplotlib.pyplot as plt

_RESULTS_DIR_ = './results/'

single_model_parameters = (3, 0, 0.3, 1, 0)
target_counties = [36061, 40117, 51059]
image_size = 300

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

# This function extract windows with "input_size" size from image, evaluate model with the windows data
# Note: the data in here is padded.
def evaluate_data_sd(model, x_data, y_data, input_size, normal_min, normal_max):
    data_shape = x_data.shape
    y_data_org = cnn_search.inverse_normal_y(y_data, normal_min, normal_max)

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

    for i in range(data_shape[1] - (input_size // 2 * 2)):
        for j in range(data_shape[2] - (input_size // 2 * 2)):
            subX = x_data[0:data_shape[0], i:i+input_size, j:j+input_size, 0:data_shape[3]]

            subY_predict_normal = model.predict(subX)
            pred_shape = subY_predict_normal.shape
            subY_predict = cnn_search.inverse_normal_y(subY_predict_normal, normal_min, normal_max)

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

def county_pixcels(county_fips):
    gridIntersection = cnn_search.loadJsonFile(cnn_search._GRID_INTERSECTION_FILENAME_)
    pixcels = []

    for x in range(len(gridIntersection)):
        for y in range(len(gridIntersection[x])):
            for county in gridIntersection[x][y]:
                if county['fips'] == county_fips:
                    pixcels.append((x, y, county_fips))
                elif (type(county['fips']) != type(county_fips)):
                    raise Exception ('type mismatch for county_fips')

    return(pixcels)

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

def predict_counties_result(counties_fips, model, normal_x_data, normal_y_data, input_size, normal_min, normal_max):
    pixcels = []
    min_x = 0
    min_y = 0
    max_x = 300
    max_y = 300

    # for fips in counties_fips:
    #     pixcels.extend(county_pixcels(fips))

    # for pixcel in pixcels:
    #     if (min_x == -1 or pixcel[0] < min_x):
    #         min_x = pixcel[0]
    #     if (min_y == -1 or pixcel[1] < min_y):
    #         min_y = pixcel[1]
    #     if (max_x == -1 or pixcel[0] > max_x):
    #         max_x = pixcel[0]
    #     if (max_y == -1 or pixcel[1] > max_y):
    #         max_y = pixcel[1]

    log('MIN x:{0}, y:{1} | MAX x:{2}, y:{3}'.format(min_x, min_y, max_x, max_y))

    result, country_org, country_pred = evaluate_data_sd(model, 
        pad_subImage(normal_x_data, input_size, min_x, min_y, max_x, max_y),
        pad_subImage(normal_y_data, input_size, min_x, min_y, max_x, max_y),
        input_size,
        normal_min,
        normal_max)

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

# get a 4D numpy array and normalize it
def normal_x(train, validation, final_test):
    data_shape = train.shape
    no_validation = validation.shape[0]
    no_final_test = final_test.shape[0]

    normalizers = []
    for b in range(data_shape[3]):
        if (b >= 6 and ((b - 6) % 4 == 0 or (b - 6) % 4 == 1)):
            normalizers.append(cnn_search.standardizer())
        else:
            normalizers.append(cnn_search.normalizer())

    for i in range(data_shape[0]):
        for j in range(data_shape[1]):
            for a in range(data_shape[2]):
                for b in range(data_shape[3]):
                    if (b >= 6 and ((b - 6) % 4 == 0 or (b - 6) % 4 == 1)):
                        normalizers[b].update_mean(train[i][j][a][b])
                    else:
                        normalizers[b].update(train[i][j][a][b])

    # calculate standardizers mean
    for b in range(6, data_shape[3], 4):
            normalizers[b].calculate_mean()
            normalizers[b + 1].calculate_mean()

    # update standardizers deviation
    for i in range(data_shape[0]):
        for j in range(data_shape[1]):
            for a in range(data_shape[2]):
                for b in range(6, data_shape[3], 4):
                    normalizers[b].update_deviation(train[i][j][a][b])
                    normalizers[b + 1].update_deviation(train[i][j][a][b + 1])

    # calculate standardizers deviation
    for b in range(6, data_shape[3], 4):
            normalizers[b].calculate_deviation()
            normalizers[b + 1].calculate_deviation()

    normal_train = zeros((data_shape[0], data_shape[1], data_shape[2], data_shape[3]))
    normal_validation = zeros((no_validation, data_shape[1], data_shape[2], data_shape[3]))
    normal_final_test = zeros((no_final_test, data_shape[1], data_shape[2], data_shape[3]))

    for i in range(data_shape[0]):
        for j in range(data_shape[1]):
            for a in range(data_shape[2]):
                for b in range(data_shape[3]):
                    if (b >= 6 and ((b - 6) % 4 == 0 or (b - 6) % 4 == 1)):
                        normal_train[i][j][a][b] = normalizers[b].standardize(train[i][j][a][b])
                        if (i < no_validation):
                            normal_validation[i][j][a][b] = normalizers[b].standardize(validation[i][j][a][b])
                        if (i < no_final_test):
                            normal_final_test[i][j][a][b] = normalizers[b].standardize(final_test[i][j][a][b])
                    else:
                        normal_train[i][j][a][b] = normalizers[b].normal(train[i][j][a][b])
                        if (i < no_validation):
                            normal_validation[i][j][a][b] = normalizers[b].normal(validation[i][j][a][b])
                        if (i < no_final_test):
                            normal_final_test[i][j][a][b] = normalizers[b].normal(final_test[i][j][a][b])

    # check deviation and mean
    for b in range(6, data_shape[3], 4):
        normalizers[b].check(b)
        normalizers[b + 1].check(b + 1)

    return (normal_train, normal_validation, normal_final_test)

def normal_y(train, validation, final_test):
    data_shape = train.shape
    no_validation = validation.shape[0]
    no_final_test = final_test.shape[0]

    obj_normalizer = cnn_search.standardizer()
    
    for i in range(data_shape[0]):
        for j in range(data_shape[1]):
            for a in range(data_shape[2]):
                obj_normalizer.update_mean(train[i][j][a])

    # calculate standardizers mean
    obj_normalizer.calculate_mean()
    
    # update standardizers deviation
    for i in range(data_shape[0]):
        for j in range(data_shape[1]):
            for a in range(data_shape[2]):
                obj_normalizer.update_deviation(train[i][j][a])
                
    # calculate standardizers deviation
    obj_normalizer.calculate_deviation()

    normal_train = zeros((data_shape[0], data_shape[1], data_shape[2], 1))
    normal_validation = zeros((no_validation, data_shape[1], data_shape[2], 1))
    normal_final_test = zeros((no_final_test, data_shape[1], data_shape[2], 1))

    for i in range(data_shape[0]):
        for j in range(data_shape[1]):
            for a in range(data_shape[2]):
                normal_train[i][j][a][0] = obj_normalizer.standardize(train[i][j][a])
                if (i < no_validation):
                    normal_validation[i][j][a][0] = obj_normalizer.standardize(validation[i][j][a])
                if (i < no_final_test):
                    normal_final_test[i][j][a][0] = obj_normalizer.standardize(final_test[i][j][a])

    obj_normalizer.check(100)
    standard_mean, standard_deviation = obj_normalizer.get_mean_deviation()

    return (normal_train, normal_validation, normal_final_test, standard_mean, standard_deviation)

def inverse_normal_y(normal_data, standard_mean, standard_deviation):
    data_shape = normal_data.shape

    obj_normalizer = cnn_search.standardizer()
    obj_normalizer.set_mean_deviation(standard_mean, standard_deviation)

    data = zeros(data_shape)

    for i in range(data_shape[0]):
        for j in range(data_shape[1]):
            for a in range(data_shape[2]):
                for b in range(data_shape[3]):
                    data[i][j][a][b] = (obj_normalizer.inverse_standardize(normal_data[i][j][a][b]))

    return data

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
    normal_y_dataTrain, normal_y_dataValidation, normal_y_dataTest, normal_min, normal_max = normal_y(y_dataTrain, y_dataValidation, y_dataTest)

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

    log('Model testing with parameters {0}'.format(single_model_parameters))
    NO_blocks = floor(log2(input_size))
    model = cnn_search.create_model(input_size, hidden_dropout, visible_dropout, NO_blocks, NO_dense_layer, increase_filters)
    cnn_search.train_data(model, normal_x_dataTrain, normal_y_dataTrain, normal_x_dataValidation, normal_y_dataValidation, 2, input_size)
    predict_counties_result(target_counties, 
        model, 
        append(append(normal_x_dataTrain, normal_x_dataValidation).reshape(no_train + no_validation, data_shape[1], data_shape[2], data_shape[3]), normal_x_dataTest).reshape(no_train + no_validation + no_test, data_shape[1], data_shape[2], data_shape[3]), 
        append(append(normal_y_dataTrain, normal_y_dataValidation).reshape(no_train + no_validation, data_shape[1], data_shape[2], 1), normal_y_dataTest).reshape(no_train + no_validation + no_test, data_shape[1], data_shape[2], 1), 
        input_size, 
        normal_min, 
        normal_max)


        