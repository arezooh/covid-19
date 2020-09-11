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
target_counties = [36059, 36061]
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

    # init counties_predict array
    counties_predict = []
    counties_predict_per_day = zeros(78031)

    for _ in range(data_shape[0]):
        counties_predict.append(counties_predict_per_day.copy())

    # load distribution
    distribution = cnn_search.loadJsonFile(cnn_search._DISTRIBUTION_FILENAME_)
    # no need of first 14 days distribution
    distribution = distribution[14:]

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

                for county in distribution[k][i][j]:
                    counties_predict[k][county['fips']] += (subY_predict[k][0][0][0] * county['percent'])

    MAE_county, MAPE_county, MASE_county, MAE_county_round, MAPE_county_round, MASE_county_round, orginal = calculate_county_error_sd(0, counties_predict)

    MAE_pixel = sum_MAE / (21 * data_shape[1] * data_shape[2])
    MAPE_pixel = sum_MAPE / sum_org
    MASE_pixel = MAE_pixel / (sum_MASE / (21 * data_shape[1] * data_shape[2]))

    MAE_country = abs(sum_org - sum_predict)
    MAPE_country = abs(sum_org - sum_predict) / sum_org
    MASE_country = MAE_country / abs(sum_simple - sum_predict)

    results = (MAE_pixel, MAPE_pixel, MASE_pixel, MAE_country, MAPE_country, MASE_country, MAE_county, MAPE_county, MASE_county, MAE_county_round, MAPE_county_round, MASE_county_round)

    return (counties_predict, orginal, results)

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
    min_x = -1
    min_y = -1
    max_x = -1
    max_y = -1

    for fips in counties_fips:
        pixcels.extend(county_pixcels(fips))

    for pixcel in pixcels:
        if (min_x == -1 or pixcel[0] < min_x):
            min_x = pixcel[0]
        if (min_y == -1 or pixcel[1] < min_y):
            min_y = pixcel[1]
        if (max_x == -1 or pixcel[0] > max_x):
            max_x = pixcel[0]
        if (max_y == -1 or pixcel[1] > max_y):
            max_y = pixcel[1]

    log('MIN x:{0}, y:{1} | MAX x:{2}, y:{3}')

    counties_predict, orginal, result = evaluate_data_sd(model, 
        pad_subImage(normal_x_data, input_size, min_x, min_y, max_x, max_y),
        pad_subImage(normal_y_data, input_size, min_x, min_y, max_x, max_y),
        input_size,
        normal_min,
        normal_max)

    counties_predict = array(counties_predict)
    orginal = array(orginal)
    for fips in counties_fips:
        plot_chart(fips, counties_predict[:, fips], orginal[:, fips])

    return result

def calculate_county_error_sd(test_start_day, predictions):
    cnn_search.init_hashCounties()
    cnn_search.init_days()
    countiesData_temporal = cnn_search.loadCounties(cnn_search._COUNTIES_DATA_TEMPORAL_)
    no_days = len(predictions)

    sum_org = 0
    sum_predict = 0
    sum_MAE = 0
    sum_MASE = 0

    sum_predict_round = 0
    sum_MAE_round = 0

    # init counties_predict array
    orginals = []
    orginals_per_day = zeros(78031)

    for _ in range(no_days):
        orginals.append(orginals_per_day.copy())

    counties = cnn_search.loadCounties(cnn_search._CONUTIES_FIPS_)
    for i in range(len(counties)):
        fips = int(counties[i]['county_fips'], 10)
        index_temporal, index_fix = cnn_search.calculateIndex(fips, (cnn_search.startDay + datetime.timedelta(days=test_start_day)).isoformat())
        if (index_temporal != -1):
            for k in range(no_days):
                orginal_death = float(countiesData_temporal[index_temporal + k]['death'])
                prediction_death = predictions[k][fips]
                simple_death = float(countiesData_temporal[index_temporal + k - 14]['death'])

                orginals[k][fips] = orginal_death
                
                if (k >= no_days - 21):
                    sum_org += orginal_death
                    sum_predict += prediction_death
                    sum_MAE += abs(orginal_death - prediction_death)
                    sum_MASE += abs(orginal_death - simple_death)

                    sum_predict_round += round(prediction_death)
                    sum_MAE_round += abs(orginal_death - round(prediction_death))
        else:
            log('index = -1 | startDay={0}, fips={1}, index_fix={2}, test_start_day={3}'.format(cnn_search.startDay, fips, index_fix, test_start_day))

    MAE = sum_MAE / (21 * len(counties))
    MAPE = sum_MAE / sum_org
    MASE = MAE / (sum_MASE / (21 * len(counties)))

    MAE_round = sum_MAE_round / (21 * len(counties))
    MAPE_round = sum_MAE_round / sum_org
    MASE_round = MAE_round / (sum_MASE / (21 * len(counties)))

    return (MAE, MAPE, MASE, MAE_round, MAPE_round, MASE_round, orginals)

################################################################ main

if __name__ == "__main__":
    cnn_search.init_hashCounties()
    cnn_search.init_days()

    # Check if instances are ready
    if (os.path.exists('x_' + cnn_search._INSTANCES_FILENAME_) and os.path.exists('y_' + cnn_search._INSTANCES_FILENAME_)):
        log('instances found')
    else:
        raise Exception('no instances found')

    x_instances = load('x_' + cnn_search._INSTANCES_FILENAME_)
    y_instances = load('y_' + cnn_search._INSTANCES_FILENAME_)

    ################################################################ split imageArray into train, validation and test

    log('spliting data into train, validation and test')

    x_dataTrain = x_instances[:-63]
    y_dataTrain = y_instances[:-63]

    x_dataValidation = x_instances[-63:-42]
    y_dataValidation = y_instances[-63:-42]

    x_dataTest = x_instances[-42:-21]
    y_dataTest = y_instances[-42:-21]

    x_dataFinalTest = x_instances[-21:]
    y_dataFinalTest = y_instances[-21:]

    ################################################################ normalize data

    log('normalizing data')

    normal_x_dataTrain, normal_x_dataValidation, normal_x_dataTest, normal_x_dataFinalTest = cnn_search.normal_x(x_dataTrain, x_dataValidation, x_dataTest, x_dataFinalTest)
    normal_y_dataTrain, normal_y_dataValidation, normal_y_dataTest, normal_y_dataFinalTest, normal_min, normal_max = cnn_search.normal_y(y_dataTrain, y_dataValidation, y_dataTest, y_dataFinalTest)

    data_shape = normal_x_dataTrain.shape
    no_train = normal_x_dataTrain.shape[0]
    no_validation = normal_x_dataValidation.shape[0]
    no_test = normal_x_dataTest.shape[0]

    ################################################################ clearing memory

    del x_instances, x_dataTrain, x_dataValidation, x_dataTest, x_dataFinalTest
    del y_instances, y_dataTrain, y_dataValidation, y_dataTest, y_dataFinalTest

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
    result = predict_counties_result(target_counties, 
        model, 
        append(append(normal_x_dataTrain, normal_x_dataValidation).reshape(no_train + no_validation, data_shape[1], data_shape[2], data_shape[3]), normal_x_dataTest).reshape(no_train + no_validation + no_test, data_shape[1], data_shape[2], data_shape[3]), 
        append(append(normal_y_dataTrain, normal_y_dataValidation).reshape(no_train + no_validation, data_shape[1], data_shape[2], 1), normal_y_dataTest).reshape(no_train + no_validation + no_test, data_shape[1], data_shape[2], 1), 
        input_size, 
        normal_min, 
        normal_max)

    log('result for Pixels, MAE:{0}, MAPE:{1}, MASE:{2}'.format(result[0], result[1], result[2]))
    log('result for Country, MAE:{0}, MAPE:{1}, MASE:{2}'.format(result[3], result[4], result[5]))
    log('result for County, MAE:{0}, MAPE:{1}, MASE:{2}'.format(result[6], result[7], result[8]))
    log('result for County with rounded prediction, MAE:{0}, MAPE:{1}, MASE:{2}'.format(result[9], result[10], result[11]))


        