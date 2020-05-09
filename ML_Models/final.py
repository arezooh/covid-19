from makeHistoricalData import makeHistoricalData
from models import GBM, GLM, KNN, NN, MM_LR
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import seaborn as sns
from matplotlib import colors as mcolors
from pexecute.process import ProcessLoom
import time
from sys import argv
from math import floor
import os


r = 14
numberOfSelectedCounties = 500  # set to -1 for all the counties


######################################################### split data to train, val, test
def splitData(numberOfCounties, main_data, target, offset, j_offset):

    X = pd.DataFrame()
    y = pd.DataFrame()
    for i in range(numberOfCounties + 1):
        j = i * numberOfDays + j_offset
        X = X.append(main_data.loc[j:j + offset - 1])
        y = y.append(target.loc[j:j + offset - 1])

    return X, y


########################################################### clean data
def clean_data(data, numberOfSelectedCounties):

    global numberOfDays
    data = data.sort_values(by=['county_fips', 'date of day t'])
    # select the number of counties we want to use
    #numberOfSelectedCounties = numberOfCounties
    if numberOfSelectedCounties == -1:
        numberOfSelectedCounties = len(data['county_fips'].unique())

    using_data = data[(data['county_fips'] <= data['county_fips'].unique()[numberOfSelectedCounties - 1])]
    using_data = using_data.reset_index(drop=True)
    main_data = using_data.drop(['county_fips', 'state_fips', 'state_name', 'county_name', 'date of day t'],
                                axis=1)
    # target = pd.DataFrame(main_data['Target'])
    # main_data = main_data.drop(['Target'], axis=1)
    # numberOfCounties = len(using_data['county_fips'].unique())
    numberOfDays = len(using_data['date of day t'].unique())

    return main_data


########################################################### preprocess
def preprocess(main_data, validationFlag):

    target = pd.DataFrame(main_data['Target'])
    main_data = main_data.drop(['Target'], axis=1)
    # specify the size of train, validation and test sets
    test_offset = 14
    train_offset = floor(0.75 * (numberOfDays - test_offset))
    val_offset = numberOfDays - (train_offset + test_offset)
    t1 = time.time()
    # produce train, validation and test data in parallel
    loom = ProcessLoom(max_runner_cap=4)

    if validationFlag:     # validationFlag is 1 if we want to have a validation set and 0 otherwise
        # add the functions to the multiprocessing object, loom
        loom.add_function(splitData, [numberOfSelectedCounties, main_data, target, train_offset, 0], {})
        loom.add_function(splitData, [numberOfSelectedCounties, main_data, target, val_offset, train_offset], {})
        loom.add_function(splitData, [numberOfSelectedCounties, main_data, target, test_offset, train_offset + val_offset], {})
        # run the processes in parallel
        output = loom.execute()
        t2 = time.time()
        #print('total time of data splitting: ', t2 - t1)

        X_train_train = (output[0]['output'][0]).reset_index(drop=True)
        X_train_val = (output[1]['output'][0]).reset_index(drop=True)
        X_test = (output[2]['output'][0]).reset_index(drop=True)

        y_train_train = np.array(output[0]['output'][1]).reshape(-1)
        y_train_val = np.array(output[1]['output'][1]).reshape(-1)
        y_test = np.array(output[2]['output'][1]).reshape(-1)

        return X_train_train, X_train_val, X_test, y_train_train, y_train_val, y_test

    else:
        loom.add_function(splitData, [numberOfSelectedCounties, main_data, target, train_offset + val_offset, 0], {})
        loom.add_function(splitData, [numberOfSelectedCounties, main_data, target, test_offset, train_offset + val_offset], {})
        # run the processes in parallel
        output = loom.execute()
        t2 = time.time()
        #print('total time of data splitting: ', t2 - t1)

        X_train = (output[0]['output'][0]).reset_index(drop=True)
        X_test = (output[1]['output'][0]).reset_index(drop=True)

        y_train = np.array(output[0]['output'][1]).reshape(-1)
        y_test = np.array(output[1]['output'][1]).reshape(-1)

        return X_train, X_test, y_train, y_test


########################################################### run algorithms in parallel except mixed models
def run_algorithms(X_train_dict, X_val_dict, y_train_dict):

    print(X_train_dict['GBM'].shape, X_train_dict['NN'].shape)
    t1 = time.time()
    loom = ProcessLoom(max_runner_cap=4)
    # add the functions to the multiprocessing object, loom
    loom.add_function(GBM, [X_train_dict['GBM'], X_val_dict['GBM'], y_train_dict['GBM']], {})
    loom.add_function(GLM, [X_train_dict['GLM'], X_val_dict['GLM'], y_train_dict['GLM']], {})
    loom.add_function(KNN, [X_train_dict['KNN'], X_val_dict['KNN'], y_train_dict['KNN']], {})
    loom.add_function(NN, [X_train_dict['NN'], X_val_dict['NN'], y_train_dict['NN']], {})
    # run the processes in parallel
    output = loom.execute()
    t2 = time.time()
    print('total time - run algorithms: ', t2 - t1)

    return output[0]['output'], output[1]['output'], output[2]['output'], (output[3]['output']).reshape(-1)


########################################################### run mixed models in parallel
def run_mixed_models(X_train_MM, X_test_MM, y_train_MM):

    t1 = time.time()
    loom = ProcessLoom(max_runner_cap=2)
    # add the functions to the multiprocessing object, loom
    loom.add_function(MM_LR, [X_train_MM['MM_LR'], X_test_MM['MM_LR'], y_train_MM['MM_LR']], {})
    loom.add_function(NN, [X_train_MM['MM_NN'], X_test_MM['MM_NN'], y_train_MM['MM_NN']], {})
    # run the processes in parallel
    output = loom.execute()
    t2 = time.time()
    print('total time - run mixed models: ', t2 - t1)

    return output[0]['output'], (output[1]['output']).reshape(-1)


########################################################### generate data for best h and c
def generate_data(h, numberOfCovariates, covariates_names):

    data = makeHistoricalData(h, 14, 'confirmed')
    data = clean_data(data, numberOfSelectedCounties)
    X_train, X_test, y_train, y_test = preprocess(data, 0)
    covariates = [covariates_names[i] for i in range(numberOfCovariates)]
    best_covariates = []
    indx_c = 0
    for c in covariates:  # iterate through sorted covariates
        indx_c += 1
        for covariate in data.columns:  # add all historical covariates of this covariate and create a feature
            if c.split(' ')[0] in covariate:
                best_covariates.append(covariate)

    X_train = X_train[best_covariates]
    X_test = X_test[best_covariates]

    return X_train, X_test, y_train, y_test


########################################################### plot the results

def plot_results(maxHistory, row, col, numberOfCovariates, methods, history, errors, mode):

    mpl.style.use('seaborn')
    fig, ax = plt.subplots(row, col, figsize=(40, 40))
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

    # Sort colors by hue, saturation, value and name.
    by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
                    for name, color in colors.items())
    sorted_names = [name for hsv, name in by_hsv]
    colorset = set(sorted_names[::-1])
    for item in colorset:
        if ('white' in item) or ('light' in item):
            colorset = colorset - {item}
    colors = list(colorset - {'lavenderblush',  'aliceblue', 'lavender', 'azure',
         'mintcream', 'honeydew', 'beige', 'ivory', 'snow', 'w'})
    #colors = ['tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan',
     #         'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    ind = 0
    for i in range(row):
        for j in range(col):
            color = 0
            for h in history:
                errors_h = []
                # x label: covariates
                covariates_list = [c for c in range(1, numberOfCovariates + 1)]
                # y label: errors
                for c in range(1, numberOfCovariates + 1):
                    errors_h.append(errors[methods[ind]][(h, c)])
                ax[i, j].plot(covariates_list, errors_h, colors[color * 2], label="h = " + str(h))
                ax[i, j].set_xlabel("Number Of Covariates")
                # if mode == 'pe':
                #     ax[i, j].set_ylabel("Percentage Of Absolute Error")
                # else:
                ax[i, j].set_ylabel(mode)
                ax[i, j].set_title(str(methods[ind]))
                ax[i, j].legend()
                ax[i, j].set_xticks(covariates_list)
                color += 1
            ind += 1
    if not os.path.exists('results/counties=' + str(numberOfSelectedCounties) + ' max_history=' + str(maxHistory)):
        os.makedirs('results/counties=' + str(numberOfSelectedCounties) + ' max_history=' + str(maxHistory))
    plt.savefig('results/counties=' + str(numberOfSelectedCounties) + ' max_history=' + str(maxHistory) + '/' + str(mode)+'.png')


########################################################### plot table for final results
def plot_table(maxHistory, table_data, cols, name):
    fig = plt.figure(dpi=150)
    ax = fig.add_subplot(1, 1, 1)
    table = ax.table(cellText=table_data, colLabels=cols, loc='center')
    table.set_fontsize(14)
    table.scale(1, 5)
    ax.axis('off')
    if not os.path.exists('results/counties=' + str(numberOfSelectedCounties) + ' max_history=' + str(maxHistory)):
        os.makedirs('results/counties=' + str(numberOfSelectedCounties) + ' max_history=' + str(maxHistory))
    plt.savefig('results/counties=' + str(numberOfSelectedCounties) + ' max_history=' + str(maxHistory) + '/' + name + '.png')


########################################################### main
def main(maxHistory):

    history = [i for i in range(1, maxHistory + 1)]
    errors = {'GBM': {}, 'GLM': {}, 'KNN': {}, 'NN': {}, 'MM_LR': {}, 'MM_NN': {}}  # percentage of absolute errors
    mae_errors = {'GBM': {}, 'GLM': {}, 'KNN': {}, 'NN': {}, 'MM_LR': {}, 'MM_NN': {}}  # mean absolute errors
    methods = ['GBM', 'GLM', 'KNN', 'NN', 'MM_LR', 'MM_NN']
    none_mixed_methods = ['GBM', 'GLM', 'KNN', 'NN']
    mixed_methods = ['MM_LR', 'MM_NN']
    #historicalCovariates = {}  # covariates for each h
    minError = {'GBM': int(1e10), 'GLM': int(1e10), 'KNN': int(1e10), 'NN': int(1e10), 'MM_LR': int(1e10), 'MM_NN': int(1e10)}
    best_h = {}
    best_numberOfCovariates = {}
    historical_X_train = {}  # X_train for best h and c
    historical_X_test = {}  # X_test for best h and c
    historical_y_train = {}  # y_train for best h and c
    historical_y_test = {}  # y_test for best h and c

    target_name = 'confirmed'
    base_data = makeHistoricalData(0, r, target_name)
    base_data = clean_data(base_data, numberOfSelectedCounties)
    total_targets = sum(base_data['Target'])
    print('total targets:', total_targets)
    #sorted_covariates, covariates_names = correlatinosWithTarget(base_data)
    #sorted_covariates = sorted_covariates.drop(['Target'], axis=1)
    covariates_names = list(base_data.columns)
    covariates_names.remove('Target')
    numberOfCovariates = len(covariates_names)
    print('number of covariates: ', numberOfCovariates)
    for h in history:
        data = makeHistoricalData(h, 14, target_name)
        data = clean_data(data, numberOfSelectedCounties)
        print(data.shape)
        X_train_train, X_train_val, X_test, y_train_train, y_train_val, y_test = preprocess(data, 1)
        print(X_train_train.shape, X_train_val.shape, X_test.shape, y_train_train.shape, y_train_val.shape, y_test.shape)
        y_train = np.array((pd.DataFrame(y_train_train).append(pd.DataFrame(y_train_val))).reset_index(drop = True)).reshape(-1)
        #print(y_train.shape, y_test.shape)
        #historicalCovariates[h] = covariates
        #numberOfCovariates[h] = len(historicalCovariates[h]) - 1
        covariates_list = []
        # covariates are sorted by their correlation with Target. We start from the first important covariate and
        # in each loop we add the next important one
        # the first covariate is Target, we start from the second one

        y_prediction = {'GBM': {}, 'GLM': {}, 'KNN': {}, 'NN': {}, 'MM_LR': {}, 'MM_NN': {}}

        indx_c = 0
        for c in covariates_names:  # iterate through sorted covariates
            indx_c += 1
            for covariate in data.columns:  # add all historical covariates of this covariate and create a feature
                if c.split(' ')[0] in covariate:
                    covariates_list.append(covariate)

            X_train_train_temp = X_train_train[covariates_list]
            X_train_val_temp = X_train_val[covariates_list]
            X_test_temp = X_test[covariates_list]
            y_prediction.clear()
            X_train_train_dict, X_train_val_dict, y_train_train_dict, y_train_val_dict = {}, {}, {}, {}

            for method in methods:
                X_train_train_dict[method] = X_train_train_temp
                X_train_val_dict[method] = X_train_val_temp
                y_train_train_dict[method] = y_train_train
                y_train_val_dict[method] = y_train_val
            # run algorithms in parallel on train and validation data
            y_prediction['GBM'], y_prediction['GLM'], y_prediction['KNN'], y_prediction['NN'] = run_algorithms(X_train_train_dict,
                                            X_train_val_dict, y_train_train_dict)
            y_predictions = []

            # Construct the outputs for the training dataset of the 'MM' methods
            y_predictions.extend([y_prediction['GBM'], y_prediction['GLM'], y_prediction['KNN'], y_prediction['NN']])
            y_prediction_np = np.array(y_predictions).reshape(len(y_predictions), -1)
            X_mixedModel = pd.DataFrame(y_prediction_np.transpose())
            X_train_MM, X_test_MM, y_train_MM, y_test_MM = train_test_split(X_mixedModel, y_train_val, test_size=0.25)

            X_train_MM_dict, X_test_MM_dict, y_train_MM_dict, y_test_MM_dict = {}, {}, {}, {}
            for mixed_method in mixed_methods:
                X_train_MM_dict[mixed_method] = X_train_MM
                X_test_MM_dict[mixed_method] = X_test_MM
                y_train_MM_dict[mixed_method] = y_train_MM
                y_test_MM_dict[mixed_method] = y_test_MM
            # mixed model with linear regression and neural network
            y_prediction['MM_LR'], y_prediction['MM_NN'] = run_mixed_models(X_train_MM_dict, X_test_MM_dict, y_train_MM_dict)

            y_train_val_temp = y_train_val
            # compute errors and find best h and c
            for method in methods:
                if method == 'MM_NN' or method == 'MM_LR':
                    y_train_val_temp = y_test_MM
                meanAbsoluteError = mean_absolute_error(y_train_val_temp, y_prediction[method])
                print("Mean Absolute Error of ", method, " for h =", h, "and #covariates =", indx_c, ": %.4f" % meanAbsoluteError)
                mae_errors[method][(h, indx_c)] = meanAbsoluteError
                if minError[method] > meanAbsoluteError:
                    minError[method] = meanAbsoluteError
                    best_h[method] = h
                    best_numberOfCovariates[method] = indx_c
                    if method != 'MM_LR' and method != 'MM_NN':
                        historical_X_train[method] = (X_train_train_temp.append(X_train_val_temp)).reset_index(drop=True)
                        historical_X_test[method] = X_test_temp
                        historical_y_train[method] = y_train
                        historical_y_test[method] = y_test
                sumOfAbsoluteError = sum(abs(y_train_val_temp - y_prediction[method]))
                percentageOfAbsoluteError = (sumOfAbsoluteError / sum(y_train_val_temp)) * 100
                errors[method][(h, indx_c)] = percentageOfAbsoluteError
                print("Percentage of Absolute Error of ", method, " for h =", h, "and #covariates =", indx_c, ": %.4f" % percentageOfAbsoluteError)

    # plot the results of methods on validation set
    plot_results(maxHistory, 3, 2, numberOfCovariates, methods, history, errors, 'Percentage Of Absolute Error')
    plot_results(maxHistory, 3, 2, numberOfCovariates, methods, history, mae_errors, 'Mean Absolute Error')

    columns_table = ['method', 'best_h', 'best_c', 'sum of absolute error', 'mean absolute error',
                     'percentage of absolute error']  # table columns names
    y_prediction = {}
    # run non-mixed methods on the whole training set with their best h and c
    X_train_dict, X_test_dict, y_train_dict, y_test_dict = {}, {}, {}, {}

    y_prediction['GBM'], y_prediction['GLM'], y_prediction['KNN'], y_prediction['NN'] = run_algorithms(
        historical_X_train, historical_X_test, historical_y_train)

    table_data = []
    for method in none_mixed_methods:
        meanAbsoluteError = mean_absolute_error(historical_y_test[method], y_prediction[method])
        print("Mean Absolute Error of ", method, " for h =", best_h[method], "and #covariates =",
              best_numberOfCovariates[method], ": %.4f" % meanAbsoluteError)
        sumOfAbsoluteError = sum(abs(historical_y_test[method] - y_prediction[method]))
        percentageOfAbsoluteError = (sumOfAbsoluteError / sum(historical_y_test[method])) * 100
        print("Percentage of Absolute Error of ", method, " for h =", best_h[method], "and #covariates =",
              best_numberOfCovariates[method], ": %.4f" % percentageOfAbsoluteError)
        table_data.append(
            [method, best_h[method], best_numberOfCovariates[method], sumOfAbsoluteError, meanAbsoluteError, percentageOfAbsoluteError])
        result = pd.DataFrame(historical_y_test[method], columns=['y_test'])
        result['y_prediction'] = y_prediction[method]
        result['absolute_error'] = abs(historical_y_test[method] - y_prediction[method])
        result.to_csv('results/counties=' + str(numberOfSelectedCounties) + ' max_history=' + str(maxHistory) + '/' + method + '.csv')
    table_name = 'non-mixed methods best results'
    plot_table(maxHistory, table_data, columns_table, table_name)

    # generate data for non-mixed methods with the best h and c of mixed models and fit mixed models on them
    # (with the whole training set)
    y_predictions = {'MM_LR': [], 'MM_NN':[]}
    y_prediction = {}
    table_data = []
    X_train_MM_dict, X_test_MM_dict, y_train_MM_dict, y_test_MM_dict = {}, {}, {}, {}
    for mixed_method in mixed_methods:
        y_test = None
        for method in none_mixed_methods:
            X_train, X_test, y_train, y_test = generate_data(best_h[mixed_method], best_numberOfCovariates[mixed_method], covariates_names)
            print('best MM:', X_train.shape, X_test.shape, y_train.shape, y_test.shape)
            X_train_dict[method] = X_train
            X_test_dict[method] = X_test
            y_train_dict[method] = y_train
            y_test_dict[method] = y_test

        y_prediction['GBM'], y_prediction['GLM'], y_prediction['KNN'], y_prediction['NN'] = run_algorithms(
                X_train_dict, X_test_dict, y_train_dict)
        y_predictions[mixed_method].extend([y_prediction['GBM'], y_prediction['GLM'], y_prediction['KNN'], y_prediction['NN']])
        y_prediction_np = np.array(y_predictions[mixed_method]).reshape(len(y_predictions[mixed_method]), -1)
        X_mixedModel = pd.DataFrame(y_prediction_np.transpose())
        print(X_mixedModel.shape)
        X_train_MM, X_test_MM, y_train_MM, y_test_MM = train_test_split(X_mixedModel, y_test, test_size=0.25)
        X_train_MM_dict[mixed_method] = X_train_MM
        X_test_MM_dict[mixed_method] = X_test_MM
        y_train_MM_dict[mixed_method] = y_train_MM
        y_test_MM_dict[mixed_method] = y_test_MM

    # mixed model with linear regression and neural network
    y_prediction['MM_LR'], y_prediction['MM_NN'] = run_mixed_models(X_train_MM_dict, X_test_MM_dict, y_train_MM_dict)
    print(y_prediction['MM_LR'].shape, y_prediction['MM_NN'].shape)
    for mixed_method in mixed_methods:
        meanAbsoluteError = mean_absolute_error(y_test_MM_dict[mixed_method], y_prediction[mixed_method])
        print("Mean Absolute Error of ", mixed_method, " for h =", best_h[mixed_method], "and #covariates =",
              best_numberOfCovariates[mixed_method], ": %.4f" % meanAbsoluteError)
        sumOfAbsoluteError = sum(abs(y_test_MM_dict[mixed_method] - y_prediction[mixed_method]))
        percentageOfAbsoluteError = (sumOfAbsoluteError / sum(y_test_MM_dict[mixed_method])) * 100
        print("Percentage of Absolute Error of ", mixed_method, " for h =", best_h[mixed_method], "and #covariates =",
              best_numberOfCovariates[mixed_method], ": %.4f" % percentageOfAbsoluteError)
        table_data.append(
            [mixed_method, best_h[mixed_method], best_numberOfCovariates[mixed_method], sumOfAbsoluteError,
             meanAbsoluteError, percentageOfAbsoluteError])
        result = pd.DataFrame(y_test_MM_dict[mixed_method], columns=['y_test'])
        result['y_prediction'] = y_prediction[mixed_method]
        result['absolute_error'] = abs(y_test_MM_dict[mixed_method] - y_prediction[mixed_method])
        result.to_csv('results/counties=' + str(numberOfSelectedCounties) + ' max_history=' + str(maxHistory) + '/' + mixed_method + '.csv')
    table_name = 'mixed methods best results'
    plot_table(maxHistory, table_data, columns_table, table_name)


if __name__ == "__main__":
    begin = time.time()
    maxHistory = 14
    main(maxHistory)
    end = time.time()
    print("The total time of execution: ", end - begin)