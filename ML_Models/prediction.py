from makeHistoricalData import makeHistoricalData
from models import GBM, GLM, KNN, NN, MM_LR
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import seaborn as sns
from matplotlib import colors as mcolors
from pexecute.process import ProcessLoom
import time
from sys import argv
import sys
from math import floor, sqrt
import os
import dill
import subprocess as cmd
import shelve


r = 14  # the following day to predict
numberOfSelectedCounties = 50


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
    # numberOfSelectedCounties = numberOfCounties
    if numberOfSelectedCounties == -1:
        numberOfSelectedCounties = len(data['county_fips'].unique())

    using_data = data[(data['county_fips'] <= data['county_fips'].unique()[numberOfSelectedCounties - 1])]
    using_data = using_data.reset_index(drop=True)
    main_data = using_data.drop(['county_fips', 'state_fips', 'state_name'],
                                axis=1)  # , 'date of day t'
    # target = pd.DataFrame(main_data['Target'])
    # main_data = main_data.drop(['Target'], axis=1)
    # numberOfCounties = len(using_data['county_fips'].unique())
    numberOfDays = len(using_data['date of day t'].unique())

    return main_data


########################################################### preprocess
def preprocess(main_data, validationFlag):

    target = pd.DataFrame(main_data[['date of day t', 'county_name', 'Target']])
    main_data = main_data.drop(['Target', 'date of day t', 'county_name'], axis=1)
    # main_data = main_data.drop(['date of day t'], axis=1)
    # main_data = main_data.drop(['county_name'], axis=1)
    # specify the size of train, validation and test sets
    test_offset = r
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

        y_train_train = output[0]['output'][1]
        y_train_val = output[1]['output'][1]
        y_test = output[2]['output'][1]

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

        y_train = output[0]['output'][1]
        y_test = output[1]['output'][1]

        return X_train, X_test, y_train, y_test


########################################################### run non-mixed methods in parallel
def parallel_run(method, X_train_train, X_train_val, y_train_train, y_train_val):
    y_prediction, y_prediction_train = None, None
    if method == 'GBM':
        y_prediction, y_prediction_train = GBM(X_train_train, X_train_val, y_train_train)
    elif method == 'GLM':
        y_prediction, y_prediction_train = GLM(X_train_train, X_train_val, y_train_train, y_train_val)
    elif method == 'KNN':
        y_prediction, y_prediction_train = KNN(X_train_train, X_train_val, y_train_train)
    elif method == 'NN':
        y_prediction, y_prediction_train = NN(X_train_train, X_train_val, y_train_train, y_train_val)

    return y_prediction, y_prediction_train


########################################################### run mixed methods in parallel
def mixed_parallel_run(method, X_train, X_test, y_train, y_test):

    y_prediction, y_prediction_train = None, None
    if method == 'MM_LR':
        y_prediction, y_prediction_train = MM_LR(X_train, X_test, y_train)
    elif method == 'MM_NN':
        y_prediction, y_prediction_train = NN(X_train, X_test, y_train, y_test)

    return y_prediction, y_prediction_train


########################################################### run algorithms in parallel except mixed models
def run_algorithms(X_train_dict, X_val_dict, y_train_dict, y_val_dict):

    t1 = time.time()
    loom = ProcessLoom(max_runner_cap=4)
    # add the functions to the multiprocessing object, loom
    loom.add_function(GBM, [X_train_dict['GBM'], X_val_dict['GBM'], y_train_dict['GBM']], {})
    loom.add_function(GLM, [X_train_dict['GLM'], X_val_dict['GLM'], y_train_dict['GLM'], y_val_dict['GLM']], {})
    loom.add_function(KNN, [X_train_dict['KNN'], X_val_dict['KNN'], y_train_dict['KNN']], {})
    loom.add_function(NN, [X_train_dict['NN'], X_val_dict['NN'], y_train_dict['NN'], y_val_dict['NN']], {})
    # run the processes in parallel
    output = loom.execute()
    t2 = time.time()
    print('total time - run algorithms: ', t2 - t1)

    return output[0]['output'], output[1]['output'], output[2]['output'], output[3]['output']


########################################################### run mixed models in parallel
def run_mixed_models(X_train_MM, X_test_MM, y_train_MM, y_test_MM):

    t1 = time.time()
    loom = ProcessLoom(max_runner_cap=2)
    # add the functions to the multiprocessing object, loom
    loom.add_function(MM_LR, [X_train_MM['MM_LR'], X_test_MM['MM_LR'], y_train_MM['MM_LR']], {})
    loom.add_function(NN, [X_train_MM['MM_NN'], X_test_MM['MM_NN'], y_train_MM['MM_NN'], y_test_MM['MM_NN']], {})
    # run the processes in parallel
    output = loom.execute()
    t2 = time.time()
    print('total time - run mixed models: ', t2 - t1)

    return output[0]['output'], output[1]['output']


########################################################### generate data for best h and c
def generate_data(h, numberOfCovariates, covariates_names):

    data = makeHistoricalData(h, 14, 'confirmed', 'mrmr', str(argv[1]))
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

def plot_results(row, col, numberOfCovariates, methods, history, errors, mode):

    mpl.style.use('seaborn')
    plt.rc('font', size=20)
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
                ax[i, j].set_ylabel(mode)
                ax[i, j].set_title(str(methods[ind]))
                ax[i, j].legend()
                ax[i, j].set_xticks(covariates_list)
                color += 1
            ind += 1
    address = validation_address + 'plots_of_errors/'
    if not os.path.exists(address):
        os.makedirs(address)
    plt.savefig(address + str(mode)+'.png')


########################################################### plot table for final results
def plot_table(table_data, col_labels, row_labels, name, mode):
    fig = plt.figure() #dpi=50 figsize=(30, 10)
    ax = fig.add_subplot(111)
    colWidths = [0.1, 0.1, 0.25, 0.25, 0.25, 0.25, 0.25]
    address = ''
    if mode == 'val':
        colWidths.pop()
        address = validation_address + 'tables/'
        if not os.path.exists(address):
            os.makedirs(address)
    else:
        address = test_address + 'tables/'
        if not os.path.exists(address):
            os.makedirs(address)
    the_table = plt.table(cellText=table_data,
                          colWidths=colWidths,
                          rowLabels=row_labels,
                          colLabels=col_labels,
                          loc='center',
                          cellLoc='center')
    the_table.auto_set_font_size(False)
    # if mode == 'test':
    #     the_table.set_fontsize(12)
    # else:
    #
    the_table.set_fontsize(9)
    the_table.scale(1.5, 1.5)
    ax.axis('off')

    plt.savefig(address + name + '.png', bbox_inches='tight')


########################################################### plotting mean errors (first error)
def plot_targets(method, x_axis, df, main_address):
    mpl.style.use('default')
    plt.rc('font', size=40)
    fig, ax = plt.subplots(figsize=(60, 20))
    ax.plot(x_axis, df['average of targets'], label='Target')
    ax.plot(x_axis, df['average of predictions'], label='Prediction')
    ax.set_xlabel('date', fontsize=40)
    ax.set_ylabel('real and predicted targets for ' + str(method), fontsize=40)
    ax.legend()
    address = main_address + 'procedure_of_prediction/'
    if not os.path.exists(address):
        os.makedirs(address)
    plt.savefig(address +'procedure_'+ str(method) +'.png')


########################################################### box plots and violin plots
def box_violin_plot(X, Y, figsizes, fontsizes, name, address):
    mpl.style.use('default')
    # box plot
    fig = plt.figure(figsize=figsizes['box'])
    plt.rc('font', size=fontsizes['box'])
    plt.locator_params(axis='y', nbins=20)
    sns.boxplot(x=X, y=Y)
    plt.savefig(address + str(name) + 'boxplot.png')
    plt.close()
    # violin plot
    fig = plt.figure(figsize=figsizes['violin'])
    plt.rc('font', size=fontsizes['violin'])
    plt.locator_params(axis='y', nbins=20)
    sns.violinplot(x=X, y=Y)
    plt.savefig(address + str(name) + 'violinplot.png')
    plt.close()


########################################################### get errors for each model in each h and c
def get_errors(h, c, method, y_prediction, y_test_date, mode):
    # make predictions rounded to their closest number and make the negatives ones zero
    y_prediction = np.round(y_prediction)
    y_prediction[y_prediction < 0] = 0
    # write outputs into a file
    orig_stdout = sys.stdout
    f = open(env_address+'out.txt', 'a')
    sys.stdout = f

    # if mode == 'val': y_test_date would be an np.array with the target date
    # if mode == 'test': y_test_date would be a dataframe with columns ['date of day t', 'county_name', 'Target']
    y_test = y_test_date
    if mode == 'test':  # use the 'Target' column for computing main errors
        y_test = np.array(y_test_date['Target']).reshape(-1)

    meanAbsoluteError = mean_absolute_error(y_test, y_prediction)
    print("Mean Absolute Error of ", method, " for h =", h, "and #covariates =", c, ": %.2f" % meanAbsoluteError)
    #sumOfAbsoluteError = sum(abs(y_test - y_prediction))
    #percentageOfAbsoluteError = (sumOfAbsoluteError / sum(y_test)) * 100
    # we change zero targets into 1 and add 1 to their predictions
    y_test_temp = y_test
    y_test_temp[y_test == 0] = 1
    y_prediction_temp = y_prediction
    y_prediction_temp[y_test == 0] += 1
    meanPercentageOfAbsoluteError = sum((abs(y_prediction_temp - y_test_temp) / y_test_temp) * 100) / len(y_test)
    print("Mean Percentage of Absolute Error of ", method, " for h =", h, "and #covariates =", c,
          ": %.2f" % meanPercentageOfAbsoluteError)
    rootMeanSquaredError = sqrt(mean_squared_error(y_test, y_prediction))
    print("Root Mean Squared Error of ", method, " for h =", h, "and #covariates =", c, ": %.2f" % rootMeanSquaredError)
    ### compute adjusted R squared error
    SS_Residual = sum((y_test - y_prediction.reshape(-1)) ** 2)
    SS_Total = sum((y_test - np.mean(y_test)) ** 2)
    r_squared = 1 - (float(SS_Residual)) / SS_Total
    adj_r_squared = 1 - (1 - r_squared) * (len(y_test) - 1) / (len(y_test) - c - 1)
    print("Adjusted R Squared Error of ", method, " for h =", h, "and #covariates =", c, ": %.2f" % adj_r_squared)
    print("-----------------------------------------------------------------------------------------")

    # for the test mode we compute some additional errors, we need 'date of day t' column so we use the main dataframe
    # we add our prediction, the difference between prediction and target ('error' column),
    # the absolute difference between prediction and target ('absolute_error' column),
    # the precentage of this difference (('percentage_error' column) -> we change zero targets into 1 and add 1 to their predictions),
    # and second_error as follows and save these in 'all_errors' file
    # then we compute the average of percentage_errors (and other values) in each day and save them in
    # 'first_error' file

    if mode == 'test':
        first_error_address = test_address + 'averages_of_errors_in_each_day/'
        all_errors_address = test_address + 'all_errors/' + str(method) + '/'
        if not os.path.exists(first_error_address):
            os.makedirs(first_error_address)
        if not os.path.exists(all_errors_address):
            os.makedirs(all_errors_address)
        dataframe = pd.DataFrame(y_test_date, copy=True)
        dataframe['prediction'] = y_prediction
        dataframe['error'] = y_prediction - y_test
        dataframe['absoulte_error'] = abs(y_prediction - y_test)
        y_test_temp = y_test
        y_test_temp[y_test == 0] = 1
        y_prediction_temp = y_prediction
        y_prediction_temp[y_test == 0] += 1
        dataframe['percentage_error'] = ((abs(y_prediction_temp - y_test_temp)) / y_test_temp) * 100
        second_error = (sum(dataframe['error']) / sum(y_test)) * 100
        dataframe.to_csv(all_errors_address + 'all_errors_' + str(method) + '.csv')
        box_violin_plot(dataframe['date of day t'], dataframe['percentage_error'], figsizes={'box': (60, 30), 'violin': (100, 50)},
                        fontsizes={'box' : 40, 'violin': 60}, name=str(method) + '_percentage_errors_in_each_day_',
                        address=all_errors_address)
        box_violin_plot(dataframe['date of day t'], dataframe['error'], figsizes={'box': (20, 10), 'violin': (50, 30)},
                        fontsizes={'box': 15, 'violin': 30}, name=str(method) + '_pure_errors_in_each_day_',
                        address=all_errors_address)
        first_error = pd.DataFrame((dataframe.groupby(['date of day t']).sum() / numberOfSelectedCounties))
        first_error.columns = ['average of targets', 'average of predictions', 'average of errors',
                               'average of absoulte_errors', 'average of percentage_errors']
        first_error.to_csv(first_error_address + 'first_error_' + str(method) + '.csv')
        plot_targets(method, first_error.index, first_error, first_error_address)
        # save outputs in 'out.txt'
        sys.stdout = orig_stdout
        f.close()
        return meanAbsoluteError, rootMeanSquaredError, meanPercentageOfAbsoluteError, adj_r_squared, second_error
    else:
        # save outputs in 'out.txt'
        sys.stdout = orig_stdout
        f.close()
        return meanAbsoluteError, rootMeanSquaredError, meanPercentageOfAbsoluteError, adj_r_squared


########################################################### push results to github
def push(message):
    try:
        cmd.run("git pull", check=True, shell=True)
        print("everything has been pulled")
        cmd.run("git add .", check=True, shell=True)
        #message = 'new results added'
        cmd.run(f"git commit -m '{message}'", check=True, shell=True)
        cmd.run("git push", check=True, shell=True)
        print('pushed.')
    except:
        print('could not push')


########################################################### main
def main(maxHistory):

    history = [i for i in range(1, maxHistory + 1)]
    methods = ['GBM', 'GLM', 'KNN', 'NN', 'MM_LR', 'MM_NN']
    none_mixed_methods = ['GBM', 'GLM', 'KNN', 'NN']
    mixed_methods = ['MM_LR', 'MM_NN']
    target_name = 'confirmed'
    base_data = makeHistoricalData(0, r, target_name, 'mrmr', str(argv[1]))
    base_data = clean_data(base_data, numberOfSelectedCounties)
    covariates_names = list(base_data.columns)
    covariates_names.remove('Target')
    covariates_names.remove('date of day t')
    covariates_names.remove('county_name')
    numberOfCovariates = len(covariates_names)
    print('number of covariates: ', numberOfCovariates)
    y_prediction = {'GBM': {}, 'GLM': {}, 'KNN': {}, 'NN': {}, 'MM_LR': {}, 'MM_NN': {}}
    y_prediction_train = {'GBM': {}, 'GLM': {}, 'KNN': {}, 'NN': {}, 'MM_LR': {}, 'MM_NN': {}}

    y_test_MM = {'MM_LR': {}, 'MM_NN': {}}
    best_h = {}
    best_c = {}
    error_names = ['MAPE', 'MAE', 'RMSE', 'adj-R2']
    # minError = {'GBM': int(1e10), 'GLM': int(1e10), 'KNN': int(1e10), 'NN': int(1e10), 'MM_LR': int(1e10),
    #             'MM_NN': int(1e10)}
    minError = {}
    for method in methods:
        minError[method] = {}
        best_c[method] = {}
        best_h[method] = {}
        for error in error_names:
            minError[method][error] = int(1e10)
            best_h[method][error] = 0
            best_c[method][error] = 0
    # minError_train = {}
    # for method in methods:
    #     minError_train[method] = {'MAPE' : int(1e10)}

    percentage_errors_train = {'GBM': {}, 'GLM': {}, 'KNN': {}, 'NN': {}, 'MM_LR': {}, 'MM_NN': {}}  # percentage of absolute errors
    percentage_errors = {'GBM': {}, 'GLM': {}, 'KNN': {}, 'NN': {}, 'MM_LR': {}, 'MM_NN': {}}  # percentage of absolute errors
    mae_errors = {'GBM': {}, 'GLM': {}, 'KNN': {}, 'NN': {}, 'MM_LR': {}, 'MM_NN': {}}  # mean absolute errors
    rmse_errors = {'GBM': {}, 'GLM': {}, 'KNN': {}, 'NN': {}, 'MM_LR': {}, 'MM_NN': {}}  # root mean squared errors
    adjR2_errors = {'GBM': {}, 'GLM': {}, 'KNN': {}, 'NN': {}, 'MM_LR': {}, 'MM_NN': {}}  # adjusted R squared errors
    columns_table = ['best_h', 'best_c', 'root mean squared error', 'mean absolute error',
                     'percentage of absolute error', 'adjusted R squared error', 'second_error']  # table columns names
    historical_X_train = {}  # X_train for best h and c
    historical_X_test = {}  # X_test for best h and c
    historical_y_train = {}  # y_train for best h and c
    historical_y_test = {}  # y_test for best h and c
    historical_y_train_date = {}  # y_train for best h and c with dates info
    historical_y_test_date = {}  # y_test for best h and c with dates info
    parallel_outputs = {}

    for h in history:
        data = makeHistoricalData(h, 14, target_name, 'mrmr', str(argv[1]))
        data = clean_data(data, numberOfSelectedCounties)
        # pre-process and split the data, 'date's have dates info
        X_train_train, X_train_val, X_test, y_train_train_date, y_train_val_date, y_test_date = preprocess(data, 1)
        y_train_date = (pd.DataFrame(y_train_train_date).append(pd.DataFrame(y_train_val_date))).reset_index(drop=True)
        y_train_train = np.array(y_train_train_date['Target']).reshape(-1)
        y_train_val = np.array(y_train_val_date['Target']).reshape(-1)
        y_test = np.array(y_test_date['Target']).reshape(-1)
        y_train = np.array((pd.DataFrame(y_train_train).append(pd.DataFrame(y_train_val))).reset_index(drop=True)).reshape(-1)

        covariates_list = []
        # covariates are sorted by their correlation with Target. We start from the first important covariate and
        # in each loop we add the next important one
        # the first covariate is Target, we start from the second one

        # initiate loom for parallel processing
        loom = ProcessLoom(max_runner_cap=len(base_data.columns) * len(none_mixed_methods) + 5)
        indx_c = 0
        for c in covariates_names:  # iterate through sorted covariates
            indx_c += 1
            print('h=', h, ' c=', indx_c)
            for covariate in data.columns:  # add all historical covariates of this covariate and create a feature
                if c.split(' ')[0] in covariate:
                    covariates_list.append(covariate)
            X_train_train_temp = X_train_train[covariates_list]
            X_train_val_temp = X_train_val[covariates_list]
            for method in none_mixed_methods:
                loom.add_function(parallel_run, [method, X_train_train_temp, X_train_val_temp, y_train_train, y_train_val])
        # run the processes in parallel
        parallel_outputs['non_mixed'] = loom.execute()
        ind = 0
        for c in range(1, numberOfCovariates + 1):
            for method in none_mixed_methods:
                y_prediction[method][(h, c)], y_prediction_train[method][(h, c)] = parallel_outputs['non_mixed'][ind]['output']
                ind += 1
        # save the entire session for each h and c
        filename = env_address + 'validation.out'
        my_shelf = shelve.open(filename, 'n')  # 'n' for new
        for key in dir():
            try:
                my_shelf[key] = locals()[key]
            except:
                print('ERROR shelving: {0}'.format(key))
        my_shelf.close()
        # initiate loom for parallel processing
        loom = ProcessLoom(max_runner_cap=len(base_data.columns) * len(mixed_methods) + 5)
        indx_c = 0
        for c in range(1, numberOfCovariates + 1):
            indx_c += 1
            for mixed_method in mixed_methods:
                y_predictions_test, y_predictions_train = [], []
                # Construct the outputs for the testing dataset of the 'MM' methods
                # y_prediction['NN'][(h, c)] = np.array(y_prediction['NN'][(h, c)]).ravel()
                y_predictions_test.extend([y_prediction['GBM'][(h, c)], y_prediction['GLM'][(h, c)],
                                      y_prediction['KNN'][(h, c)], y_prediction['NN'][(h, c)]])
                y_prediction_test_np = np.array(y_predictions_test).reshape(len(y_predictions_test), -1)
                X_test_mixedModel = pd.DataFrame(y_prediction_test_np.transpose())
                # Construct the outputs for the training dataset of the 'MM' methods
                # y_prediction_train['NN'][(h, c)] = np.array(y_prediction_train['NN'][(h, c)]).ravel()
                y_predictions_train.extend([y_prediction_train['GBM'][(h, c)], y_prediction_train['GLM'][(h, c)],
                                      y_prediction_train['KNN'][(h, c)], y_prediction_train['NN'][(h, c)]])
                y_prediction_train_np = np.array(y_predictions_train).reshape(len(y_predictions_train), -1)
                X_train_mixedModel = pd.DataFrame(y_prediction_train_np.transpose())
                loom.add_function(mixed_parallel_run, [mixed_method, X_train_mixedModel, X_test_mixedModel, y_train_train, y_train_val])
                # print(X_train_mixedModel)
        # run the processes in parallel
        parallel_outputs['mixed'] = loom.execute()
        ind = 0
        for c in range(1, numberOfCovariates + 1):
            for mixed_method in mixed_methods:
                y_prediction[mixed_method][(h, c)], y_prediction_train[mixed_method][(h, c)] = parallel_outputs['mixed'][ind]['output']
                print()
                # y_prediction[mixed_method][(h, c)] = np.array(y_prediction[mixed_method][(h, c)]).ravel()
                # y_prediction_train[mixed_method][(h, c)] = np.array(y_prediction_train[mixed_method][(h, c)]).ravel()
                ind += 1
        # print( y_prediction['MM_LR'][(h, 1)], y_prediction_train['MM_LR'][(h, 1)])
        # save the entire session for each h and c
        filename = env_address + 'validation.out'
        my_shelf = shelve.open(filename, 'n')  # 'n' for new
        for key in dir():
            try:
                my_shelf[key] = locals()[key]
            except:
                print('ERROR shelving: {0}'.format(key))
        my_shelf.close()

        indx_c = 0
        for c in covariates_names:  # iterate through sorted covariates
            indx_c += 1
            for covariate in data.columns:  # add all historical covariates of this covariate and create a feature
                if c.split(' ')[0] in covariate:
                    covariates_list.append(covariate)
            X_train_train_temp = X_train_train[covariates_list]
            X_train_val_temp = X_train_val[covariates_list]
            X_test_temp = X_test[covariates_list]
            y_val = np.array(y_train_val_date['Target']).reshape(-1)

            for method in methods:
                mae_errors[method][(h, indx_c)], rmse_errors[method][(h, indx_c)], percentage_errors[method][(h, indx_c)], \
                adjR2_errors[method][(h, indx_c)] = get_errors(h, indx_c, method, y_prediction[method][(h, indx_c)], y_val, mode='val')
                # percentage error on train_train
                # print(y_prediction_train[method][(h, indx_c)].shape)
                # print(y_prediction_train[method][(h, indx_c)])
                dummy1, dummy2, percentage_errors_train[method][(h, indx_c)], dummy3 = get_errors(h, indx_c, method,
                    y_prediction_train[method][(h, indx_c)], y_train_train, mode='val')
                # find best errors
                if mae_errors[method][(h, indx_c)] < minError[method]['MAE']:
                    minError[method]['MAE'] = mae_errors[method][(h, indx_c)]
                    best_h[method]['MAE'] = h
                    best_c[method]['MAE'] = indx_c
                if rmse_errors[method][(h, indx_c)] < minError[method]['RMSE']:
                    minError[method]['RMSE'] = rmse_errors[method][(h, indx_c)]
                    best_h[method]['RMSE'] = h
                    best_c[method]['RMSE'] = indx_c
                if adjR2_errors[method][(h, indx_c)] < minError[method]['adj-R2']:
                    minError[method]['adj-R2'] = adjR2_errors[method][(h, indx_c)]
                    best_h[method]['adj-R2'] = h
                    best_c[method]['adj-R2'] = indx_c
                if percentage_errors[method][(h, indx_c)] < minError[method]['MAPE']:
                    minError[method]['MAPE'] = percentage_errors[method][(h, indx_c)]
                    best_h[method]['MAPE'] = h
                    best_c[method]['MAPE'] = indx_c
                    if method != 'MM_LR' and method != 'MM_NN':
                        historical_X_train[method] = (X_train_train_temp.append(X_train_val_temp)).reset_index(drop=True)
                        historical_X_test[method] = X_test_temp
                        historical_y_train[method] = y_train
                        historical_y_test[method] = y_test
                        historical_y_train_date[method] = y_train_date
                        historical_y_test_date[method] = y_test_date
                # if percentage_errors_train[method][(h, indx_c)] < minError_train[method]['MAPE']:
                #     minError[method]['MAPE'] = percentage_errors_train[method][(h, indx_c)]


            # save the entire session for each h and c
            filename = env_address + 'validation.out'
            my_shelf = shelve.open(filename, 'n')  # 'n' for new
            for key in dir():
                try:
                    my_shelf[key] = locals()[key]
                except:
                    print('ERROR shelving: {0}'.format(key))
            my_shelf.close()
        # save the entire session for each h
        filename = env_address + 'validation.out'
        my_shelf = shelve.open(filename, 'n')  # 'n' for new
        for key in dir():
            try:
                my_shelf[key] = locals()[key]
            except:
                print('ERROR shelving: {0}'.format(key))
        my_shelf.close()

        # push the file of outputs
        #push('logs of h=' + str(h) + ' added')
    # plot table for best results
    table_data = []
    for method in methods:
        table_data.append([best_h[method]['MAPE'], best_c[method]['MAPE'], round(minError[method]['RMSE'], 2),
                           round(minError[method]['MAE'], 2), round(minError[method]['MAPE'], 2), round(minError[method]['adj-R2'], 2)])
    table_name = 'tabel_of_best_validation_results'
    plot_table(table_data, columns_table, methods, table_name, mode='val')
    # plot the results of methods on validation set
    plot_results(3, 2, numberOfCovariates, methods, history, percentage_errors_train, 'Percentage Of Absolute Error On train_train')
    plot_results(3, 2, numberOfCovariates, methods, history, percentage_errors, 'Percentage Of Absolute Error')
    plot_results(3, 2, numberOfCovariates, methods, history, mae_errors, 'Mean Absolute Error')
    plot_results(3, 2, numberOfCovariates, methods, history, rmse_errors, 'Root Mean Squared Error')
    plot_results(3, 2, numberOfCovariates, methods, history, adjR2_errors, 'Adjusted R Squared Error')
    #push('plots added')
    #################################################################################################################

    y_prediction = {}
    # run non-mixed methods on the whole training set with their best h and c
    X_train_dict, X_test_dict, y_train_dict, y_test_dict = {}, {}, {}, {}

    GBM, GLM, KNN, NN = run_algorithms(historical_X_train, historical_X_test, historical_y_train, historical_y_test)

    y_prediction['GBM'], y_prediction_train['GBM'] = GBM
    y_prediction['GLM'], y_prediction_train['GLM'] = GLM
    y_prediction['KNN'], y_prediction_train['KNN'] = KNN
    y_prediction['NN'], y_prediction_train['NN'] = NN
    table_data = []

    for method in none_mixed_methods:
        meanAbsoluteError, rootMeanSquaredError, percentageOfAbsoluteError, adj_r_squared, second_error = get_errors(best_h[method]['MAPE'],
             best_c[method]['MAPE'], method, y_prediction[method], historical_y_test_date[method], mode='test')
        table_data.append([best_h[method]['MAPE'], best_c[method]['MAPE'], round(rootMeanSquaredError, 2), round(meanAbsoluteError, 2),
             round(percentageOfAbsoluteError, 2), round(adj_r_squared, 2), round(second_error, 2)])
    # table_name = 'non-mixed methods best results'
    # plot_table(table_data, columns_table, none_mixed_methods, table_name)
    #push('a new table added')
    # generate data for non-mixed methods with the best h and c of mixed models and fit mixed models on them
    # (with the whole training set)
    y_predictions = {'MM_LR': [], 'MM_NN': []}
    y_prediction = {}
    #table_data = []
    X_train_MM_dict, X_test_MM_dict, y_train_MM_dict, y_test_MM_dict = {}, {}, {}, {}
    y_train, y_test = {}, {}
    y_test_date = {}
    for mixed_method in mixed_methods:
        X_train, X_test, y_train_date, y_test_date[mixed_method] = generate_data(best_h[mixed_method]['MAPE'], best_c[mixed_method]['MAPE'],
                                                                   covariates_names)
        y_test_date_temp = y_test_date[mixed_method]
        y_train[mixed_method] = np.array(y_train_date['Target']).reshape(-1)
        y_test[mixed_method] = np.array(y_test_date_temp['Target']).reshape(-1)
        for method in none_mixed_methods:
            X_train_dict[method] = X_train
            X_test_dict[method] = X_test
            y_train_dict[method] = y_train[mixed_method]
            y_test_dict[method] = y_test[mixed_method]
        # if y_test_dict['GBM'] == y_test_dict['GLM']: print('YES!')
        # print(y_test_dict['GBM'])
        # print(y_test_dict['GLM'])
        GBM, GLM, KNN, NN = run_algorithms(X_train_dict, X_test_dict, y_train_dict, y_test_dict)
        y_prediction['GBM'], y_prediction_train['GBM'] = GBM
        y_prediction['GLM'], y_prediction_train['GLM'] = GLM
        y_prediction['KNN'], y_prediction_train['KNN'] = KNN
        y_prediction['NN'], y_prediction_train['NN'] = NN
        y_predictions_test, y_predictions_train = [], []
        # Construct the outputs for the testing dataset of the 'MM' methods
        y_predictions_test.extend([y_prediction['GBM'], y_prediction['GLM'], y_prediction['KNN'], y_prediction['NN']])
        y_prediction_test_np = np.array(y_predictions_test).reshape(len(y_predictions_test), -1)
        X_test_mixedModel = pd.DataFrame(y_prediction_test_np.transpose())
        # Construct the outputs for the training dataset of the 'MM' methods
        y_predictions_train.extend([y_prediction_train['GBM'], y_prediction_train['GLM'], y_prediction_train['KNN'], y_prediction_train['NN']])
        y_prediction_train_np = np.array(y_predictions_train).reshape(len(y_predictions_train), -1)
        X_train_mixedModel = pd.DataFrame(y_prediction_train_np.transpose())
        X_train_MM_dict[mixed_method] = X_train_mixedModel
        X_test_MM_dict[mixed_method] = X_test_mixedModel
        y_train_MM_dict[mixed_method] = y_train[mixed_method]
        y_test_MM_dict[mixed_method] = y_test[mixed_method]
    # save the entire session
    filename = env_address + 'test.out'
    my_shelf = shelve.open(filename, 'n')  # 'n' for new
    for key in dir():
        try:
            my_shelf[key] = locals()[key]
        except:
            print('ERROR shelving: {0}'.format(key))
    my_shelf.close()
    # mixed model with linear regression and neural network
    MM_LR, MM_NN = run_mixed_models(X_train_MM_dict, X_test_MM_dict, y_train_MM_dict, y_test_MM_dict)
    y_prediction['MM_LR'], dummy = MM_LR
    y_prediction['MM_NN'], dummy = MM_NN
    for mixed_method in mixed_methods:
        meanAbsoluteError, rootMeanSquaredError, percentageOfAbsoluteError, adj_r_squared, second_error = get_errors(best_h[mixed_method]['MAPE'],
        best_c[mixed_method]['MAPE'], mixed_method, y_prediction[mixed_method], y_test_date[mixed_method], mode='test')
        table_data.append([best_h[mixed_method]['MAPE'], best_c[mixed_method]['MAPE'], round(rootMeanSquaredError, 2),
             round(meanAbsoluteError, 2), round(percentageOfAbsoluteError, 2), round(adj_r_squared, 2), round(second_error, 2)])

    # save the entire session
    filename = env_address + 'test.out'
    my_shelf = shelve.open(filename, 'n')  # 'n' for new
    for key in dir():
        try:
            my_shelf[key] = locals()[key]
        except:
            print('ERROR shelving: {0}'.format(key))
    my_shelf.close()
    table_name = 'table_of_best_test_results'
    plot_table(table_data, columns_table, methods, table_name, mode='test')
    #push('a new table added')


if __name__ == "__main__":

    begin = time.time()
    maxHistory = 2
    # make directories for saving the results
    validation_address = str(argv[1]) + 'results/counties=' + str(numberOfSelectedCounties) + ' max_history=' + str(maxHistory) + '/validation/'
    test_address = str(argv[1]) + 'results/counties=' + str(numberOfSelectedCounties) + ' max_history=' + str(maxHistory) + '/test/'
    env_address = str(argv[1]) + 'results/counties=' + str(numberOfSelectedCounties) + ' max_history=' + str(maxHistory) + '/session_parameters/'

    if not os.path.exists(test_address):
        os.makedirs(test_address)
    if not os.path.exists(validation_address):
        os.makedirs(validation_address)
    if not os.path.exists(env_address):
        os.makedirs(env_address)
    #push('new folders added')
    main(maxHistory)
    end = time.time()
    #push('final results added')
    print("The total time of execution in minutes: ", round((end - begin) / 60, 2))
