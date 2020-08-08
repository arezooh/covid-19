from makeHistoricalData import makeHistoricalData
from models import GBM, GLM, KNN, NN, MM_GLM, GBM_grid_search, NN_grid_search
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
#import dill
import glob
import shutil
import zipfile
import email, smtplib, ssl
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import mimetypes
import subprocess as cmd
import shelve
import matplotlib.pyplot as plt
import random
import datetime
import statistics

plt.rcParams.update({'figure.max_open_warning': 0})

r = 21  # the following day to predict
numberOfSelectedCounties = -1
target_mode = 'weeklymovingaverage'
spatial_mode = 'country'
numberOfSelectedCountiesname = 1535

######################################################### split data to train, val, test
def splitData(numberOfCounties, main_data, target, mode ):

    if numberOfCounties == -1:
      numberOfCounties = len(main_data['county_fips'].unique())

    if mode == 'val':
      main_data = main_data.sort_values(by=['date of day t' , 'county_fips'])
      target = target.sort_values(by=['date of day t' , 'county_fips'])
      X_train_train = main_data.iloc[:-2*(r*numberOfCounties),:].sort_values(by=['county_fips' , 'date of day t'])
      # X_train_train = X_train_train.drop(['date of day t', 'county_fips'], axis=1)
      X_train_val = main_data.iloc[-2*(r*numberOfCounties):-(r*numberOfCounties),:].sort_values(by=['county_fips' , 'date of day t'])
      # X_train_val = X_train_val.drop(['date of day t', 'county_fips'], axis=1)
      X_test = main_data.tail(r*numberOfCounties).sort_values(by=['county_fips' , 'date of day t'])
      # X_test = X_test.drop(['date of day t', 'county_fips'], axis=1)

      y_train_train = target.iloc[:-2*(r*numberOfCounties),:].sort_values(by=['county_fips' , 'date of day t'])
      y_train_val = target.iloc[-2*(r*numberOfCounties):-(r*numberOfCounties),:].sort_values(by=['county_fips' , 'date of day t'])
      y_test = target.tail(r*numberOfCounties).sort_values(by=['county_fips' , 'date of day t'])

      return X_train_train , X_train_val , X_test , y_train_train , y_train_val , y_test

    if mode == 'test':
      main_data = main_data.sort_values(by=['date of day t' , 'county_fips'])
      target = target.sort_values(by=['date of day t' , 'county_fips'])

      X_train = main_data.iloc[:-(r*numberOfCounties),:].sort_values(by=['county_fips' , 'date of day t'])
      # X_train = X_train.drop(['date of day t', 'county_fips'], axis=1)
      X_test = main_data.tail(r*numberOfCounties).sort_values(by=['county_fips' , 'date of day t'])
      # X_test = X_test.drop(['date of day t', 'county_fips'], axis=1)

      y_train = target.iloc[:-(r*numberOfCounties),:].sort_values(by=['county_fips' , 'date of day t'])
      y_test = target.tail(r*numberOfCounties).sort_values(by=['county_fips' , 'date of day t'])

      return X_train , X_test , y_train , y_test



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
    main_data = using_data.drop(['county_name', 'state_fips', 'state_name'],
                                axis=1)  # , 'date of day t'
    # target = pd.DataFrame(main_data['Target'])
    # main_data = main_data.drop(['Target'], axis=1)
    # numberOfCounties = len(using_data['county_fips'].unique())
    numberOfDays = len(using_data['date of day t'].unique())

    return main_data


########################################################### preprocess
def preprocess(main_data, validationFlag):

    target = pd.DataFrame(main_data[['date of day t', 'county_fips', 'Target']])
    main_data = main_data.drop(['Target'], axis=1)
    # specify the size of train, validation and test sets
    t1 = time.time()
    # produce train, validation and test data in parallel

    if validationFlag:     # validationFlag is 1 if we want to have a validation set and 0 otherwise
        # add the functions to the multiprocessing object, loom

        X_train_train , X_train_val , X_test , y_train_train , y_train_val , y_test = splitData(numberOfSelectedCounties, main_data, target,'val')
        return X_train_train, X_train_val, X_test, y_train_train, y_train_val, y_test

    else:

        X_train , X_test , y_train , y_test = splitData(numberOfSelectedCounties, main_data, target,'test')
        return X_train, X_test, y_train, y_test




################################ MASE_denominator
def mase_denominator(r, h, target_name, target_mode ,numberOfSelectedCounties, spatial_mode):

    data = makeHistoricalData(h, r, target_name, 'mrmr', spatial_mode, target_mode, './')
    if numberOfSelectedCounties == -1 :
      numberOfSelectedCounties = len(data['county_fips'].unique())
    data = clean_data(data, numberOfSelectedCounties)
    X_train_train, X_train_val, X_test, y_train_train_date, y_train_val_date, y_test_date = preprocess(data, 1)

    train_train = (y_train_train_date.reset_index(drop=True)).sort_values(by=['date of day t', 'county_fips'])
    train_val = (y_train_val_date.reset_index(drop=True)).sort_values(by=['date of day t', 'county_fips'])
    test = (y_test_date.reset_index(drop=True)).sort_values(by=['date of day t', 'county_fips'])

    train_lag = train_train.copy().iloc[:-(numberOfSelectedCounties*r), :].tail(numberOfSelectedCounties*r).rename(
        columns={'Target': 'train-lag-Target'})
    train_train = train_train.tail(numberOfSelectedCounties*r).rename(columns={'Target': 'train-Target'}) #[['train-Target']]
    train_val = train_val.tail(numberOfSelectedCounties*r).rename(columns={'Target': 'val-Target'}) #[['val-Target']]
    test = test.tail(numberOfSelectedCounties*r).rename(columns={'Target': 'test-Target'}) #[['test-Target']]

    df_for_train_lag_MASE_denominator=pd.concat([train_train.reset_index(drop=True), train_lag.reset_index(drop=True)], axis=1)
    #[['county_fips','date of day t','Target']]
    df_for_train_lag_MASE_denominator['absolute-error']=abs(df_for_train_lag_MASE_denominator['train-Target'] -
                                                            df_for_train_lag_MASE_denominator['train-lag-Target'])

    df_for_train_val_MASE_denominator = pd.concat([train_train.reset_index(drop=True),train_val.reset_index(drop=True)], axis=1)
    #[['county_fips','date of day t','Target']]
    df_for_train_val_MASE_denominator['absolute-error']=abs(df_for_train_val_MASE_denominator['val-Target'] -
                                                            df_for_train_val_MASE_denominator['train-Target'])

    df_for_val_test_MASE_denominator=pd.concat([train_val.reset_index(drop=True),test.reset_index(drop=True)], axis=1)
    #[['county_fips','date of day t','Target']]
    df_for_val_test_MASE_denominator['absolute-error']=abs(df_for_val_test_MASE_denominator['test-Target'] -
                                                           df_for_val_test_MASE_denominator['val-Target'])

    train_val_MASE_denominator = df_for_train_val_MASE_denominator['absolute-error'].mean()
    val_test_MASE_denominator = df_for_val_test_MASE_denominator['absolute-error'].mean()
    train_lag_MASE_denominator = df_for_train_lag_MASE_denominator['absolute-error'].mean()

    return train_val_MASE_denominator, val_test_MASE_denominator, train_lag_MASE_denominator


########################################################### run non-mixed methods in parallel
def parallel_run(method, X_train_train, X_train_val, y_train_train, y_train_val, best_loss, c):

    y_prediction, y_prediction_train = None, None
    if method == 'GBM':
        y_prediction, y_prediction_train = GBM(X_train_train, X_train_val, y_train_train, best_loss['GBM'])
    elif method == 'GLM':
        y_prediction, y_prediction_train = GLM(X_train_train, X_train_val, y_train_train)
    elif method == 'KNN':
        y_prediction, y_prediction_train = KNN(X_train_train, X_train_val, y_train_train)
    elif method == 'NN':
        y_prediction, y_prediction_train = NN(X_train_train, X_train_val, y_train_train, y_train_val, best_loss['NN'])

    return y_prediction, y_prediction_train


########################################################### run mixed methods in parallel
def mixed_parallel_run(method, X_train, X_test, y_train, y_test, best_loss):

    y_prediction, y_prediction_train = None, None
    if method == 'MM_GLM':
        y_prediction, y_prediction_train = MM_GLM(X_train, X_test, y_train)
    elif method == 'MM_NN':
        y_prediction, y_prediction_train = NN(X_train, X_test, y_train, y_test, best_loss[method])

    return y_prediction, y_prediction_train


########################################################### run algorithms in parallel except mixed models
def run_algorithms(X_train_dict, X_val_dict, y_train_dict, y_val_dict, best_loss, c , spatial_mode, county_fips):
    from models import GBM, GLM, KNN, NN
    t1 = time.time()
    methods = ['GBM','GLM','KNN','NN']
    X_train = {method : None for method in methods}
    X_val = {method : None for method in methods}
    y_train = {method : None for method in methods}
    y_val = {method : None for method in methods}
    loom = ProcessLoom(max_runner_cap=4)
    # add the functions to the multiprocessing object, loom
    if spatial_mode == 'country':
      for method in methods:
        X_train[method] = X_train_dict[method].drop(['county_fips','date of day t'],axis=1)
        X_val[method] = X_val_dict[method].drop(['county_fips','date of day t'],axis=1)
        y_train[method] = np.array(y_train_dict[method]['Target']).reshape(-1)
        y_val[method] = np.array(y_val_dict[method]['Target']).reshape(-1)
      loom.add_function(GBM, [X_train['GBM'], X_val['GBM'], y_train['GBM'], best_loss['GBM']], {})
      loom.add_function(GLM, [X_train['GLM'], X_val['GLM'], y_train['GLM']], {})
      loom.add_function(KNN, [X_train['KNN'], X_val['KNN'], y_train['KNN']], {})
      loom.add_function(NN, [X_train['NN'], X_val['NN'], y_train['NN'], y_val['NN'], best_loss['NN']], {})
    if spatial_mode == 'county':
      for method in methods:
        X_train[method] = X_train_dict[method]
        X_train[method] = X_train[method][X_train[method]['county_fips']==county_fips].drop(['county_fips','date of day t'],axis=1)
        X_val[method] = X_val_dict[method]
        X_val[method] = X_val[method][X_val[method]['county_fips']==county_fips].drop(['county_fips','date of day t'],axis=1)
        y_train[method] = y_train_dict[method]
        y_train[method] = y_train[method][y_train[method]['county_fips']==county_fips].drop(['county_fips','date of day t'],axis=1)
        y_val[method] = y_val_dict[method]
        y_val[method] = y_val[method][y_val[method]['county_fips']==county_fips].drop(['county_fips','date of day t'],axis=1)
        y_train[method] = np.array(y_train[method]['Target']).reshape(-1)
        y_val[method] = np.array(y_val[method]['Target']).reshape(-1)
      loom.add_function(GBM, [X_train['GBM'], X_val['GBM'], y_train['GBM'], best_loss['GBM']], {})
      loom.add_function(GLM, [X_train['GLM'], X_val['GLM'], y_train['GLM']], {})
      loom.add_function(KNN, [X_train['KNN'], X_val['KNN'], y_train['KNN']], {})
      loom.add_function(NN, [X_train['NN'], X_val['NN'], y_train['NN'], y_val['NN'], best_loss['NN']], {})

    # run the processes in parallel
    output = loom.execute()
    t2 = time.time()
    print('total time - run algorithms: ', t2 - t1)

    return output[0]['output'], output[1]['output'], output[2]['output'], output[3]['output']


########################################################### run mixed models in parallel
def run_mixed_models(X_train_MM, X_test_MM, y_train_MM, y_test_MM, best_loss):

    from models import GBM, GLM, KNN, NN, MM_GLM
    t1 = time.time()
    loom = ProcessLoom(max_runner_cap=2)
    # add the functions to the multiprocessing object, loom
    loom.add_function(MM_GLM, [X_train_MM['MM_GLM'], X_test_MM['MM_GLM'], y_train_MM['MM_GLM']], {})
    loom.add_function(NN, [X_train_MM['MM_NN'], X_test_MM['MM_NN'], y_train_MM['MM_NN'], y_test_MM['MM_NN'], best_loss['MM_NN']], {})
    # run the processes in parallel
    output = loom.execute()
    t2 = time.time()
    print('total time - run mixed models: ', t2 - t1)

    return output[0]['output'], output[1]['output']

####################################################################### update best loss

def update_best_loss(model_type ,spatial_mode ,county_fips,best_loss,X_train_train_to_use,X_train_val_to_use,y_train_train,\
                     y_train_val,y_prediction_train,y_prediction,covariates,\
                     numberOfCovariates,max_c):
    h = 1
    if model_type == 'mixed_model':
          loom = ProcessLoom(max_runner_cap=1)
          c = numberOfCovariates
          if numberOfCovariates > max_c :
            c = max_c
          y_predictions_test, y_predictions_train = [], []
          if spatial_mode == 'county':
            # Construct the outputs for the testing dataset of the 'MM' methods
            y_predictions_test.extend([y_prediction[county_fips]['GBM'][(h, c)], y_prediction[county_fips]['GLM'][(h, c)],
                                        y_prediction[county_fips]['KNN'][(h, c)], y_prediction[county_fips]['NN'][(h, c)]])
            
          elif spatial_mode == 'country':
            y_predictions_test.extend([y_prediction['GBM'][(h, c)], y_prediction['GLM'][(h, c)],
                                        y_prediction['KNN'][(h, c)], y_prediction['NN'][(h, c)]])
          y_prediction_test_np = np.array(y_predictions_test).reshape(len(y_predictions_test), -1)
          X_test_mixedModel = pd.DataFrame(y_prediction_test_np.transpose())
          if spatial_mode == 'county':
            # Construct the outputs for the training dataset of the 'MM' methods
            y_predictions_train.extend([y_prediction_train[county_fips]['GBM'][(h, c)], y_prediction_train[county_fips]['GLM'][(h, c)],
                                        y_prediction_train[county_fips]['KNN'][(h, c)], y_prediction_train[county_fips]['NN'][(h, c)]])
          elif spatial_mode == 'country':
            y_predictions_train.extend([y_prediction_train['GBM'][(h, c)], y_prediction_train['GLM'][(h, c)],
                                        y_prediction_train['KNN'][(h, c)], y_prediction_train['NN'][(h, c)]])
          y_prediction_train_np = np.array(y_predictions_train).reshape(len(y_predictions_train), -1)
          X_train_mixedModel = pd.DataFrame(y_prediction_train_np.transpose())
          loom.add_function(NN_grid_search, [X_train_mixedModel,y_train_train , X_test_mixedModel,y_train_val],{})
          best_loss_output = loom.execute()
          best_loss['MM_NN'] = best_loss_output[0]['output']
          
    if model_type == 'none_mixed_model':   
          print('check 292')
        
          loom = ProcessLoom(max_runner_cap= 2)
          if spatial_mode == 'country':
            loom.add_function(GBM_grid_search, [X_train_train_to_use['GBM'][covariates],
                                                    y_train_train , X_train_val_to_use['GBM'][covariates],
                                                    y_train_val],{})
            
            loom.add_function(NN_grid_search, [X_train_train_to_use['NN'][covariates],
                                                    y_train_train , X_train_val_to_use['NN'][covariates],
                                                    y_train_val],{})
          if spatial_mode == 'county':
            loom.add_function(GBM_grid_search, [X_train_train_to_use[county_fips][h]['GBM'][covariates],
                                                    y_train_train , X_train_val_to_use[county_fips][h]['GBM'][covariates],
                                                    y_train_val],{})
            loom.add_function(NN_grid_search, [X_train_train_to_use[county_fips][h]['NN'][covariates],
                                                    y_train_train , X_train_val_to_use[county_fips][h]['NN'][covariates],
                                                    y_train_val],{})
          
          best_loss_output=loom.execute()
          
          
          best_loss['GBM'],best_loss['NN'] = best_loss_output[0]['output'],best_loss_output[1]['output']
          
    return best_loss

########################################################### 

def get_best_loss_mode(counties_best_loss_list):

  methods_with_loss=['GBM', 'NN', 'MM_NN']
  best_loss = {method: None for method in methods_with_loss}
  for method in methods_with_loss:

    counties_best_loss_array=np.array(counties_best_loss_list[method])
    # when we choose number_of_selected_counties smaller than number of different losses
    # some times its not possibel to find mode
    if len(np.unique(counties_best_loss_array))==len(counties_best_loss_array):
      best_loss[method] = random.choice(counties_best_loss_list[method])
    else:
      best_loss[method] = statistics.mode(counties_best_loss_list[method])
  return(best_loss)

########################################################### generate data for best h and c

def generate_data(h, numberOfCovariates, covariates_names, numberOfSelectedCounties):

    data = makeHistoricalData(h, r, 'death', 'mrmr', spatial_mode, target_mode, './')
    data = clean_data(data, numberOfSelectedCounties)

    X_train, X_test, y_train, y_test = preprocess(data, 0)
    covariates = [covariates_names[i] for i in range(numberOfCovariates)]
    best_covariates = []
    indx_c = 0
    for covar in covariates:  # iterate through sorted covariates
        indx_c += 1
        for covariate in data.columns:  # add all historical covariates of this covariate and create a feature
            if covar.split(' ')[0] in covariate:
                best_covariates.append(covariate)

    best_covariates += ['county_fips','date of day t'] # we add this two columns to use when we want break data to county_data
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
                covariates_list = [c for c in range(1, numberOfCovariates + 1)][:maxC]
                # y label: errors
                for c in range(1, numberOfCovariates + 1):
                    errors_h.append(errors[methods[ind]][(h, c)])
                    if c == maxC:
                      break
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
    plt.savefig(address + str(mode)+'.pdf')


########################################################### plot table for final results
def plot_table(table_data, col_labels, row_labels, name, mode):
    fig = plt.figure() #dpi=50 figsize=(30, 10)
    ax = fig.add_subplot(111)
    colWidths = [0.1, 0.1, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]
    address = ''
    if mode == 'val':
        # colWidths.pop()
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
    the_table.set_fontsize(9)
    the_table.scale(1.5, 1.5)
    ax.axis('off')

    plt.savefig(address + name + '.pdf', bbox_inches='tight')


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
    plt.savefig(address +'procedure_'+ str(method) +'.pdf')


########################################################### box plots and violin plots
def box_violin_plot(X, Y, figsizes, fontsizes, name, address):
    mpl.style.use('default')
    # box plot
    fig = plt.figure(figsize=figsizes['box'])
    plt.rc('font', size=fontsizes['box'])
    plt.locator_params(axis='y', nbins=20)
    sns.boxplot(x=X, y=Y)
    plt.savefig(address + str(name) + 'boxplot.pdf')
    plt.close()
    # violin plot
    fig = plt.figure(figsize=figsizes['violin'])
    plt.rc('font', size=fontsizes['violin'])
    plt.locator_params(axis='y', nbins=20)
    sns.violinplot(x=X, y=Y)
    plt.savefig(address + str(name) + 'violinplot.pdf')
    plt.close()
########################################################### plot prediction and real values


def real_prediction_plot(df,r,target_name,target_mode,best_h,maxHistory,spatial_mode,methods,numberOfSelectedCounties):

    address = test_address + 'plots_of_real_prediction_values/'
    if not os.path.exists(address):
        os.makedirs(address)

    for method in methods:
        
        data=makeHistoricalData(best_h[method]['MAPE'], r, target_name, 'mrmr', spatial_mode, target_mode, './')
        if numberOfSelectedCounties == -1 :
            method_numberOfSelectedCounties = len(data['county_fips'].unique())
        else:
            method_numberOfSelectedCounties = numberOfSelectedCounties
        
        data = data.sort_values(by=['county_fips', 'date of day t'])
        data = data[(data['county_fips'] <= data['county_fips'].unique()[method_numberOfSelectedCounties - 1])]
        
        data = data.reset_index(drop=True)
        data = data[['state_fips','county_name','county_fips','date of day t','Target']]
        data=data.sort_values(by=['date of day t','county_fips'])
        data_train_train=data.iloc[:-2*(r*method_numberOfSelectedCounties),:]
        data_train_val=data.iloc[-2*(r*method_numberOfSelectedCounties):-(r*method_numberOfSelectedCounties),:]
        
        data_test=data.tail(r*method_numberOfSelectedCounties)
        
        if spatial_mode == 'country' :
            data_train_train=data_train_train.sort_values(by=['county_fips','date of day t'])
            data_train_val=data_train_val.sort_values(by=['county_fips','date of day t'])
            data_test=data_test.sort_values(by=['county_fips','date of day t'])
            data=data_train_train.append(data_train_val)
            data=data.append(data_test)
        if spatial_mode == 'county' :
            data_train = data_train_train.append(data_train_val)
            data_train = data_train.sort_values(by=['county_fips','date of day t'])
            data_test = data_test.sort_values(by=['county_fips','date of day t'])
            data = data_train.append(data_test)
        if spatial_mode == 'state' :
            data_train = data_train_train.append(data_train_val)
            data_train = data_train.sort_values(by=['state_fips','date of day t'])
            data_test = data_test.sort_values(by=['state_fips','date of day t'])
            data = data_train.append(data_test)
        method_prediction_df = pd.DataFrame(df[method],columns=[method])
        df_for_plot = pd.concat([data.reset_index(drop=True),method_prediction_df.reset_index(drop=True)],axis=1)
        
       
        if target_mode != 'weeklyaverage':
          df_for_plot['date'] = df_for_plot['date of day t'].apply(lambda x:datetime.datetime.strptime(x,'%m/%d/%y')+datetime.timedelta(days=r))
          df_for_plot['date'] = df_for_plot['date'].apply(lambda x:datetime.datetime.strftime(x,'%m/%d/%y'))
        else:
          df_for_plot['date'] = df_for_plot['date of day t'].apply(lambda x:'week '+str(x+r))
        
        df_for_plot.loc[df_for_plot[method]<0,method] = 0
        
        counties = []
        for i in [36061,40117,51059]: # newyork + two random county
          if len(df_for_plot[df_for_plot['county_fips']==i]) > 0 :
              counties.append(i)
          else :
              counties = counties + random.sample(df_for_plot['county_fips'].unique().tolist(),1)

        length=list()
        for county in counties:
          length.append(len(df_for_plot[df_for_plot['county_fips']==county]))

        plot_with=max(length)+20

        fig, ax = plt.subplots(figsize=(plot_with,75))
        mpl.style.use('default')
        plt.rc('font', size=45)
        
        for index,county in enumerate(counties):

            plt.subplot(311+index)
            plt.plot(df_for_plot.loc[df_for_plot['county_fips']==county,'date'][:-(r-1)],df_for_plot.loc[df_for_plot['county_fips']==county,method].round()[:-(r-1)],label='Train prediction',linewidth=2.0)#,color='forestgreen'
            plt.plot(df_for_plot.loc[df_for_plot['county_fips']==county,'date'][-r:],df_for_plot.loc[df_for_plot['county_fips']==county,method].round()[-r:],label='Test prediction',linewidth=2.0)#,color='dodgerblue'
            plt.plot(df_for_plot.loc[df_for_plot['county_fips']==county,'date'],df_for_plot.loc[df_for_plot['county_fips']==county,'Target'],label='Real values',color='black',linewidth=2.0)
            if target_mode != 'cumulative' :
                plt.plot(df_for_plot.loc[df_for_plot['county_fips']==county,'date'][-r:],df_for_plot.loc[df_for_plot['county_fips']==county,'Target'][-(2*r):-r],'-.',color='gray',label='Naive prediction',linewidth=2.0)
            plt.xticks(rotation=65)
            fig.subplots_adjust(hspace=0.4)
            plt.ylabel('Number of deaths')
            countyname = df_for_plot.loc[df_for_plot['county_fips']==county,'county_name'].unique()
            if len(countyname)>0 : # it is False when newyork is not in selected counties and make error
              plt.title(df_for_plot.loc[df_for_plot['county_fips']==county,'county_name'].unique()[0])
            plt.legend()
        plt.xlabel('Date')
        plt.savefig(address + str(method) + ' real_prediction_values.pdf')
        plt.close()



########################################################### get errors for each model in each h and c
def get_errors(h, c, method, y_prediction, y_prediction_train, y_test_date, y_train_date, MASE_denominator, numberOfSelectedCounties, target_name, mode):
    # set negative predictions to zero
    
    y_prediction[y_prediction < 0] = 0
    
    
    # y_test_date is a dataframe with columns ['date of day t', 'county_fips', 'Target']
    y_test = np.array(y_test_date['Target']).reshape(-1)
    
    if numberOfSelectedCounties == -1 :
          numberOfSelectedCounties = len(y_test_date['county_fips'].unique())
            
    # if target mode is cumulative we need to return the target variable to its original state
    if target_mode == 'cumulative':
        cumulative_data = y_train_date.append(y_test_date)
        cumulative_data['prediction'] = list(y_train_date['Target'])+list(y_prediction)
        cumulative_data = cumulative_data.sort_values(by=['date of day t','county_fips'])
        reverse_dates = cumulative_data['date of day t'].unique()[-(r+1):][::-1]
        for index in range(len(reverse_dates)):
            date=reverse_dates[index]
            print(date)
            past_date=reverse_dates[index+1]
            print(past_date)
            cumulative_data.loc[cumulative_data['date of day t']==date,'Target']=list(np.array(cumulative_data.loc[cumulative_data['date of day t']==date,'Target'])-np.array(cumulative_data.loc[cumulative_data['date of day t']==past_date,'Target']))
            cumulative_data.loc[cumulative_data['date of day t']==date,'prediction']=list(np.array(cumulative_data.loc[cumulative_data['date of day t']==date,'prediction'])-np.array(cumulative_data.loc[cumulative_data['date of day t']==past_date,'prediction']))
            if index == len(reverse_dates)-2:
                break
        cumulative_data = cumulative_data.sort_values(by=['date of day t','county_fips'])
        y_test = np.array(cumulative_data.tail(r*numberOfSelectedCounties)['Target']).reshape(-1)
        y_prediction = np.array(cumulative_data.tail(r*numberOfSelectedCounties)['prediction']).reshape(-1)

        
    # if target mode is logarithmic we need to return the target variable to its original state
    if target_mode == 'logarithmic':
        y_test = np.array(np.exp(y_test)-1).reshape(-1)
        y_prediction = np.array(np.exp(y_prediction)-1).reshape(-1)
        

    # if target mode is moving average we need to return the target variable to its original state
    if target_mode == 'weeklymovingaverage':
        regular_data = makeHistoricalData(h, r, target_name, 'mrmr', spatial_mode, 'regular', './')
        regular_data = clean_data(regular_data, numberOfSelectedCounties)
        if mode == 'val':
            regular_X_train_train, regular_X_train_val, regular_X_test, regular_y_train_train_date, regular_y_train_val_date, regular_y_test_date = preprocess(regular_data, 1)
            # past values of targets that will be use for return the weekly moving averaged target (predicted)
            #  to original state to calculate errors
            regular_real_predicted_target = regular_y_train_train_date.append(regular_y_train_val_date) #dataframe with columns ['date of day t', 'county_fips', 'Target']
            regular_real_predicted_target['prediction'] = list(regular_y_train_train_date['Target'])+list(y_prediction)
            
        if mode == 'test':
            regular_X_train, regular_X_test, regular_y_train_date, regular_y_test_date = preprocess(regular_data, 0)
            regular_real_predicted_target = regular_y_train_date.append(regular_y_test_date) #dataframe with columns ['date of day t', 'county_fips', 'Target']
            regular_real_predicted_target['prediction'] = list(regular_y_train_date['Target'])+list(y_prediction)
            
        regular_real_predicted_target = regular_real_predicted_target.sort_values(by=['date of day t','county_fips'])
        regular_real_predicted_target=regular_real_predicted_target.tail((r+6)*numberOfSelectedCounties)
#         print('regular_real_predicted_target')
#         print(regular_real_predicted_target)
        dates = regular_real_predicted_target['date of day t'].unique()
        for index in range(len(dates)):
            ind=index+6
            date=dates[ind]
            regular_real_predicted_target.loc[regular_real_predicted_target['date of day t']==date,'prediction'] = list(7*np.array(regular_real_predicted_target.loc[regular_real_predicted_target['date of day t']==date,'prediction']))
            for i in range(6):
                past_date=dates[ind-(i+1)]
                regular_real_predicted_target.loc[regular_real_predicted_target['date of day t']==date,'prediction']=list(np.array(regular_real_predicted_target.loc[regular_real_predicted_target['date of day t']==date,'prediction'])-np.array(regular_real_predicted_target.loc[regular_real_predicted_target['date of day t']==past_date,'prediction']))
            if ind == len(dates)-1:
                break
#         print('regular_real_predicted_target')
#         print(regular_real_predicted_target)
        y_test = np.array(regular_real_predicted_target.tail(r*numberOfSelectedCounties)['Target']).reshape(-1)
        y_prediction = np.array(regular_real_predicted_target.tail(r*numberOfSelectedCounties)['prediction']).reshape(-1)
        
    # if target mode is differential we need to return the target variable to its original state
    if target_mode == 'differential':
        regular_data = makeHistoricalData(h, r, target_name, 'mrmr', spatial_mode, 'regular', './')
        regular_data = clean_data(regular_data, numberOfSelectedCounties)
        if mode == 'val':
            regular_X_train_train, regular_X_train_val, regular_X_test, regular_y_train_train_date, regular_y_train_val_date, regular_y_test_date = preprocess(regular_data, 1)
            # past values of targets that will be use for return the differential target (predicted)
            # to original state to calculate errors
            regular_real_predicted_target = regular_y_train_train_date.append(regular_y_train_val_date) #dataframe with columns ['date of day t', 'county_fips', 'Target']
            regular_real_predicted_target['prediction'] = list(regular_y_train_train_date['Target'])+list(y_prediction)
            
        if mode == 'test':
            regular_X_train, regular_X_test, regular_y_train_date, regular_y_test_date = preprocess(regular_data, 0)
            regular_real_predicted_target = regular_y_train_date.append(regular_y_test_date) #dataframe with columns ['date of day t', 'county_fips', 'Target']
            regular_real_predicted_target['prediction'] = list(regular_y_train_date['Target'])+list(y_prediction)
            
        regular_real_predicted_target = regular_real_predicted_target.sort_values(by=['date of day t','county_fips'])
        regular_real_predicted_target=regular_real_predicted_target.tail((r+1)*numberOfSelectedCounties)
        print('regular_real_predicted_target before')
        print(regular_real_predicted_target)
        dates = regular_real_predicted_target['date of day t'].unique()
        for index in range(len(dates)):
            date=dates[index+1]
            past_date=dates[index]
#             cumulative_data.loc[cumulative_data['date']==date,'Target']=list(np.array(cumulative_data.loc[cumulative_data['date']==date,'Target'])-np.array(cumulative_data.loc[cumulative_data['date']==past_date,'Target']))
            regular_real_predicted_target.loc[regular_real_predicted_target['date of day t']==date,'prediction']=list(np.array(regular_real_predicted_target.loc[regular_real_predicted_target['date of day t']==date,'prediction'])+np.array(regular_real_predicted_target.loc[regular_real_predicted_target['date of day t']==past_date,'prediction']))
            if index == len(dates)-2:
                break
        print('regular_real_predicted_target after')
        print(regular_real_predicted_target)
        y_test = np.array(regular_real_predicted_target.tail(r*numberOfSelectedCounties)['Target']).reshape(-1)
        y_prediction = np.array(regular_real_predicted_target.tail(r*numberOfSelectedCounties)['prediction']).reshape(-1)
            
            
    # make predictions rounded to their closest number 
    y_prediction = np.round(y_prediction)
    #############################################################
    # write outputs into a file
    orig_stdout = sys.stdout
    f = open(env_address+'out.txt', 'a')
    sys.stdout = f
    meanAbsoluteError = mean_absolute_error(y_test, y_prediction)
    print("Mean Absolute Error of ", method, " for h =", h, "and #covariates =", c, ": %.2f" % meanAbsoluteError)
    sumOfAbsoluteError = sum(abs(y_test - y_prediction))
    percentageOfAbsoluteError = (sumOfAbsoluteError / sum(y_test)) * 100
    # we change zero targets into 1 and add 1 to their predictions
    y_test_temp = y_test.copy()
    y_test_temp[y_test == 0] = 1
    y_prediction_temp = y_prediction.copy()
    y_prediction_temp[y_test == 0] += 1
    # meanPercentageOfAbsoluteError = sum((abs(y_prediction_temp - y_test_temp) / y_test_temp) * 100) / len(y_test)
    print("Percentage of Absolute Error of ", method, " for h =", h, "and #covariates =", c,
          ": %.2f" % percentageOfAbsoluteError)
    rootMeanSquaredError = sqrt(mean_squared_error(y_test, y_prediction))
    print("Root Mean Squared Error of ", method, " for h =", h, "and #covariates =", c, ": %.2f" % rootMeanSquaredError)

    second_error = sum(abs(y_prediction - y_test))
    ### compute adjusted R squared error
    SS_Residual = sum((y_test - y_prediction.reshape(-1)) ** 2)
    SS_Total = sum((y_test - np.mean(y_test)) ** 2)
    r_squared = 1 - (float(SS_Residual)) / SS_Total
    adj_r_squared = 1 - (1 - r_squared) * (len(y_test) - 1) / (len(y_test) - c - 1)
    print("Adjusted R Squared Error of ", method, " for h =", h, "and #covariates =", c, ": %.2f" % adj_r_squared)
    
    MASE_numerator = sum(abs(y_prediction_temp - y_test_temp))/len(y_test)
    MASE = MASE_numerator/MASE_denominator
    print("MASE Error of ", method, " for h =", h, "and #covariates =", c, ": %.2f" % MASE)


    print("-----------------------------------------------------------------------------------------")

    # save outputs in 'out.txt'
    sys.stdout = orig_stdout
    f.close()
    # for the test mode we compute some additional errors, we need 'date of day t' column so we use the main dataframe
    # we add our prediction, the difference between prediction and target ('error' column),
    # the absolute difference between prediction and target ('absolute_error' column),
    # the precentage of this difference (('percentage_error' column) -> we change zero targets into 1 and add 1 to their predictions),
    # and second_error as follows and save these in 'all_errors' file
    # then we compute the average of percentage_errors (and other values) in each day and save them in
    # 'first_error' file

    if mode == 'test':
        # write outputs into a file
        orig_stdout = sys.stdout
        f = open(env_address + 'out.txt', 'a')
        sys.stdout = f

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
        y_test_temp = y_test.copy()
        y_test_temp[y_test == 0] = 1
        y_prediction_temp = y_prediction.copy()
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
        dataframe['county_fips']=dataframe['county_fips'].astype(float)
        if numberOfSelectedCounties == -1:
          numberOfSelectedCounties = len(dataframe['county_fips'])
        first_error = pd.DataFrame((dataframe.groupby(['date of day t']).sum() / numberOfSelectedCounties))
        first_error.columns = ['fips','average of targets', 'average of predictions', 'average of errors',
                               'average of absoulte_errors', 'average of percentage_errors']
        first_error = first_error.drop(['fips'], axis=1)
        first_error.to_csv(first_error_address + 'first_error_' + str(method) + '.csv')
        plot_targets(method, first_error.index, first_error, first_error_address)

        # save outputs in 'out.txt'
        sys.stdout = orig_stdout
        f.close()
    return meanAbsoluteError, percentageOfAbsoluteError, adj_r_squared, second_error, MASE


########################################################### push results to github
def push(message):
    try:
        cmd.run("git pull", check=True, shell=True)
        print("everything has been pulled")
        cmd.run("git add .", check=True, shell=True)
        cmd.run(f"git commit -m '{message}'", check=True, shell=True)
        cmd.run("git push", check=True, shell=True)
        print('pushed.')

    except:
        print('could not push')


########################################################### zip some of the results
def make_zip(selected_for_email, subject):

    for source_root in selected_for_email:
        for i in [x[0] for x in os.walk(source_root)]:
            address = mail_address  + '//'+ '/'.join(i.split('/')[3:])
            # print(address)
            if not os.path.exists(address):
                    os.makedirs(address)
            for pdffile in glob.iglob(os.path.join(i, "*.pdf")):
                shutil.copy(pdffile, address)
    shutil.make_archive(subject, 'zip', mail_address)


########################################################### mail some of the results
def send_email(*attachments):
    subject = "Server results"
    body = " "
    sender_email = "covidserver1@gmail.com"
    receiver_email = ["arezo.h1371@yahoo.com","arashmarioriyad@gmail.com"]#
    CC_email = ["p.ramazi@gmail.com"]#
    password = "S.123456.S"

    # Create a multipart message and set headers
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = ','.join(receiver_email)#receiver_email
    message["Subject"] = subject
    message["CC"] = ','.join(CC_email) # Recommended for mass emails

    # Add body to email
    message.attach(MIMEText(body, "plain"))

    # Add attachments
    for file_name in attachments:
            f = open(file_name, 'rb')
            ctype, encoding = mimetypes.guess_type(file_name)
            if ctype is None or encoding is not None:
                ctype = 'application/octet-stream'
            maintype, subtype = ctype.split('/', 1)
            # in case of a text file
            if maintype == 'text':
                part = MIMEText(f.read(), _subtype=subtype)
            # any other file
            else:
                part = MIMEBase(maintype, subtype)
                part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', 'attachment; filename="%s"' % os.path.basename(file_name))
            message.attach(part)
            f.close()
            text = message.as_string()

    # Log in to server using secure context and send email
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email+CC_email , text)



########################################################## flatten
def flatten(data=None, h=None, c=None, method=None, covariates_list=None, state=1):
    if state == 1:
        result = []
        for county_fips in data:
            result += list(data[county_fips][method][(h, c)])
    elif state == 2:
        result = []
        for county_fips in data:
            result += list(data[county_fips][(h, c)])
    elif state == 3:
        result = pd.DataFrame(columns=covariates_list)
        for county_fips in data:
            result = pd.concat([result, data[county_fips][h][method][covariates_list]], ignore_index=True)
    elif state == 4:
        for county_fips in data:
            result = pd.DataFrame(columns=data[county_fips].columns.values)
            break
        for county_fips in data:
            result = pd.concat([result, data[county_fips]], ignore_index=True)
    elif state == 5:
        result = []
        for county_fips in data:
            result += list(data[county_fips])
        result = np.array(result)
    elif state == 6:
        result = []
        for county_fips in data:
            result += list(data[county_fips][method])
    return result


########################################################### main
def main(maxHistory):

    history = [i for i in range(1, maxHistory + 1)]
    methods = ['GBM', 'GLM', 'KNN', 'NN', 'MM_GLM', 'MM_NN']
    none_mixed_methods = ['GBM', 'GLM', 'KNN', 'NN']
    mixed_methods = ['MM_GLM', 'MM_NN']
    target_name = 'death'
    base_data = makeHistoricalData(0, r, target_name, 'mrmr', spatial_mode, target_mode,'./')
    base_data = clean_data(base_data, numberOfSelectedCounties)
    covariates_names = list(base_data.columns)
    covariates_names.remove('Target')
    covariates_names.remove('date of day t')
    covariates_names.remove('county_fips')
    # covariates_names.remove('daily-country-test-per-1000 t')
    numberOfCovariates = len(covariates_names)
    print('number of covariates: ', numberOfCovariates)

    y_prediction = {'GBM': {}, 'GLM': {}, 'KNN': {}, 'NN': {}, 'MM_GLM': {}, 'MM_NN': {}}
    y_prediction_train = {'GBM': {}, 'GLM': {}, 'KNN': {}, 'NN': {}, 'MM_GLM': {}, 'MM_NN': {}}

    error_names = ['MAPE', 'MAE', 'adj-R2', 'sec', 'MASE']
    complete_error_names = {'MAPE': 'Percentage Of Absolute Error', 'MAE': 'Mean Absolute Error',
                            'MASE': 'Mean Absolute Scaled Error', 'adj-R2': 'Adjusted R Squared Error',
                            'sec': 'Sum Of Absolute Error'}
    validation_errors = {error: {method: {} for method in methods} for error in error_names}
    minError = {method: {error: int(1e10) for error in error_names} for method in methods}
    best_h = {method: {error: 0 for error in error_names} for method in methods}
    best_c = {method: {error: 0 for error in error_names} for method in methods}

    # best_loss = {method: None for method in ['GBM', 'NN', 'MM_NN']}
    best_loss = {'GBM': 'poisson', 'MM_NN': 'poisson', 'NN': 'MeanAbsoluteError'}


    columns_table_t = ['best_h', 'best_c', 'mean absolute error', 'percentage of absolute error', 'adjusted R squared error',
                      'second error', 'mean absolute scaled error']  # table columns names
    columns_table = ['best_h', 'best_c', 'mean absolute error', 'percentage of absolute error',
                      'adjusted R squared error',
                      'sum of absolute error', 'mean absolute scaled error']  # table columns names
    train_val_MASE_denominator = {h:None for h in history}
    val_test_MASE_denominator = {h:None for h in history}
    train_lag_MASE_denominator = {h:None for h in history}
    
    historical_X_train = {}  # X_train for best h and c
    historical_X_test = {}  # X_test for best h and c
    historical_y_train = {}  # y_train for best h and c
    historical_y_test = {}  # y_test for best h and c
    historical_y_train_date = {}  # y_train for best h and c with dates info
    historical_y_test_date = {}  # y_test for best h and c with dates info
    parallel_outputs = {}
    

    ############################################################
    # we define test as function to call it when h equal to half the maxHistory or when none of the models have improved in current h
    def test_process(h, r, target_name,spatial_mode, target_mode,best_h,best_c,historical_X_train,\
                    historical_X_test, historical_y_train_date, historical_y_test_date, best_loss,\
                    numberOfSelectedCounties, covariates_names, maxHistory, train_val_MASE_denominator,\
                 val_test_MASE_denominator, test_address, env_address , mail_address):
      
        columns_table_t = ['best_h', 'best_c', 'mean absolute error', 'percentage of absolute error', 'adjusted R squared error',
                          'second error', 'mean absolute scaled error']
        columns_table = ['best_h', 'best_c', 'mean absolute error', 'percentage of absolute error',
                          'adjusted R squared error',
                          'sum of absolute error', 'mean absolute scaled error']
        methods = ['GBM', 'GLM', 'KNN', 'NN', 'MM_GLM', 'MM_NN']
        none_mixed_methods = ['GBM', 'GLM', 'KNN', 'NN']
        mixed_methods = ['MM_GLM', 'MM_NN']

        
        df_for_prediction_plot = {method : None for method in methods}
        y_prediction = {}
        y_prediction_train = {}
        # run non-mixed methods on the whole training set with their best h and c
        X_train_dict, X_test_dict, y_train_dict, y_test_dict = {}, {}, {}, {}

        GBM, GLM, KNN, NN = run_algorithms(historical_X_train, historical_X_test, historical_y_train_date, historical_y_test_date, best_loss, 0, spatial_mode, None)

        y_prediction['GBM'], y_prediction_train['GBM'] = GBM
        y_prediction['GLM'], y_prediction_train['GLM'] = GLM
        y_prediction['KNN'], y_prediction_train['KNN'] = KNN
        y_prediction['NN'], y_prediction_train['NN'] = NN
        table_data = []

        for method in none_mixed_methods:
            meanAbsoluteError, percentageOfAbsoluteError, adj_r_squared, second_error, meanAbsoluteScaledError = get_errors(best_h[method]['MAPE'],
            best_c[method]['MAPE'], method, y_prediction[method], y_prediction_train[method], historical_y_test_date[method], historical_y_train_date[method],
                                val_test_MASE_denominator[best_h[method]['MAPE']], numberOfSelectedCounties, target_name, mode='test')
                                                                                                                
            table_data.append([best_h[method]['MAPE'], best_c[method]['MAPE'],  round(meanAbsoluteError, 2),
                                round(percentageOfAbsoluteError, 2), round(adj_r_squared, 2), round(second_error, 2), round(meanAbsoluteScaledError, 2)])

        push('a new table added')

        for method in none_mixed_methods:
          prediction=list(y_prediction_train[method])+list(y_prediction[method])
          df_for_prediction_plot[method]=prediction

        # generate data for non-mixed methods with the best h and c of mixed models and fit mixed models on them
        # (with the whole training set)
        y_predictions = {'MM_GLM': [], 'MM_NN': []}
        y_prediction = {}
        #table_data = []
        X_train_MM_dict, X_test_MM_dict, y_train_MM_dict, y_test_MM_dict = {}, {}, {}, {}
        y_train, y_test = {}, {}
        y_test_date = {}

        for mixed_method in mixed_methods:
            X_train, X_test, y_train_date, y_test_date[mixed_method] = generate_data(best_h[mixed_method]['MAPE'], best_c[mixed_method]['MAPE'],
                                                                                      covariates_names, numberOfSelectedCounties)
            y_test_date_temp = y_test_date[mixed_method]
            y_train[mixed_method] = y_train_date
            y_test[mixed_method] = y_test_date_temp
            mixed_model_covariates_names = list(X_train.columns)
            X_train_to_use = {method: None for method in methods}
            X_test_to_use = {method: None for method in methods}
            for method in none_mixed_methods:
                X_train_to_use[method] = X_train.copy()
                X_test_to_use[method] = X_test.copy()
                if method in models_to_log:
                    # make temporal and some fixed covariates logarithmic
                    negative_features=['temperature']
                    for covar in mixed_model_covariates_names:
                        if (' t' in covar) and (covar.split(' ')[0] not in negative_features) and (covar not in ['county_fips','date of day t']):
                            X_train_to_use[method][covar] = np.log((X_train_to_use[method][covar] + 1).astype(float))
                            X_test_to_use[method][covar] = np.log((X_test_to_use[method][covar] + 1).astype(float))

                    fix_log_list = ['total_population', 'population_density', 'area', 'median_household_income',
                                    'houses_density', 'airport_distance','deaths_per_100000']
                    for covar in fix_log_list:
                        if covar in mixed_model_covariates_names:
                            X_train_to_use[method][covar] = np.log((X_train_to_use[method][covar] + 1).astype(float))
                            X_test_to_use[method][covar] = np.log((X_test_to_use[method][covar] + 1).astype(float))

                X_train_dict[method] = X_train_to_use[method]
                X_test_dict[method] = X_test_to_use[method]
                y_train_dict[method] = y_train[mixed_method]
                y_test_dict[method] = y_test[mixed_method]

            GBM, GLM, KNN, NN = run_algorithms(X_train_dict, X_test_dict, y_train_dict, y_test_dict, best_loss, 0, spatial_mode, None)
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

            y_test_MM_dict[mixed_method] = np.array(y_test_MM_dict[mixed_method]['Target']).reshape(-1)
            y_train_MM_dict[mixed_method] = np.array(y_train_MM_dict[mixed_method]['Target']).reshape(-1)
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
        MM_GLM, MM_NN = run_mixed_models(X_train_MM_dict, X_test_MM_dict, y_train_MM_dict, y_test_MM_dict ,best_loss)
        y_prediction['MM_GLM'], y_prediction_train['MM_GLM'] = MM_GLM
        y_prediction['MM_NN'], y_prediction_train['MM_NN'] = MM_NN
        for mixed_method in mixed_methods:
            meanAbsoluteError, percentageOfAbsoluteError, adj_r_squared, second_error, meanAbsoluteScaledError = get_errors(best_h[mixed_method]['MAPE'],
            best_c[mixed_method]['MAPE'], mixed_method, y_prediction[mixed_method], y_prediction_train[mixed_method], y_test_date[mixed_method],
                                y_train[mixed_method] ,val_test_MASE_denominator[best_h[mixed_method]['MAPE']], numberOfSelectedCounties, target_name, mode='test')
            table_data.append([best_h[mixed_method]['MAPE'], best_c[mixed_method]['MAPE'], round(meanAbsoluteError, 2), round(percentageOfAbsoluteError, 2),
                                round(adj_r_squared, 2), round(second_error, 2), round(meanAbsoluteScaledError, 2)])

        table_name = 'table_of_best_test_results'
        plot_table(table_data, columns_table_t, methods, table_name, mode='test')
        push('a new table added')

        for method in mixed_methods:
          prediction=list(y_prediction_train[method])+list(y_prediction[method])
          df_for_prediction_plot[method]=prediction

        real_prediction_plot(df_for_prediction_plot,r,target_name,target_mode,best_h, maxHistory, spatial_mode, methods, numberOfSelectedCounties)

        # mail the test results
        selected_for_email = [test_address + '/tables', test_address + '/all_errors/NN', test_address + '/all_errors/KNN' , test_address + '/plots_of_real_prediction_values']
        zip_file_name = 'test results for h =' + str(maxHistory) + ' #counties=' + str(numberOfSelectedCountiesname)
        make_zip(selected_for_email, zip_file_name)
        send_email(zip_file_name + '.zip')

        # save the entire session
        filename = env_address + 'test.out'
        my_shelf = shelve.open(filename, 'n')  # 'n' for new
        for key in dir():
            try:
                my_shelf[key] = locals()[key]
            except:
                print('ERROR shelving: {0}'.format(key))
        my_shelf.close()



    #############################################################

    for h in history:
        data = makeHistoricalData(h, r, target_name, 'mrmr', spatial_mode, target_mode, './')
        data = clean_data(data, numberOfSelectedCounties)

        # pre-process and split the data, 'date's have dates info
        X_train_train_to_use = {method: None for method in methods}
        X_train_val_to_use = {method: None for method in methods}
        X_test_to_use = {method: None for method in methods}
        X_train_train, X_train_val, X_test, y_train_train_date, y_train_val_date, y_test_date = preprocess(data, 1)

        train_val_MASE_denominator[h], val_test_MASE_denominator[h], train_lag_MASE_denominator[h] = mase_denominator(r, h, target_name, target_mode, numberOfSelectedCounties, spatial_mode)

        for method in methods:
            X_train_train_to_use[method] = X_train_train.copy()
            X_train_val_to_use[method]= X_train_val.copy()
            X_test_to_use[method] = X_test.copy()
            if method in models_to_log:
                # make temporal and some fixed covariates logarithmic
                negative_features=['temperature']
                for covar in covariates_names:
                    if (' t' in covar) and (covar.split(' ')[0] not in negative_features):
                        X_train_train_to_use[method][covar] = np.log((X_train_train_to_use[method][covar] + 1).astype(float))
                        X_train_val_to_use[method][covar] = np.log((X_train_val_to_use[method][covar] + 1).astype(float))
                        X_test_to_use[method][covar] = np.log((X_test_to_use[method][covar] + 1).astype(float))

                fix_log_list = ['total_population', 'population_density', 'area', 'median_household_income',
                                'houses_density', 'airport_distance','deaths_per_100000']
                for covar in fix_log_list:
                    if covar in covariates_names:
                        X_train_train_to_use[method][covar] = np.log((X_train_train_to_use[method][covar] + 1).astype(float))
                        X_train_val_to_use[method][covar] = np.log((X_train_val_to_use[method][covar] + 1).astype(float))
                        X_test_to_use[method][covar] = np.log((X_test_to_use[method][covar] + 1).astype(float))

      
        y_train_date = (pd.DataFrame(y_train_train_date).append(pd.DataFrame(y_train_val_date))).reset_index(drop=True)
        y_train_train = np.array(y_train_train_date['Target']).reshape(-1)
        y_train_val = np.array(y_train_val_date['Target']).reshape(-1)
        y_test = np.array(y_test_date['Target']).reshape(-1)
        y_train = np.array((pd.DataFrame(y_train_train).append(pd.DataFrame(y_train_val))).reset_index(drop=True)).reshape(-1)


        

        # find best loss
        if (h==1):
          best_loss = update_best_loss('none_mixed_model',spatial_mode,None,best_loss,X_train_train_to_use,X_train_val_to_use,\
                                      y_train_train,y_train_val,None,None,data.columns.drop(['Target','date of day t','county_fips']),\
                                        numberOfCovariates,maxC)

        
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
            
            for method in none_mixed_methods:
                X_train_train_temp = X_train_train_to_use[method][covariates_list]
                X_train_val_temp = X_train_val_to_use[method][covariates_list]
                
                loom.add_function(parallel_run, [method, X_train_train_temp, X_train_val_temp, y_train_train, y_train_val, best_loss, indx_c])
                
              
            if indx_c == maxC:
                            break  
        # run the processes in parallel

        parallel_outputs['non_mixed'] = loom.execute()
        
        ind = 0
        for c in range(1, numberOfCovariates + 1):
            for method in none_mixed_methods:
                y_prediction[method][(h, c)], y_prediction_train[method][(h, c)] = parallel_outputs['non_mixed'][ind]['output']
                ind += 1
            if c == maxC:
              break
            
        # save the entire session for each h and c
        filename = env_address + 'validation.out'
        my_shelf = shelve.open(filename, 'n')  # 'n' for new
        for key in dir():
            try:
                my_shelf[key] = locals()[key]
            except:
                print('ERROR shelving: {0}'.format(key))
        my_shelf.close()
        

        # find best loss
        if h == 1 :
          best_loss = update_best_loss('mixed_model',spatial_mode,None,best_loss,None,None,y_train_train,\
                    y_train_val,y_prediction_train,y_prediction,None,\
                    numberOfCovariates,maxC)
        # initiate loom for parallel processing        
        loom = ProcessLoom(max_runner_cap=len(base_data.columns) * len(mixed_methods) + 5)
        indx_c = 0
        for c in range(1, numberOfCovariates + 1):
            indx_c += 1
            for mixed_method in mixed_methods:
                y_predictions_test, y_predictions_train = [], []
                # Construct the outputs for the testing dataset of the 'MM' methods
                y_predictions_test.extend([y_prediction['GBM'][(h, c)], y_prediction['GLM'][(h, c)],
                                            y_prediction['KNN'][(h, c)], y_prediction['NN'][(h, c)]])
                y_prediction_test_np = np.array(y_predictions_test).reshape(len(y_predictions_test), -1)
                X_test_mixedModel = pd.DataFrame(y_prediction_test_np.transpose())
                # Construct the outputs for the training dataset of the 'MM' methods
                y_predictions_train.extend([y_prediction_train['GBM'][(h, c)], y_prediction_train['GLM'][(h, c)],
                                            y_prediction_train['KNN'][(h, c)], y_prediction_train['NN'][(h, c)]])
                y_prediction_train_np = np.array(y_predictions_train).reshape(len(y_predictions_train), -1)
                X_train_mixedModel = pd.DataFrame(y_prediction_train_np.transpose())
            

                loom.add_function(mixed_parallel_run, [mixed_method, X_train_mixedModel, X_test_mixedModel, y_train_train, y_train_val, best_loss])
            if c == maxC:
                break

        # run the processes in parallel
        parallel_outputs['mixed'] = loom.execute()
        ind = 0
        for c in range(1, numberOfCovariates + 1):
            for mixed_method in mixed_methods:
                y_prediction[mixed_method][(h, c)], y_prediction_train[mixed_method][(h, c)] = parallel_outputs['mixed'][ind]['output']
                y_prediction[mixed_method][(h, c)] = np.array(y_prediction[mixed_method][(h, c)]).ravel()
                y_prediction_train[mixed_method][(h, c)] = np.array(y_prediction_train[mixed_method][(h, c)]).ravel()
                ind += 1
            if c == maxC:
                    break

        # save the entire session for each h and c
        filename = env_address + 'validation.out'
        my_shelf = shelve.open(filename, 'n')  # 'n' for new
        for key in dir():
            try:
                my_shelf[key] = locals()[key]
            except:
                print('ERROR shelving: {0}'.format(key))
        my_shelf.close()

        number_of_improved_methods = 0 # we count number_of_improved_methods to run test if no method have improved in current h

        indx_c = 0
        covariates_list=['county_fips','date of day t']
        for c in covariates_names:  # iterate through sorted covariates
            indx_c += 1
            for covariate in data.columns:  # add all historical covariates of this covariate and create a feature
                if c.split(' ')[0] in covariate:
                    covariates_list.append(covariate)
            y_val = np.array(y_train_val_date['Target']).reshape(-1)

            for method in methods:
                X_train_train_temp = X_train_train_to_use[method][covariates_list]
                X_train_val_temp = X_train_val_to_use[method][covariates_list]
                X_test_temp = X_test_to_use[method][covariates_list]

                validation_errors['MAE'][method][(h, indx_c)], validation_errors['MAPE'][method][(h, indx_c)], \
                validation_errors['adj-R2'][method][(h, indx_c)], validation_errors['sec'][method][(h, indx_c)], \
                validation_errors['MASE'][method][(h, indx_c)] = \
                    get_errors(h, indx_c, method, y_prediction[method][(h, indx_c)], y_prediction_train[method][(h, indx_c)], y_train_val_date,
                               y_train_train_date, train_val_MASE_denominator[h],
                                numberOfSelectedCounties ,target_name , mode='val')

                # find best errors
                for error in error_names:
                    if validation_errors[error][method][(h, indx_c)] < minError[method][error]:
                        minError[method][error] = validation_errors[error][method][(h, indx_c)]
                        best_h[method][error] = h
                        best_c[method][error] = indx_c
                        if error == 'MAPE':
                            number_of_improved_methods += 1

                        if error == 'MAPE' and method != 'MM_GLM' and method != 'MM_NN':
                            historical_X_train[method] = (X_train_train_temp.append(X_train_val_temp)).reset_index(
                                drop=True)
                            historical_X_test[method] = X_test_temp
                            historical_y_train[method] = y_train
                            historical_y_test[method] = y_test
                            historical_y_train_date[method] = y_train_date
                            historical_y_test_date[method] = y_test_date

            if indx_c == maxC:
                        break
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
        push('logs of h=' + str(h) + ' added')

        # we run test if none of models have improved in curent h or if we passed half of maxhistory 
        if (number_of_improved_methods == 0) or (h == maxHistory//2) :###########################
          print('jump to test process')
          test_process(h, r, target_name,spatial_mode, target_mode,best_h,best_c,historical_X_train,\
                    historical_X_test, historical_y_train_date, historical_y_test_date, best_loss,\
                    numberOfSelectedCounties, covariates_names, maxHistory,train_val_MASE_denominator,\
                     val_test_MASE_denominator, test_address, env_address, mail_address)
        

    # plot table for best results
    table_data = []
    for method in methods:
        table_data.append([best_h[method]['MAPE'], best_c[method]['MAPE'], round(minError[method]['MAE'], 2),
                            round(minError[method]['MAPE'], 2), round(minError[method]['adj-R2'], 2),
                            round(minError[method]['sec'], 2), round(minError[method]['MASE'], 2)])
    table_name = 'tabel_of_best_validation_results'
    plot_table(table_data, columns_table, methods, table_name, mode='val')
    # plot the results of methods on validation set

    for error in error_names:
        plot_results(3, 2, numberOfCovariates, methods, history, validation_errors[error], complete_error_names[error])

    # mail the validation results
    selected_for_email = [validation_address]
    zip_file_name = 'validation results for h =' + str(maxHistory) + ' #counties=' + str(numberOfSelectedCountiesname)
#     make_zip(selected_for_email, zip_file_name)
#     send_email(zip_file_name + '.zip')
    push('plots added')
    ################################################################################################################# test zone
    test_process(h, r, target_name,spatial_mode, target_mode,best_h,best_c,historical_X_train,\
                    historical_X_test, historical_y_train_date, historical_y_test_date, best_loss,\
                    numberOfSelectedCounties, covariates_names, maxHistory,train_val_MASE_denominator,\
                     val_test_MASE_denominator, test_address, env_address, mail_address)


if __name__ == "__main__":

    begin = time.time()
    maxHistory = 14
    maxC = 100
    
    # make directories for saving the results
    validation_address = './'+'results/counties=' + str(numberOfSelectedCountiesname) + ' max_history=' + str(maxHistory) + '/validation/'
    test_address = './' + 'results/counties=' + str(numberOfSelectedCountiesname) + ' max_history=' + str(maxHistory) + '/test/'
    env_address = './' + 'results/counties=' + str(numberOfSelectedCountiesname) + ' max_history=' + str(maxHistory) + '/session_parameters/'
    mail_address = './results/counties=' + str(numberOfSelectedCountiesname) + ' max_history=' + str(maxHistory) + '/email'

    if not os.path.exists(mail_address):
        os.makedirs(mail_address)
    if not os.path.exists(test_address):
        os.makedirs(test_address)
    if not os.path.exists(validation_address):
        os.makedirs(validation_address)
    if not os.path.exists(env_address):
        os.makedirs(env_address)
    push('new folders added')
    models_to_log = ['NN', 'GLM', 'GBM'] # models we want to make the features logarithmic for them, we remove KNN
    main(maxHistory)
    end = time.time()
    push('final results added')
    print("The total time of execution in minutes: ", round((end - begin) / 60, 2))
