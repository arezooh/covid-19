from sitemakeHistoricalData import makeHistoricalData
from models import GBM, GLM, KNN, NN, MM_GLM, GBM_grid_search, NN_grid_search
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import time
import os
import random
import tensorflow as tf
from numpy.random import seed
import requests

address = './csvFiles/'


seed(1)
tf.random.set_seed(1)

numberOfSelectedCounties = -1
target_name = 'death'
spatial_mode = 'country'
future_mode = False
pivot = 'country'
test_size = 28

first_run = 1

r_list=[14,30]#


best_methods={'daily':{14:'MM_GLM'},'weekly':{4:'KNN'}}
county_errors = {error:None for error in ['MAPE','MASE','MAE','RMSE']}
country_errors = {error:None for error in ['MAPE','MASE','MAE','RMSE']}

best_h={'daily':{14:3#8#2#10
                 },'weekly':{4:1}}
best_c={'daily':{14:1#30#38#9
                 },'weekly':{4:30}}


future_features = ['social-distancing-travel-distance-grade', 'social-distancing-encounters-grade',
                           'social-distancing-total-grade']  # sorted by their mrmr rank
force_features = []

best_scaler = 1

all_covariates = ['death', 'confirmed',
       'houses_density', 'virus-pressure', 'meat_plants', 'longitude',
       'social-distancing-travel-distance-grade', 'passenger_load',
       'total_population', 'population_density', 'temperature', 'weekend',
       'icu_beds', 'party', 'age_60_79', 'female-percent',
       'daily-state-test', 'precipitation', 'median_household_income',
       'total_college_population', 'airport_distance', 'Religious', 'smokers',
       'social-distancing-encounters-grade', '%insured',
       'ventilator_capacity', 'area', 'age_15_39', 'diabetes', 'hospital_beds',
       'high_school_diploma_only', 'age_0_14', 'latitude',
       'social-distancing-total-grade', 'deaths_per_100000',
       'less_than_high_school_diploma', 'age_40_59', 'some_college_or_higher', 'age_80_or_higher']

selected_covariates ={
       'daily':{14:all_covariates[:best_c['daily'][14]]},          
       'weekly':{4:all_covariates[:best_c['weekly'][4]]}}

output = pd.DataFrame(columns=['Prediction Date','Date', 'Real', 'Prediction', 'Model', 'Country MASE (test)',\
                               'Country MAPE (test)', 'Country MAE (test)', 'Country RMSE (test)', 'County MASE (test)',\
                               'County MAPE (test)', 'County MAE (test)', 'County RMSE (test)'])

none_mixed_methods = ['GBM', 'GLM', 'KNN', 'NN']
mixed_methods = ['MM_GLM', 'MM_NN']
models_to_log = ['NN', 'GLM', 'GBM', 'KNN']
best_loss = {'GBM': 'least_absolute_deviation', 'MM_NN': 'MeanAbsoluteError', 'NN': 'MeanAbsoluteError'}#MeanAbsoluteError

temporal_data=pd.read_csv(address + 'temporal-data.csv')
temporal_data['date'] = temporal_data['date'].apply(lambda x : datetime.datetime.strptime(x, '%m/%d/%y'))
current_date = max(temporal_data['date'])

zero_removing = 1
# if pivot == 'country':
#     zero_removing = 0

######################################################### split data to train, val, test
def splitData(numberOfCounties, main_data, target, spatial_mode, mode):
    numberOfCounties = len(main_data['county_fips'].unique())
    main_data = main_data.sort_values(by=['date of day t', 'county_fips'])
    target = target.sort_values(by=['date of day t', 'county_fips'])
    # we set the base number of days to the minimum number of days existed between the counties
    # and then compute the validation size for the non-default state.
    baseNumberOfDays = (main_data.groupby(['county_fips']).size()).min()
    val_size = round(0.3 * (baseNumberOfDays - test_size))

    if mode == 'val':
        if not future_mode:  # the default state
            X_train_train = main_data.iloc[:-2 * (r * numberOfCounties), :].sort_values(
                by=['county_fips', 'date of day t'])
            X_train_val = main_data.iloc[-2 * (r * numberOfCounties):-(r * numberOfCounties), :].sort_values(
                by=['county_fips', 'date of day t'])
            X_test = main_data.tail(r * numberOfCounties).sort_values(by=['county_fips', 'date of day t'])

            y_train_train = target.iloc[:-2 * (r * numberOfCounties), :].sort_values(
                by=['county_fips', 'date of day t'])
            y_train_val = target.iloc[-2 * (r * numberOfCounties):-(r * numberOfCounties), :].sort_values(
                by=['county_fips', 'date of day t'])
            y_test = target.tail(r * numberOfCounties).sort_values(by=['county_fips', 'date of day t'])
        else:
            X_test = main_data.tail(test_size * numberOfCounties).copy()
            X_train_val = main_data.iloc[:-(test_size * numberOfCounties)].tail(val_size * numberOfCounties).copy()
            X_train_train = main_data.iloc[:-((val_size + test_size) * numberOfCounties)].copy()

            y_test = target.tail(test_size * numberOfCounties).copy()
            y_train_val = target.iloc[:-(test_size * numberOfCounties)].tail(val_size * numberOfCounties).copy()
            y_train_train = target.iloc[:-((val_size + test_size) * numberOfCounties)].copy()

        return X_train_train, X_train_val, X_test, y_train_train, y_train_val, y_test

    if mode == 'test':
        if not future_mode:
            X_train = main_data.iloc[:-(r * numberOfCounties), :].sort_values(by=['county_fips', 'date of day t'])
            X_test = main_data.tail(r * numberOfCounties).sort_values(by=['county_fips', 'date of day t'])

            y_train = target.iloc[:-(r * numberOfCounties), :].sort_values(by=['county_fips', 'date of day t'])
            y_test = target.tail(r * numberOfCounties).sort_values(by=['county_fips', 'date of day t'])

        else:
            X_test = main_data.tail(test_size * numberOfCounties).copy()
            X_train = main_data.iloc[:-(test_size * numberOfCounties)].copy()

            y_test = target.tail(test_size * numberOfCounties).copy()
            y_train = target.iloc[:-(test_size * numberOfCounties)]

        return X_train, X_test, y_train, y_test
    
    
########################################################### clean data
def clean_data(data, numberOfSelectedCounties, spatial_mode):
    global numberOfDays
    data = data.sort_values(by=['county_fips', 'date of day t'])
    # select the number of counties we want to use
    # numberOfSelectedCounties = numberOfCounties
    if numberOfSelectedCounties == -1:
        numberOfSelectedCounties = len(data['county_fips'].unique())

    using_data = data[(data['county_fips'] <= data['county_fips'].unique()[numberOfSelectedCounties - 1])]
    using_data = using_data.reset_index(drop=True)
    if (spatial_mode == 'county') or (spatial_mode == 'country'):
        if pivot == 'county' :
            main_data = using_data.drop(['county_name', 'state_fips', 'state_name'],
                                        axis=1)  # , 'date of day t'
        elif pivot == 'state':
            main_data = using_data.drop(['county_name'],
                                        axis=1)  # , 'date of day t'
        elif pivot == 'country':
            main_data = using_data

    elif (spatial_mode == 'state'):
        main_data = using_data.drop(['county_name', 'state_name'],
                                    axis=1)
    numberOfDays = len(using_data['date of day t'].unique())

    return main_data


########################################################### preprocess
def preprocess(main_data, spatial_mode, validationFlag):
    if spatial_mode == 'state':
        target = pd.DataFrame(main_data[['date of day t', 'county_fips', 'state_fips', 'Target']])
    else:
        target = pd.DataFrame(main_data[['date of day t', 'county_fips', 'Target']])

    main_data = main_data.drop(['Target'], axis=1)

    # produce train, validation and test data
    if validationFlag:  # validationFlag is 1 if we want to have a validation set and 0 otherwise

        X_train_train, X_train_val, X_test, y_train_train, y_train_val, y_test = splitData(numberOfSelectedCounties,
                                                                                           main_data, target,
                                                                                           spatial_mode, 'val')
        return X_train_train, X_train_val, X_test, y_train_train, y_train_val, y_test

    else:

        X_train, X_test, y_train, y_test = splitData(numberOfSelectedCounties, main_data, target, spatial_mode, 'test')
        return X_train, X_test, y_train, y_test
    
########################################################### logarithmize covariates
def logarithm_covariates(data):
    # make temporal and some fixed covariates logarithmic
    if temporal_mode == 'daily':
      negative_features = ['temperature',]
    # for weekly average mode we dont logarithm the target so its bether not to logarithm its history
    elif temporal_mode == 'weekly':
      negative_features = ['temperature',target_name]
    
    for covar in data.columns:
        if (' t' in covar) and (covar.split(' ')[0] not in negative_features) and (
                covar not in ['county_fips', 'date of day t']):
            data[covar] = np.log((data[covar] + 1).astype(float))

    fix_log_list = ['total_population', 'population_density', 'area', 'median_household_income',
                    'houses_density', 'airport_distance', 'deaths_per_100000']
    for covar in fix_log_list:
        if covar in data.columns:
            data[covar] = np.log((data[covar] + 1).astype(float))
    return(data)
    
########################################################### run algorithms in parallel except mixed models
def run_algorithms(X_train, X_val, y_train, y_val, best_loss, algorithm , mode):
    
    from models import GBM, GLM, KNN, NN
    
    if mode =='test': y_val[pd.isnull(y_val['Target'])]['Target']=1 # it doesnt have values for test mode and we set these values to 1 to preventing errors
        
    y_prediction={method:None for method in none_mixed_methods+mixed_methods}
    y_prediction_train={method:None for method in none_mixed_methods+mixed_methods}
    Xtrain={method:None for method in none_mixed_methods+mixed_methods}
    Xval={method:None for method in none_mixed_methods+mixed_methods}
    
    X_train = X_train.drop(['county_fips', 'date of day t'], axis=1)
    X_val = X_val.drop(['county_fips', 'date of day t'], axis=1)
    y_train = np.array(y_train['Target']).reshape(-1)
    y_val = np.array(y_val['Target']).reshape(-1)
    
    for method in none_mixed_methods:
        Xtrain[method] = X_train
        Xval[method] = X_val
        if method in models_to_log:
            Xtrain[method] = logarithm_covariates(Xtrain[method])
            Xval[method] = logarithm_covariates(Xval[method])
        
    if algorithm == 'GBM' or algorithm in mixed_methods:
        y_prediction['GBM'], y_prediction_train['GBM'] = GBM(Xtrain['GBM'], Xval['GBM'], y_train, best_loss['GBM'])
        
    if algorithm == 'GLM' or algorithm in mixed_methods:
        y_prediction['GLM'], y_prediction_train['GLM'] = GLM(Xtrain['GLM'], Xval['GLM'], y_train)
        
    if algorithm == 'KNN' or algorithm in mixed_methods:
        y_prediction['KNN'], y_prediction_train['KNN'] = KNN(Xtrain['KNN'], Xval['KNN'], y_train)
        
    if algorithm == 'NN' or algorithm in mixed_methods:
        y_prediction['NN'], y_prediction_train['NN'] = NN(Xtrain['NN'], Xval['NN'], y_train, y_val, best_loss['NN'])
    
    if algorithm in mixed_methods:
        
        y_predictions_test, y_predictions_train = [], []
        # Construct the outputs for the testing dataset of the 'MM' methods
        y_predictions_test.extend([y_prediction['GBM'], y_prediction['GLM'], y_prediction['KNN'], y_prediction['NN']])
        y_prediction_test_np = np.array(y_predictions_test).reshape(len(y_predictions_test), -1)
        X_test_mixedModel = pd.DataFrame(y_prediction_test_np.transpose())
        # Construct the outputs for the training dataset of the 'MM' methods
        y_predictions_train.extend(
            [y_prediction_train['GBM'], y_prediction_train['GLM'], y_prediction_train['KNN'], y_prediction_train['NN']])
        y_prediction_train_np = np.array(y_predictions_train).reshape(len(y_predictions_train), -1)
        X_train_mixedModel = pd.DataFrame(y_prediction_train_np.transpose())
        
        if algorithm == 'MM_GLM':
            y_prediction['MM_GLM'], y_prediction_train['MM_GLM'] = GLM(X_train_mixedModel, X_test_mixedModel, y_train)
        elif algorithm == 'MM_NN':
            y_prediction['MM_NN'], y_prediction_train['MM_NN'] = NN(X_train_mixedModel, X_test_mixedModel, y_train, y_val, best_loss['NN'])

    
    return(y_prediction[algorithm], y_prediction_train[algorithm])

########################################################### get errors for each model in each h and c
def get_errors(y_prediction, y_prediction_train, y_test_date, y_train_date, regular_data, numberOfSelectedCounties):
    
    # y_test_date and y_train_date are a dataframes with columns ['date of day t', 'county_fips', 'Target']
    # set negative predictions to zero
    y_prediction[y_prediction < 0] = 0
    y_test = np.array(y_test_date['Target']).reshape(-1)
    
    county_errors = {error: None for error in
                      ['MAE', 'MAPE','MASE','RMSE']}
    # country_errors show error for prediction of target variable for whole country
    country_errors = {error: None for error in
                      ['MAE', 'MAPE','MASE','RMSE']}

    if numberOfSelectedCounties == -1:
        numberOfSelectedCounties = len(y_test_date['county_fips'].unique())
        
    ##################################### MASE denominator
    X_train_train, X_train_val, X_test, mase_y_train_train_date, mase_y_train_val_date, mase_y_test_date = preprocess(regular_data,
                                                                                                       spatial_mode, 1)

    train_train = (mase_y_train_train_date.reset_index(drop=True)).sort_values(by=['date of day t', 'county_fips'])
    train_val = (mase_y_train_val_date.reset_index(drop=True)).sort_values(by=['date of day t', 'county_fips'])

    train_train = train_train.tail(len(train_val)).rename(
        columns={'Target': 'train-Target', 'date of day t': 'train-date'})
    train_val = train_val.rename(columns={'Target': 'val-Target', 'date of day t': 'val-date'})

    df_for_train_val_MASE_denominator = pd.concat(
        [train_train.reset_index(drop=True), train_val.reset_index(drop=True)], axis=1)
    df_for_train_val_MASE_denominator['absolute-error'] = abs(df_for_train_val_MASE_denominator['val-Target'] -
                                                              df_for_train_val_MASE_denominator['train-Target'])
    train_val_MASE_denominator = df_for_train_val_MASE_denominator['absolute-error'].mean()

    # we need to have mase denominator based on target values for whole country (sum of target for all counties)
    # this will be used for calculation of country error
    df_for_train_val_MASE_denominator_country = df_for_train_val_MASE_denominator.groupby(['val-date']).sum()
    df_for_train_val_MASE_denominator_country['absolute-error'] = abs(
        df_for_train_val_MASE_denominator_country['val-Target'] -
        df_for_train_val_MASE_denominator_country['train-Target'])

    train_val_MASE_denominator_country = df_for_train_val_MASE_denominator_country['absolute-error'].mean()
    #####################################


    # if target mode is logarithmic we need to return the target variable to its original state
    if target_mode == 'logarithmic':
        print('logarithmic')
        y_test = np.array(np.round(np.exp(y_test) - 1)).reshape(-1)
        y_test_date['Target'] = list(np.round(np.exp(y_test_date['Target']) - 1))
        y_prediction = np.array(np.exp(y_prediction) - 1).reshape(-1)

    # make predictions rounded to their closest number
    y_prediction = np.array(y_prediction)
    if target_mode != 'weeklyaverage':
        y_prediction = np.round(y_prediction)
    # for calculating the country error we must sum up all the county's target values to get country target value
    y_test_date['prediction'] = y_prediction
    y_test_date.to_csv('errors.csv')
    y_test_date_country = y_test_date.groupby(['date of day t']).sum()
    y_test_country = np.array(y_test_date_country['Target']).reshape(-1)
    y_prediction_country = np.array(y_test_date_country['prediction']).reshape(-1)
    
    ############################################################## calculate whole country error
    min_error = 1e10
    best_scaler = 1
    for i in range(10):
        print(i+1)
        error = mean_absolute_error(y_test_country, np.array(y_prediction_country)*(i+1))
        if error < min_error :
            min_error = error
            best_scaler = i+1
    print('best_scaler: ',best_scaler)
    # best_scaler = 1
    y_prediction = np.array(y_prediction)*best_scaler
    country_errors['MAE'] = mean_absolute_error(y_test_country, y_prediction_country)
    rootMeanSquaredError = np.sqrt(mean_squared_error(y_test_country, y_prediction_country))
    country_errors['RMSE']= rootMeanSquaredError
    sumOfAbsoluteError = sum(abs(y_test_country - y_prediction_country))
    country_errors['MAPE'] = (sumOfAbsoluteError / sum(y_test_country)) * 100
    y_test_temp_country = y_test_country.copy()
    y_test_temp_country[y_test_country == 0] = 1
    y_prediction_temp_country = y_prediction_country.copy()
    y_prediction_temp_country[y_test_country == 0] += 1
    
    MASE_numerator = sumOfAbsoluteError / len(y_test_country)
    country_errors['MASE'] = MASE_numerator / train_val_MASE_denominator_country
    

    ############################################################## calculate county error
    y_prediction = np.array(y_prediction)*best_scaler
    county_errors['MAE'] = mean_absolute_error(y_test, y_prediction)
    print("Mean Absolute Error of ", county_errors['MAE'])
    rootMeanSquaredError = np.sqrt(mean_squared_error(y_test, y_prediction))
    county_errors['RMSE']= rootMeanSquaredError
    sumOfAbsoluteError = sum(abs(y_test - y_prediction))
    county_errors['MAPE'] = (sumOfAbsoluteError / sum(y_test)) * 100
    # we change zero targets into 1 and add 1 to their predictions
    y_test_temp = y_test.copy()
    y_test_temp[y_test == 0] = 1
    y_prediction_temp = y_prediction.copy()
    y_prediction_temp[y_test == 0] += 1
    # meanPercentageOfAbsoluteError = sum((abs(y_prediction_temp - y_test_temp) / y_test_temp) * 100) / len(y_test)
    print("Percentage of Absolute Error of ", county_errors['MAPE'])
    MASE_numerator = sumOfAbsoluteError / len(y_test)
    county_errors['MASE'] = MASE_numerator / train_val_MASE_denominator
    print("MASE Error of ", county_errors['MASE'])

    print("-----------------------------------------------------------------------------------------")

    # # save outputs in 'out.txt'
    # sys.stdout = orig_stdout
    # f.close()
    
    return county_errors, country_errors, best_scaler

###################################### make dates nominal
def date_nominalize(data,temporal_mode,column_name):
    if column_name == 'Date':
        if temporal_mode == "daily" :
            data['Date'] = data['Date'].apply(lambda x: datetime.datetime.strftime(x,"%d %b %Y"))#.strptime(x,'%m/%d/%y')
        elif temporal_mode == "weekly" :
            weeks = data.sort_values(by=['Date'])['Date'].unique()
            for index,week in enumerate(weeks) :
                week_first_day = current_date + datetime.timedelta(days=1+(index*7))
                week_end_day = week_first_day + datetime.timedelta(days=6)
                week_dates = week_first_day.strftime("%d %b")+' -- '+week_end_day.strftime("%d %b %Y")
                data['Date'] = data['Date'].replace(week,week_dates)
    if column_name == 'Prediction Date':
        data['Prediction Date'] = data['Prediction Date'].apply(lambda x: datetime.datetime.strptime(x,'%m/%d/%y').strftime("%d %b %Y"))
    return(data)

###################################################### make data weekly
def make_weekly(dailydata):
    dailydata.sort_values(by=['date','county_fips'],inplace=True)
    numberofcounties=len(dailydata['county_fips'].unique())
    numberofweeks=len(dailydata['date'].unique())//7

    weeklydata=pd.DataFrame(columns=dailydata.columns)

    for i in range(numberofweeks):
        temp_df=dailydata.tail(numberofcounties*7) # weekly average of last week for all counties
        date=temp_df.tail(1)['date'].iloc[0]
        dailydata=dailydata.iloc[:-(numberofcounties*7),:]
        temp_df=temp_df.drop(['date'],axis=1)
        temp_df=temp_df.groupby(['county_fips']).mean().reset_index()
        temp_df['date']=date # last day of week 
        weeklydata=weeklydata.append(temp_df)
    weeklydata.sort_values(by=['county_fips','date'],inplace=True)
    weeklydata=weeklydata.reset_index(drop=True)
    return(weeklydata)

def get_csv(web_addres,file_address):
    url=web_addres
    print(url)
    req = requests.get(url)
    url_content = req.content
    csv_file = open(file_address, 'wb')
    csv_file.write(url_content)
    csv_file.close
############################################## getting updated real values from data source
def correct_negative_numbers(data):
    data2=data.copy()
    reverse_dates=data.columns[4:][::-1]
    while data.iloc[:,4:].min().sum()<0:
        for i in range(len(reverse_dates)):
            date = reverse_dates[i]
            negative_index=data[data[date]<0].index
            data2.loc[negative_index,date] = 0
        for i in range(len(reverse_dates)):
            date = reverse_dates[i]
            past_date = reverse_dates[i+1]
            negative_index=data[data[date]<0].index
            data2.loc[negative_index,past_date] = data2.loc[negative_index,past_date]+data.loc[negative_index,date]
            if i==len(reverse_dates)-2:
                break
        data=data2.copy()
    return(data)

def get_updated_covid_data(address):

    get_csv('https://usafactsstatic.blob.core.windows.net/public/data/covid-19/covid_confirmed_usafacts.csv',\
            address+'covid_confirmed_cases.csv')
    get_csv('https://usafactsstatic.blob.core.windows.net/public/data/covid-19/covid_deaths_usafacts.csv',\
            address+'covid_deaths.csv')

    fix=pd.read_csv(address+'fixed-data.csv')
    cof=pd.read_csv(address+'covid_confirmed_cases.csv')
    det=pd.read_csv(address+'covid_deaths.csv')

    valid_dates=cof.columns[4:].tolist()

    fips=pd.DataFrame(columns=['fips'])
    fips['fips']=fix['county_fips'].tolist()*len(valid_dates)
    fips.sort_values(by='fips',inplace=True)
    temporal_data=pd.DataFrame(columns=['county_fips','date']) # create template for data
    temporal_data['county_fips']=fips['fips']
    temporal_data['date']=valid_dates*3142 

    temporal_data['date']=temporal_data['date'].apply(lambda x: datetime.datetime.strptime(x,'%m/%d/%y'))

    temporal_data.sort_values(by=['county_fips','date'],inplace=True)

    cof2=cof.copy()
    for i in range(5,cof.shape[1]):
        cof2.iloc[:,i]=cof.iloc[:,i]-cof.iloc[:,i-1]
    cof=cof2.copy()

    det2=det.copy()
    for i in range(5,det.shape[1]):
        det2.iloc[:,i]=det.iloc[:,i]-det.iloc[:,i-1]
    det=det2.copy()

    cof=cof[cof['countyFIPS'].isin(temporal_data['county_fips'])]
    det=det[det['countyFIPS'].isin(temporal_data['county_fips'])]

    cof = correct_negative_numbers(cof)
    det = correct_negative_numbers(det)

    cof=cof[cof['countyFIPS'].isin(temporal_data['county_fips'])]
    det=det[det['countyFIPS'].isin(temporal_data['county_fips'])]

    for i in cof.columns[4:cof.shape[1]]:
        j=datetime.datetime.strptime(i,'%m/%d/%y')
        temporal_data.loc[temporal_data['date']==j,'confirmed']=cof[i].copy().tolist()

    for i in det.columns[4:det.shape[1]]:
        j=datetime.datetime.strptime(i,'%m/%d/%y')
        temporal_data.loc[temporal_data['date']==j,'death']=det[i].copy().tolist()
    return(temporal_data)

#################################################### add past real values to output csv

def add_real_values(data,address):
    data['prediction datetime'] = data['Prediction Date'].apply(lambda x : datetime.datetime.strptime(x,"%Y-%m-%d"))
    data['datetime'] = data['Date'].apply(lambda x : datetime.datetime.strptime(x[-11:],"%d %b %Y"))
    previous_prediction_date = data.tail(1)['prediction datetime'].iloc[0]

    # we need past data to find number of counties contained in country prediction
    past_data = makeHistoricalData(best_h['daily'][14], 14, test_size, target_name, 'mrmr', spatial_mode, 'regular', './',
                                  future_features, 'county', previous_prediction_date, 0)
    counties = past_data['county_fips'].unique()
    temporal_data = get_updated_covid_data(address)
    temp = temporal_data[temporal_data['county_fips'].isin(counties)]

    first_day = previous_prediction_date
    last_day = previous_prediction_date + datetime.timedelta(days=14)
    daily_part = temp[(temp['date']>first_day) & (temp['date']<=last_day)]
    if len(daily_part)>0 : last_day = max(daily_part['date'])
    daily_part['naive_prediction'] = list(temp.loc[(temp['date']> (first_day - datetime.timedelta(days=14))) & (temp['date']<=(last_day - datetime.timedelta(days=14))),target_name])
    daily_part = daily_part.groupby(['date']).sum().reset_index()

    first_day = previous_prediction_date + datetime.timedelta(days=14)
    last_day = previous_prediction_date + datetime.timedelta(days=28)
    weekly_part = temp[(temp['date']>first_day) & (temp['date']<=last_day)]
    if len(weekly_part)>0 : last_day = max(weekly_part['date'])
    weekly_part['naive_prediction'] = list(temp.loc[(temp['date']> (first_day - datetime.timedelta(days=28))) & (temp['date']<=(last_day - datetime.timedelta(days=28))),target_name])
    weekly_part = weekly_part.groupby(['date']).sum().reset_index()
    weekly_part['county_fips'] = 1
    weekly_part = make_weekly(weekly_part)

    real_values_df=daily_part[['date','death','naive_prediction']].append(weekly_part[['date','death','naive_prediction']])

    data= pd.merge(data, real_values_df, how='left', left_on=['datetime'],right_on=['date'])
    new_ind=~pd.isnull(data['death'])
    # add real values
    data.loc[new_ind,'Real'] = data.loc[~pd.isnull(data['death']),'death']
    # calculate errors
    data.loc[new_ind,'Country MAE (test)']=abs(data.loc[new_ind,'Real']-data.loc[new_ind,'Prediction'])
    data.loc[new_ind,'Country RMSE (test)']=np.sqrt((data.loc[new_ind,'Real']-data.loc[new_ind,'Prediction'])**2)
    data.loc[new_ind,'Naive MAE']=abs(data.loc[new_ind,'Real']-data.loc[new_ind,'naive_prediction'])
    data.loc[new_ind,'Country MASE (test)'] = data.loc[new_ind,'Country MAE (test)'] / data.loc[new_ind,'Naive MAE']
    data.loc[new_ind,'corrected_real']=data.loc[new_ind,'Real'].replace(0,1)
    data.loc[new_ind,'corrected_prediction']=data.loc[new_ind,'Prediction']
    data.loc[(new_ind) & (data['Real']==0),'corrected_prediction']=data.loc[(new_ind) & (data['Real']==0),'corrected_prediction']+1
    data.loc[new_ind,'Absolute_error']=abs(data.loc[new_ind,'corrected_real']-data.loc[new_ind,'corrected_prediction'])
    data.loc[new_ind,'Country MAPE (test)']=(data.loc[new_ind,'Absolute_error']/data.loc[new_ind,'corrected_real'])*100
    data=data.drop(['corrected_real','corrected_prediction','Absolute_error','Naive MAE','naive_prediction'],axis=1)

    data = data.drop(['prediction datetime','datetime','death','date'],axis=1)
    return(data)

#################################################### return data to original mode

def make_original_data(data,target_mode):
    # if target mode is logarithmic we need to return the target variable to its original state
    if target_mode == 'logarithmic':
        data['Target'] = list(np.round(np.exp(data['Target']) - 1))
        data['Prediction'] = list(np.round(np.exp(data['Prediction']) - 1))
    return(data)

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

##################################################################################################### main 
if __name__ == "__main__":

    for r in r_list:
        
        print ("r = "+str(r))
        
        if r>14:
            r //= 7
            test_size //= 7
            temporal_mode="weekly"
            target_mode='weeklyaverage'
            future_mode = True
            force_mode = 0  # with force_mode we determine how many future feature have to be forced (be used in all iterations)
            for f in range(force_mode):
                force_features.append('future-' + future_features[f])
        else:
            temporal_mode="daily"
            target_mode='logarithmic'
        
        h = best_h[temporal_mode][r]
        print('h = '+str(h))
            
            
        ############################## reading data
        
        # we use zero_removed data for training the models 
        # and then predict target for all counties (set zero_remove argument to 0)
        
        data = makeHistoricalData(h, r, test_size, target_name, 'mrmr', spatial_mode, target_mode, './',
                                      future_features, pivot, None, zero_removing)
        data = clean_data(data, numberOfSelectedCounties, spatial_mode)
        
        all_counties_data = makeHistoricalData(h, r, test_size, target_name, 'mrmr', spatial_mode, target_mode, './',
                                      future_features, pivot, None, 0)
        all_counties_data = clean_data(all_counties_data, numberOfSelectedCounties, spatial_mode)

        if target_mode not in ['regular',
                              'weeklyaverage']:  # we need regular data to return predicted values to first state
            regular_data = makeHistoricalData(h, r, test_size, target_name, 'mrmr', spatial_mode, 'regular', './',
                                              future_features, pivot, None, zero_removing)
            regular_data = clean_data(regular_data, numberOfSelectedCounties, spatial_mode)
        else:
            regular_data = data
        
        ####################### finding errors
        
        X_train_train, X_train_val, X_test, y_train_train_date, y_train_val_date, y_test_date = preprocess(data,
                                                                                                          spatial_mode,
                                                                                                          1)
        candid_features = list(X_train_train.columns)
        best_features = candid_features.copy()
        futured_features = ["{}{}".format('future-',i) for i in future_features]
        for feature in candid_features:
            if feature.split(' ')[0] not in selected_covariates[temporal_mode][r]:
                print(feature)
                best_features.remove(feature)
        print(best_features)
        best_features += ['county_fips',
                            'date of day t']
        best_features += force_features
        best_features = np.unique(best_features)
        
        X_train_train = X_train_train[best_features]
        X_train_val = X_train_val[best_features]
        print('X_train_val.columns',X_train_val.columns)
        

            
                            
        y_prediction, y_prediction_train = run_algorithms(X_train_train, X_train_val, y_train_train_date, y_train_val_date,
                                                          best_loss, best_methods[temporal_mode][r] , 'val')
        county_errors,country_errors,best_scaler = get_errors(y_prediction, y_prediction_train, y_train_val_date, y_train_train_date,
                                                          regular_data, numberOfSelectedCounties)
        
        
        ####################### prediction future

        
        X_train, temp_1 , y_train_date, temp_2 = preprocess(data,spatial_mode,0)
        
        X_X_train, X_test , y_y_train_date, y_test_date = preprocess(all_counties_data,spatial_mode,0)
        
        X_test = X_X_train.append(X_test)
        y_test_date = y_y_train_date.append(y_test_date)
        
        print(y_test_date[['Target']].tail(10), 'line 444')
        candid_features = list(X_train_train.columns)
        best_features = candid_features.copy()
        futured_features = ["{}{}".format('future-',i) for i in future_features]
        for feature in candid_features:
            if feature.split(' ')[0] not in selected_covariates[temporal_mode][r]:
                best_features.remove(feature)
        best_features += ['county_fips',
                            'date of day t'] 
        best_features += force_features
        best_features = np.unique(best_features)
        
        print(y_train_date.columns, 'line 449')
        
        X_train = X_train[best_features]
        X_test = X_test[best_features]
        
        
        y_prediction, y_prediction_train = run_algorithms(X_train, X_test, y_train_date, y_test_date,
                                                          best_loss, best_methods[temporal_mode][r], 'test')
        


        y_test_date['Prediction'] = list(y_prediction)
        y_test_date = make_original_data(y_test_date,target_mode)
        
        print('y_prediction',y_test_date['Prediction'])
        y_test_date['Prediction'] = best_scaler*y_test_date['Prediction']
        print('y_prediction',y_test_date['Prediction'])

        if temporal_mode == 'daily':
          y_train_date['Date'] = y_train_date['date of day t'].apply(
                      lambda x: datetime.datetime.strptime(x, '%m/%d/%y') + datetime.timedelta(days=r))
          y_test_date['Date'] = y_test_date['date of day t'].apply(
                      lambda x: datetime.datetime.strptime(x, '%m/%d/%y') + datetime.timedelta(days=r))
        elif temporal_mode == 'weekly':
          y_train_date['Date'] = y_train_date['date of day t'].apply(lambda x: x+r)
          y_test_date['Date'] = y_test_date['date of day t'].apply(lambda x: x+r)

        print(y_test_date[['Target']].tail(10), 'line 480')


        ####################################### get country prediction

        y_test_date = y_test_date.groupby(['Date']).sum()
        print(y_test_date[['Target']].tail(10), 'line 495')
        print(y_test_date[['Target']].head(10), 'line 495')
        y_test_date = y_test_date.reset_index()
        print(y_test_date[['Target']].tail(10), 'line 497')
        print(y_test_date[['Target']].head(10), 'line 497')
        y_test_date['Model'] = best_methods[temporal_mode][r]
        y_test_date['Country MASE (test)'] = country_errors['MASE']
        y_test_date['Country MAPE (test)'] = country_errors['MAPE']
        y_test_date['Country MAE (test)'] = country_errors['MAE']
        y_test_date['Country RMSE (test)'] = country_errors['RMSE']
        y_test_date['County MASE (test)'] = county_errors['MASE']
        y_test_date['County MAPE (test)'] = county_errors['MAPE']
        y_test_date['County MAE (test)'] = county_errors['MAE']
        y_test_date['County RMSE (test)'] = county_errors['RMSE']
        y_test_date['Prediction Date'] = current_date
        y_test_date.rename(columns={'Target':'Real'},inplace=True)
        y_test_date = y_test_date[['Prediction Date','Date', 'Real', 'Prediction', 'Model', 'Country MASE (test)',\
                                  'Country MAPE (test)', 'Country MAE (test)', 'Country RMSE (test)', 'County MASE (test)',\
                                  'County MAPE (test)', 'County MAE (test)', 'County RMSE (test)']]
        print('y_test_date.columns',y_test_date.columns)
        
        observed = y_test_date.iloc[:-r,:]
        unobserved = y_test_date.tail(r)
        unobserved['Real']=np.NaN 

        if temporal_mode=="daily" :
            observed = date_nominalize(observed,temporal_mode,'Date')
            unobserved = date_nominalize(unobserved,temporal_mode,'Date')

        elif temporal_mode=="weekly" :
            unobserved = date_nominalize(unobserved,temporal_mode,'Date')

        print(y_test_date[['Real']].tail(10), 'line 513')
            
        
        if first_run==1 and temporal_mode=="daily" and r==14 :
            output = pd.concat([observed,output], ignore_index=True)
            output = pd.concat([output,unobserved], ignore_index=True)
            
        elif temporal_mode=="daily" :
            output = pd.concat([output,unobserved.tail(7)], ignore_index=True)
            
        elif temporal_mode=="weekly" :
            output = pd.concat([output,unobserved.tail(2)], ignore_index=True)

    output['Prediction Date']=output['Prediction Date'].apply(lambda x: datetime.datetime.strftime(x,'%Y-%m-%d'))
    if pivot == 'country':
      for col in ['County MAE (test)','County MASE (test)','County MAPE (test)','County RMSE (test)']:
        output[col]=np.NaN

    if first_run==0: 
        output_excel = pd.read_excel("Deaths-Prediction.xlsx")
        output_excel = output_excel.append(output)
        # add real values of previous days
        output_excel = add_real_values(output_excel,address)
    else:
        output_excel = output
        output_excel = add_real_values(output_excel,address)#######################

    # remove duplaicates
    for col in ['Model', 'Country MASE (test)', 'Country MAPE (test)', 'Country MAE (test)',
          'Country RMSE (test)', 'County MASE (test)', 'County MAPE (test)',
          'County MAE (test)', 'County RMSE (test)']:
          output_excel.loc[output_excel[col].duplicated(), col]=np.nan


    output_excel.to_excel("Deaths-Prediction.xlsx")
        
    push('Predictions updated')
        
    
    