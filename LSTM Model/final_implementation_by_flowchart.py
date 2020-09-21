import pandas as pd
import numpy as np
from makeHistoricalData import makeHistoricalData
import multiprocessing as mp
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error
from pathlib import Path
import smtplib 
from email.mime.multipart import MIMEMultipart 
from email.mime.text import MIMEText 
from email.mime.base import MIMEBase 
from email import encoders 
import itertools
import shelve
from zipfile import ZipFile
import time
import sys


# mkdir for saving the results in
Path("Results").mkdir(parents=True, exist_ok=True)

r = 14
H = 21
test_size = 21
numberOfSelectedCounties = -1
fixed_data = pd.read_csv('data/fixed-data.csv')
temporal_data = pd.read_csv('data/temporal-data.csv')

number_of_fixed_covariates = nf = len(fixed_data.columns)
number_of_temporal_covariates = nt = len(temporal_data.columns)

# creating some dataframes to store the results in
moving_average_results_all_country_df = pd.DataFrame(columns=['h', 'c', 'MAE', 'MAPE', 'MASE'])
moving_average_results_per_state_df = pd.DataFrame(columns=['h', 'c', 'state_fips', 'MAE', 'MAPE', 'MASE'])
moving_average_results_per_county_df = pd.DataFrame(columns=['h', 'c', 'county_fips', 'MAE', 'MAPE', 'MASE'])

regular_data_list = [makeHistoricalData(fixed_data, temporal_data, h, r, test_size, 'death', 'mrmr', 'country', 'regular', []) for h in range(1, H+1)]
moving_average_data = []

comparison_criteria = 'MASE'

default_model_type = 14

types = [
	[8, 16, 16], 
	[8, 16, 32], 
	[8, 32, 32], 
	[8, 32, 64], 
	[8, 64, 64], 
	[8, 16, 32, 16], 
	[8, 16, 32, 32], 
	[8, 32, 64, 32], 
	[8, 32, 128, 64],
	[8, 64, 128, 32], 
	[8, 16, 128, 128, 64], 
	[8, 16, 128, 256, 64], 
	[8, 32, 256, 256, 128], 
	[8, 64, 256, 128, 64], 
	[8, 256, 256, 128, 128], # default_model_type
	[8, 16, 32, 128, 64, 32], 
	[8, 16, 64, 256, 128, 64], 
	[8, 32, 128, 256, 128, 64], 
	[8, 64, 128, 256, 256, 128], 
	[8, 64, 256, 256, 128, 128]
]


def mean_absolute_percentage_error(y_true, y_pred):
	y_true, y_pred = np.array(y_true), np.array(y_pred)
	sumOfAbsoluteError = sum(abs(y_true - y_pred))
	mape = (sumOfAbsoluteError / sum(y_true)) * 100
	return mape


def MASE(y_true, y_pred, y_naive):
	mae_on_pred = mean_absolute_error(y_true, y_pred)
	mae_on_naive = mean_absolute_error(y_true, y_naive)

	return mae_on_pred/mae_on_naive


######################################################### split data to train, val, test
def splitData(numberOfCounties, main_data, target, spatial_mode, mode, future_mode):
	numberOfCounties = len(main_data['county_fips'].unique())
	main_data = main_data.sort_values(by=['date of day t', 'county_fips'])
	target = target.sort_values(by=['date of day t', 'county_fips'])
	# we set the base number of days to the minimum number of days existed between the counties
	# and then compute the validation size for the non-default state.
	baseNumberOfDays = (main_data.groupby(['county_fips']).size()).min()
	val_size = round(0.3 * (baseNumberOfDays - test_size))

	if mode == 'val':
		if not future_mode:  # the default state
			X_train_train = main_data.iloc[:-2 * (r * numberOfCounties), :].sort_values( by=['county_fips', 'date of day t'])
			X_train_val = main_data.iloc[-2 * (r * numberOfCounties):-(r * numberOfCounties), :].sort_values(by=['county_fips', 'date of day t'])
			X_test = main_data.tail(r * numberOfCounties).sort_values(by=['county_fips', 'date of day t'])

			y_train_train = target.iloc[:-2 * (r * numberOfCounties), :].sort_values(by=['county_fips', 'date of day t'])
			y_train_val = target.iloc[-2 * (r * numberOfCounties):-(r * numberOfCounties), :].sort_values(by=['county_fips', 'date of day t'])
			y_test = target.tail(r * numberOfCounties).sort_values(by=['county_fips', 'date of day t'])

			val_naive_pred = target.iloc[-3*(r*numberOfCounties):-2*(r*numberOfCounties),:].sort_values(by=['county_fips', 'date of day t'])
			test_naive_pred = target.iloc[-2*(r*numberOfCounties):-(r*numberOfCounties),:].sort_values(by=['county_fips', 'date of day t'])
			
		else:
			X_test = main_data.tail(test_size * numberOfCounties).copy()
			X_train_val = main_data.iloc[:-(test_size * numberOfCounties)].tail(val_size * numberOfCounties).copy()
			X_train_train = main_data.iloc[:-((val_size + test_size) * numberOfCounties)].copy()

			y_test = target.tail(test_size * numberOfCounties).copy()
			y_train_val = target.iloc[:-(test_size * numberOfCounties)].tail(val_size * numberOfCounties).copy()
			y_train_train = target.iloc[:-((val_size + test_size) * numberOfCounties)].copy()

		return X_train_train, X_train_val, X_test, y_train_train, y_train_val, y_test, val_naive_pred, test_naive_pred

	if mode == 'test':
		if not future_mode:
			X_train = main_data.iloc[:-(r * numberOfCounties), :].sort_values(by=['county_fips', 'date of day t'])
			X_test = main_data.tail(r * numberOfCounties).sort_values(by=['county_fips', 'date of day t'])

			y_train = target.iloc[:-(r * numberOfCounties), :].sort_values(by=['county_fips', 'date of day t'])
			y_test = target.tail(r * numberOfCounties).sort_values(by=['county_fips', 'date of day t'])

			test_naive_pred = target.iloc[-2*(r*numberOfCounties):-(r*numberOfCounties),:].sort_values(by=['county_fips', 'date of day t'])

		else:
			X_test = main_data.tail(test_size * numberOfCounties).copy()
			X_train = main_data.iloc[:-(test_size * numberOfCounties)].copy()

			y_test = target.tail(test_size * numberOfCounties).copy()
			y_train = target.iloc[:-(test_size * numberOfCounties)]

		return X_train, X_test, y_train, y_test, test_naive_pred


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
		main_data = using_data.drop(['county_name', 'state_fips', 'state_name'], axis=1)
	elif (spatial_mode == 'state'):
		main_data = using_data.drop(['county_name', 'state_name'], axis=1)
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
		X_train, X_val, X_test, y_train, y_val, y_test, val_naive_pred, test_naive_pred = \
			splitData(numberOfSelectedCounties, main_data, target, spatial_mode, 'val', False)
		
		y_train_date = y_train.copy()
		y_test_date = y_test.copy()
		y_val_date = y_val.copy()
  
		X_train.drop(['date of day t', 'county_fips'], axis=1, inplace=True)
		X_val.drop(['date of day t', 'county_fips'], axis=1, inplace=True)
		X_test.drop(['date of day t', 'county_fips'], axis=1, inplace=True)

		y_train.drop(['date of day t', 'county_fips'], axis=1, inplace=True)
		y_val.drop(['date of day t', 'county_fips'], axis=1, inplace=True)
		y_test.drop(['date of day t', 'county_fips'], axis=1, inplace=True)

		val_naive_pred.drop(['date of day t', 'county_fips'], axis=1, inplace=True)
		test_naive_pred.drop(['date of day t', 'county_fips'], axis=1, inplace=True)

		val_naive_pred = val_naive_pred.values.flatten()
		test_naive_pred = test_naive_pred.values.flatten()
		
		return X_train, X_val, X_test, y_train, y_val, y_test, y_train_date, y_test_date, y_val_date, val_naive_pred, test_naive_pred

	else:
		X_train, X_test, y_train, y_test, test_naive_pred = splitData(numberOfSelectedCounties, main_data, target, spatial_mode, 'test', False)

		y_train_date = y_train.copy()
		y_test_date = y_test.copy()

		return X_train, X_test, y_train, y_test, y_train_date, y_test_date, test_naive_pred


########################################################### convert predictions to regular mode
def make_org_data(h, y_prediction, y_test_date, y_train_date, regular_data, numberOfSelectedCounties, target_mode, mode):
	# y_test_date and y_train_date are a dataframes with columns ['date of day t', 'county_fips', 'Target']
	# set negative predictions to zero
	if target_mode == 'regular':
		y_prediction[y_prediction < 0] = 0
		y_pred = np.round(y_prediction.astype(np.double))
		y_true = np.array(y_test_date['Target']).reshape(-1)
		return y_true, y_pred.reshape(-1)
	
	y_prediction[y_prediction < 0] = 0

	# next 8 lines sort y_prediction and y_prediction_train like output of preprocess function
	# we need to sort predictions because in county and state mode their order may be cluttered
	# y_train_date['prediction'] = y_prediction_train
	y_train_date = y_train_date.sort_values(by=['county_fips','date of day t'])
	# y_prediction_train = list(y_train_date['prediction'])
	# y_train_date= y_train_date.drop(['prediction'],axis=1)

	y_test_date['prediction'] = y_prediction
	y_test_date = y_test_date.sort_values(by=['county_fips','date of day t'])
	y_prediction = list(y_test_date['prediction'])
	y_test_date = y_test_date.drop(['prediction'],axis=1)
	y_test = np.array(y_test_date['Target']).reshape(-1)
	
	if numberOfSelectedCounties == -1 :
		  numberOfSelectedCounties = len(y_test_date['county_fips'].unique())
	
	# we need data with regular target to return modified target to its original state
	# in validation mode we read regular data in main function and passed to get_error to avoid redundancy 
	# but in test mode its not possible because each method has different h(best_h)
	if mode == 'test':
		regular_data = makeHistoricalData(fixed_data, temporal_data, h, r, test_size, 'death', 'mrmr', 'country', 'regular', [])
		regular_data = clean_data(regular_data, numberOfSelectedCounties, 'country')
		_, _, _, _, regular_y_train_date, regular_y_test_date , _ = preprocess(regular_data, 'country', 0)
 
	if mode == 'val':
		_, _, _, _, _, _, regular_y_train_date, _, regular_y_test_date, _, _ = preprocess(regular_data, 'country', 1)
		
			
	# if target mode is cumulative we need to return the target variable to its original state
	if target_mode == 'cumulative':
		cumulative_data = y_train_date.append(y_test_date)
		cumulative_data['prediction'] = list(y_train_date['Target'])+list(y_prediction)
		cumulative_data = cumulative_data.sort_values(by=['date of day t','county_fips'])
		reverse_dates = cumulative_data['date of day t'].unique()[-(r+1):][::-1]
		for index in range(len(reverse_dates)):
			date=reverse_dates[index]
			past_date=reverse_dates[index+1]
			cumulative_data.loc[cumulative_data['date of day t']==date,'Target']=list(np.array(cumulative_data.loc[cumulative_data['date of day t']==date,'Target'])-np.array(cumulative_data.loc[cumulative_data['date of day t']==past_date,'Target']))
			cumulative_data.loc[cumulative_data['date of day t']==date,'prediction']=list(np.array(cumulative_data.loc[cumulative_data['date of day t']==date,'prediction'])-np.array(cumulative_data.loc[cumulative_data['date of day t']==past_date,'prediction']))
			if index == len(reverse_dates)-2:
				break

		y_test_date = cumulative_data.tail(r*numberOfSelectedCounties)
		y_test_date = y_test_date.sort_values(by=['county_fips','date of day t'])
		y_test = np.array(y_test_date['Target']).reshape(-1)
		y_prediction = np.array(y_test_date['prediction']).reshape(-1)

		# cumulative_data = cumulative_data.sort_values(by=['date of day t','county_fips'])
		# y_test_date = cumulative_data.tail(r*numberOfSelectedCounties)
		# y_test = np.array(y_test_date['Target']).reshape(-1)
		# y_prediction = np.array(cumulative_data.tail(r*numberOfSelectedCounties)['prediction']).reshape(-1)		

	# if target mode is moving average we need to return the target variable to its original state
	if target_mode == 'weeklymovingaverage':
		
		# past values of targets that will be use for return the weeklymovingaverage target (predicted)
		# to original state to calculate errors
		regular_real_predicted_target = regular_y_train_date.append(regular_y_test_date) #dataframe with columns ['date of day t', 'county_fips', 'Target']
		regular_real_predicted_target['prediction'] = list(regular_y_train_date['Target'])+list(y_prediction)
		
		regular_real_predicted_target = regular_real_predicted_target.sort_values(by=['date of day t','county_fips'])
		regular_real_predicted_target=regular_real_predicted_target.tail((r+6)*numberOfSelectedCounties)
		
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
				
		y_test_date = regular_real_predicted_target.tail(r*numberOfSelectedCounties)
		y_test_date = y_test_date.sort_values(by=['county_fips','date of day t'])
		y_test = np.array(y_test_date['Target']).reshape(-1)
		y_prediction = np.array(y_test_date['prediction']).reshape(-1)
	
	# make predictions rounded to their closest number
	y_prediction = np.array(y_prediction)
	if target_mode != 'weeklyaverage':
		y_prediction = np.round(y_prediction)
	
	return y_test, y_prediction


def data_normalize(X_train, y_train, X_val, y_val, X_test, y_test):
	scaler = preprocessing.StandardScaler()

	X_train = X_train.values
	X_train = scaler.fit_transform(X_train)

	X_val = X_val.values
	X_val = scaler.fit_transform(X_val)

	X_test = X_test.values
	X_test = scaler.fit_transform(X_test)

	y_train = y_train.values
	y_train = scaler.fit_transform(y_train.reshape(-1, 1))

	y_val = y_val.values
	y_val = scaler.fit_transform(y_val.reshape(-1, 1))

	y_test = y_test.values
	# y_test = min_max_scaler.fit_transform(y_test.reshape(-1, 1))

	X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
	X_val = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))
	X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

	y_train = y_train.reshape((y_train.shape[0]), )
	y_val = y_val.reshape((y_val.shape[0]), )
	y_test = y_test.reshape((y_test.shape[0]), )

	return X_train, y_train, X_val, y_val, X_test, y_test, scaler


def part_of_data(X_train, X_val, X_test, y_train, y_val, y_test, c):
	train_X = X_train.iloc[:, 0:c].copy()
	train_y = y_train.copy()

	val_X = X_val.iloc[:, 0:c].copy()
	val_y = y_val.copy()

	test_X = X_test.iloc[:, 0:c].copy()
	test_y = y_test.copy()

	return train_X, val_X, test_X, train_y, val_y, test_y


def default_model(n):
	model = Sequential()
	model.add(LSTM(8, kernel_initializer=tf.keras.initializers.Identity(), return_sequences=True, input_shape=(1, n)))
	model.add(LeakyReLU(alpha=0.3))
	model.add(LSTM(256, dropout=0.2, return_sequences=True))
	model.add(LeakyReLU(alpha=0.3))
	model.add(LSTM(256, dropout=0.2, return_sequences=True))
	model.add(LeakyReLU(alpha=0.3))
	model.add(LSTM(128, dropout=0.2, return_sequences=True))
	model.add(LeakyReLU(alpha=0.3))
	model.add(LSTM(128))
	model.add(LeakyReLU(alpha=0.3))
	model.add(Dense(1))
	model.add(LeakyReLU(alpha=0.3))

	model.compile (
		loss=tf.keras.losses.MeanAbsoluteError(),
		optimizer=keras.optimizers.Adam(0.0001),
		metrics=[tf.keras.metrics.MeanAbsoluteError()]
	)

	return model


def fit_model_all_country(dataset, h, c):

	global numberOfSelectedCounties

	results = []

	numberOfSelectedCounties = len(dataset['county_fips'].unique())
	new_dataset = clean_data(dataset, numberOfSelectedCounties, 'country')
	X_train, X_val, X_test, y_train, y_val, y_test, y_train_date, _, y_val_date, _, _ = preprocess(new_dataset, 'country', 1)

	print('*************************************')
	print('c = ' + str(c) + ', h = ' + str(h))

	train_X, val_X, test_X, train_y, val_y, test_y = part_of_data(X_train, X_val, X_test, y_train, y_val, y_test, c)
	train_X, train_y, val_X, val_y, test_X, test_y, scaler = data_normalize(train_X, train_y, val_X, val_y, test_X, test_y)

	model = default_model(train_X.shape[2])

	early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1, patience=3)
	
	model.fit (
		train_X, train_y,
		epochs=500,
		batch_size=128,
		validation_split=0.2,
		verbose=1,
		callbacks=[early_stop], 
		shuffle=False
	)

	y_pred = model.predict(val_X)
	y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))
	
	regular_data = regular_data_list[h]
	numberOfSelectedCounties = len(regular_data['county_fips'].unique())
	new_dataset_2 = clean_data(regular_data, numberOfSelectedCounties, 'country')
	_, _, _, _, _, _, _, _, _, val_naive_pred_2, _ = preprocess(new_dataset_2, 'country', 1)

	y_true, y_pred = make_org_data(h, y_pred, y_val_date, y_train_date, regular_data, -1, 'weeklymovingaverage', 'val')
	y_pred[y_pred <= 0] = 0

	mase = MASE(y_true, y_pred, val_naive_pred_2)
	mae = mean_absolute_error(y_true, y_pred)
	mape = mean_absolute_percentage_error(y_true, y_pred)

	result_dict = {"h": h, "c": c, "MAE": mae, "MAPE": mape, "MASE": mase}
	results.append(result_dict.copy())

	K.clear_session()

	return results


def fit_model_per_state(dataset, h, c):

	global numberOfSelectedCounties

	results = []

	dataset.reset_index(drop=True, inplace=True)
	states = dataset.state_fips.unique()

	val_naive_pred_array = np.array([])
	val_model_pred_array = np.array([])
	val_true_pred_array = np.array([])

	for state in states:
		state_dataset = dataset.loc[dataset['state_fips'] == state]
		numberOfSelectedCounties = len(state_dataset['county_fips'].unique())
		new_dataset = clean_data(state_dataset, numberOfSelectedCounties, 'country')
		X_train, X_val, X_test, y_train, y_val, y_test, y_train_date, _, y_val_date, _, _ = preprocess(new_dataset, 'country', 1)

		print('*************************************')
		print('c = ' + str(c) + ', h = ' + str(h) + ', state_fips = ' + str(state))

		train_X, val_X, test_X, train_y, val_y, test_y = part_of_data(X_train, X_val, X_test, y_train, y_val, y_test, c)
		train_X, train_y, val_X, val_y, test_X, test_y, scaler = data_normalize(train_X, train_y, val_X, val_y, test_X, test_y)

		model = default_model(train_X.shape[2])

		early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1, patience=3)
	
		model.fit (
			train_X, train_y,
			epochs=500,
			batch_size=128,
			validation_split=0.2,
			verbose=1,
			callbacks=[early_stop], 
			shuffle=False
		)

		y_pred = model.predict(val_X)
		y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))

		regular_data = regular_data_list[h]
		regular_data = regular_data.loc[regular_data['state_fips'] == state]
		numberOfSelectedCounties = len(regular_data['county_fips'].unique())
		new_dataset_2 = clean_data(regular_data, numberOfSelectedCounties, 'country')
		_, _, _, _, _, _, _, _, _, val_naive_pred_2, _ = preprocess(new_dataset_2, 'country', 1)

		y_true, y_pred = make_org_data(h, y_pred, y_val_date, y_train_date, regular_data, -1, 'weeklymovingaverage', 'val')
		y_pred[y_pred <= 0] = 0

		val_true_pred_array = np.concatenate((val_true_pred_array, y_true))
		val_naive_pred_array = np.concatenate((val_naive_pred_array, val_naive_pred_2))
		val_model_pred_array = np.concatenate((val_model_pred_array, y_pred))

		K.clear_session()
		
	mase = MASE(val_true_pred_array, val_model_pred_array, val_naive_pred_array)
	mae = mean_absolute_error(val_true_pred_array, val_model_pred_array)
	mape = mean_absolute_percentage_error(val_true_pred_array, val_model_pred_array)

	result_dict = {"h": h, "c": c, "MAE": mae, "MAPE": mape, "MASE": mase}
	results.append(result_dict.copy())

	return results


def fit_model_per_county(dataset, h, c):

	global numberOfSelectedCounties

	results = []

	dataset.reset_index(drop=True, inplace=True)
	counties = dataset.county_fips.unique()

	val_naive_pred_array = np.array([])
	val_model_pred_array = np.array([])
	val_true_pred_array = np.array([])

	for county in counties:
		county_dataset = dataset.loc[dataset['county_fips'] == county]
		numberOfSelectedCounties = len(county_dataset['county_fips'].unique())
		new_dataset = clean_data(county_dataset, numberOfSelectedCounties, 'country')
		X_train, X_val, X_test, y_train, y_val, y_test, y_train_date, _, y_val_date, _, _ = preprocess(new_dataset, 'country', 1)

		print('*************************************')
		print('c = ' + str(c) + ', h = ' + str(h) + ', county_fips = ' + str(county))

		train_X, val_X, test_X, train_y, val_y, test_y = part_of_data(X_train, X_val, X_test, y_train, y_val, y_test, c)
		train_X, train_y, val_X, val_y, test_X, test_y, scaler = data_normalize(train_X, train_y, val_X, val_y, test_X, test_y)

		model = default_model(train_X.shape[2])

		early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1, patience=3)
	
		model.fit (
			train_X, train_y,
			epochs=500,
			batch_size=128,
			validation_split=0.2,
			verbose=1,
			callbacks=[early_stop], 
			shuffle=False
		)

		y_pred = model.predict(val_X)
		y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))

		regular_data = regular_data_list[h]
		regular_data = regular_data.loc[regular_data['county_fips'] == county]
		numberOfSelectedCounties = len(regular_data['county_fips'].unique())
		new_dataset_2 = clean_data(regular_data, numberOfSelectedCounties, 'country')
		_, _, _, _, _, _, _, _, _, val_naive_pred_2, _ = preprocess(new_dataset_2, 'country', 1)

		y_true, y_pred = make_org_data(h, y_pred, y_val_date, y_train_date, regular_data, -1, 'weeklymovingaverage', 'val')
		y_pred[y_pred <= 0] = 0

		val_true_pred_array = np.concatenate((val_true_pred_array, y_true))
		val_naive_pred_array = np.concatenate((val_naive_pred_array, val_naive_pred_2))
		val_model_pred_array = np.concatenate((val_model_pred_array, y_pred))

		K.clear_session()

	mase = MASE(val_true_pred_array, val_model_pred_array, val_naive_pred_array)
	mae = mean_absolute_error(val_true_pred_array, val_model_pred_array)
	mape = mean_absolute_percentage_error(val_true_pred_array, val_model_pred_array)

	result_dict = {"h": h, "c": c, "MAE": mae, "MAPE": mape, "MASE": mase}
	results.append(result_dict.copy())

	return results


def get_best_model(results_dataframe):	# each element is a tuple of (h, c, R2_SCORE)
	best_score, best_h, best_c = 0, 0, 0

	results_dataframe[comparison_criteria] = pd.to_numeric(results_dataframe[comparison_criteria])
	index_of_min_score = results_dataframe[comparison_criteria].idxmin()

	best_h = results_dataframe.iloc[index_of_min_score]['h']
	best_c = results_dataframe.iloc[index_of_min_score]['c']
	best_score = results_dataframe.iloc[index_of_min_score][comparison_criteria]

	return tuple((best_h, best_c, best_score))


def choose_winner_model(moving_average_results_all_country_df, moving_average_results_per_state_df, moving_average_results_per_county_df):

	best_h_all_country, best_c_all_country, best_score_all_country = get_best_model(moving_average_results_all_country_df)
	best_h_per_state, best_c_per_state, best_score_per_state = get_best_model(moving_average_results_per_state_df)
	best_h_per_county, best_c_per_county, best_score_per_county = get_best_model(moving_average_results_per_county_df)

	temp_list = [best_score_all_country, best_score_per_state, best_score_per_county]

	best = temp_list.index(min(temp_list))

	if best == 0:
		return tuple((int(best_h_all_country), int(best_c_all_country), 0))	# 0 shows winner model should be implemented in all country mode
	elif best == 1:
		return tuple((int(best_h_per_state), int(best_c_per_state), 1))	# 1 shows winner model should be implemented in per state mode
	elif best == 2:
		return tuple((int(best_h_per_county), int(best_c_per_county), 2))	# 2 shows winner model should be implemented in per county mode


def lstm_model_all_country(dataset, h, c, dropout_value, model_type, method):	# this function works with specific values for h and c

	global numberOfSelectedCounties

	dataset.reset_index(drop=True, inplace=True)
	numberOfSelectedCounties = len(dataset['county_fips'].unique())
	new_dataset = clean_data(dataset, numberOfSelectedCounties, 'country')
	X_train, X_val, X_test, y_train, y_val, y_test, y_train_date, _, y_val_date, _, _ = preprocess(new_dataset, 'country', 1)

	train_X, val_X, test_X, train_y, val_y, test_y = part_of_data(X_train, X_val, X_test, y_train, y_val, y_test, c)
	train_X, train_y, val_X, val_y, test_X, test_y, scaler = data_normalize(train_X, train_y, val_X, val_y, test_X, test_y)

	model = Sequential()
	model.add(LSTM(types[model_type][0], kernel_initializer=tf.keras.initializers.Identity(), return_sequences=True, input_shape=(1, train_X.shape[2])))
	model.add(LeakyReLU(alpha=0.3))
	for i in types[model_type][1:len(types[model_type])-1]:
		model.add(LSTM(i, dropout=dropout_value, return_sequences=True))
		model.add(LeakyReLU(alpha=0.3))
	model.add(LSTM(types[model_type][-1]))
	model.add(LeakyReLU(alpha=0.3))
	model.add(Dense(1))
	model.add(LeakyReLU(alpha=0.3))

	model.compile (
		loss=tf.keras.losses.MeanAbsoluteError(),
		optimizer=keras.optimizers.Adam(0.0001),
		metrics=[tf.keras.metrics.MeanAbsoluteError()]
	)

	early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1, patience=3)
	
	model.fit (
		train_X, train_y,
		epochs=500,
		batch_size=128,
		validation_split=0.2,
		verbose=1,
		callbacks=[early_stop], 
		shuffle=False
	)

	y_pred = model.predict(val_X)
	y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))
	
	regular_data = regular_data_list[h]
	numberOfSelectedCounties = len(regular_data['county_fips'].unique())
	new_dataset_2 = clean_data(regular_data, numberOfSelectedCounties, 'country')
	_, _, _, _, _, _, _, _, _, val_naive_pred_2, _ = preprocess(new_dataset_2, 'country', 1)

	y_true, y_pred = make_org_data(h, y_pred, y_val_date, y_train_date, regular_data, -1, method, 'val')
	y_pred[y_pred <= 0] = 0
	
	mase = MASE(y_true, y_pred, val_naive_pred_2)
	# mae = mean_absolute_error(y_true, y_pred)
	# mape = mean_absolute_percentage_error(y_true, y_pred)

	K.clear_session()

	return mase


def lstm_model_per_state(dataset, h, c, dropout_value, model_type, method):
	
	global numberOfSelectedCounties

	dataset.reset_index(drop=True, inplace=True)
	states = dataset.state_fips.unique()

	val_naive_pred_array = np.array([])
	val_model_pred_array = np.array([])
	val_true_pred_array = np.array([])

	for state in states:
		state_dataset = dataset.loc[dataset['state_fips'] == state]
		numberOfSelectedCounties = len(state_dataset['county_fips'].unique())
		new_dataset = clean_data(state_dataset, numberOfSelectedCounties, 'country')
		X_train, X_val, X_test, y_train, y_val, y_test, y_train_date, _, y_val_date, _, _ = preprocess(new_dataset, 'country', 1)

		train_X, val_X, test_X, train_y, val_y, test_y = part_of_data(X_train, X_val, X_test, y_train, y_val, y_test, c)
		train_X, train_y, val_X, val_y, test_X, test_y, scaler = data_normalize(train_X, train_y, val_X, val_y, test_X, test_y)

		model = Sequential()
		model.add(LSTM(types[model_type][0], kernel_initializer=tf.keras.initializers.Identity(), return_sequences=True, input_shape=(1, train_X.shape[2])))
		model.add(LeakyReLU(alpha=0.3))
		for i in types[model_type][1:len(types[model_type])-1]:
			model.add(LSTM(i, dropout=dropout_value, return_sequences=True))
			model.add(LeakyReLU(alpha=0.3))
		model.add(LSTM(types[model_type][-1]))
		model.add(LeakyReLU(alpha=0.3))
		model.add(Dense(1))
		model.add(LeakyReLU(alpha=0.3))

		model.compile (
			loss=tf.keras.losses.MeanAbsoluteError(),
			optimizer=keras.optimizers.Adam(0.0001),
			metrics=[tf.keras.metrics.MeanAbsoluteError()]
		)

		early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1, patience=3)
	
		model.fit (
			train_X, train_y,
			epochs=500,
			batch_size=128,
			validation_split=0.2,
			verbose=1,
			callbacks=[early_stop], 
			shuffle=False
		)

		y_pred = model.predict(val_X)
		y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))

		regular_data = regular_data_list[h]
		regular_data = regular_data.loc[regular_data['state_fips'] == state]
		numberOfSelectedCounties = len(regular_data['county_fips'].unique())
		new_dataset_2 = clean_data(regular_data, numberOfSelectedCounties, 'country')
		_, _, _, _, _, _, _, _, _, val_naive_pred_2, _ = preprocess(new_dataset_2, 'country', 1)

		y_true, y_pred = make_org_data(h, y_pred, y_val_date, y_train_date, regular_data, -1, method, 'val')
		y_pred[y_pred <= 0] = 0

		val_true_pred_array = np.concatenate((val_true_pred_array, y_true))
		val_naive_pred_array = np.concatenate((val_naive_pred_array, val_naive_pred_2))
		val_model_pred_array = np.concatenate((val_model_pred_array, y_pred))

		K.clear_session()

	mase = MASE(val_true_pred_array, val_model_pred_array, val_naive_pred_array)
	# mae = mean_absolute_error(val_true_pred_array, val_model_pred_array)
	# mape = mean_absolute_percentage_error(val_true_pred_array, val_model_pred_array)

	return mase


def lstm_model_per_county(dataset, h, c, dropout_value, model_type, method):
	
	global numberOfSelectedCounties

	dataset.reset_index(drop=True, inplace=True)
	counties = dataset.county_fips.unique()

	val_naive_pred_array = np.array([])
	val_model_pred_array = np.array([])
	val_true_pred_array = np.array([])

	for county in counties:
		county_dataset = dataset.loc[dataset['county_fips'] == county]
		numberOfSelectedCounties = len(county_dataset['county_fips'].unique())
		new_dataset = clean_data(county_dataset, numberOfSelectedCounties, 'country')
		X_train, X_val, X_test, y_train, y_val, y_test, y_train_date, _, y_val_date, _, _ = preprocess(new_dataset, 'country', 1)

		train_X, val_X, test_X, train_y, val_y, test_y = part_of_data(X_train, X_val, X_test, y_train, y_val, y_test, c)
		train_X, train_y, val_X, val_y, test_X, test_y, scaler = data_normalize(train_X, train_y, val_X, val_y, test_X, test_y)

		model = Sequential()
		model.add(LSTM(types[model_type][0], kernel_initializer=tf.keras.initializers.Identity(), return_sequences=True, input_shape=(1, train_X.shape[2])))
		model.add(LeakyReLU(alpha=0.3))
		for i in types[model_type][1:len(types[model_type])-1]:
			model.add(LSTM(i, dropout=dropout_value, return_sequences=True))
			model.add(LeakyReLU(alpha=0.3))
		model.add(LSTM(types[model_type][-1]))
		model.add(LeakyReLU(alpha=0.3))
		model.add(Dense(1))
		model.add(LeakyReLU(alpha=0.3))

		model.compile (
			loss=tf.keras.losses.MeanAbsoluteError(),
			optimizer=keras.optimizers.Adam(0.0001),
			metrics=[tf.keras.metrics.MeanAbsoluteError()]
		)

		early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1, patience=3)
	
		model.fit (
			train_X, train_y,
			epochs=500,
			batch_size=128,
			validation_split=0.2,
			verbose=1,
			callbacks=[early_stop], 
			shuffle=False
		)

		y_pred = model.predict(val_X)
		y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))

		regular_data = regular_data_list[h]
		regular_data = regular_data.loc[regular_data['county_fips'] == county]
		numberOfSelectedCounties = len(regular_data['county_fips'].unique())
		new_dataset_2 = clean_data(regular_data, numberOfSelectedCounties, 'country')
		_, _, _, _, _, _, _, _, _, val_naive_pred_2, _ = preprocess(new_dataset_2, 'country', 1)

		y_true, y_pred = make_org_data(h, y_pred, y_val_date, y_train_date, regular_data, -1, method, 'val')
		y_pred[y_pred <= 0] = 0

		val_true_pred_array = np.concatenate((val_true_pred_array, y_true))
		val_naive_pred_array = np.concatenate((val_naive_pred_array, val_naive_pred_2))
		val_model_pred_array = np.concatenate((val_model_pred_array, y_pred))

		K.clear_session()

	mase = MASE(val_true_pred_array, val_model_pred_array, val_naive_pred_array)
	# mae = mean_absolute_error(val_true_pred_array, val_model_pred_array)
	# mape = mean_absolute_percentage_error(val_true_pred_array, val_model_pred_array)

	return mase


def choose_best_model(moving_average_results_all_country_df, moving_average_results_per_state_df, moving_average_results_per_county_df):

	global numberOfSelectedCounties

	best_h, best_c, mode = choose_winner_model(moving_average_results_all_country_df, moving_average_results_per_state_df, moving_average_results_per_county_df)

	print("*********** best_h = " + str(best_h) + " ***********")

	normal_data = makeHistoricalData(fixed_data, temporal_data, best_h, r, test_size, 'death', 'mrmr', 'country', 'regular', [])
	moving_avg_data = makeHistoricalData(fixed_data, temporal_data, best_h, r, test_size, 'death', 'mrmr', 'country', 'weeklymovingaverage', [])
	cumulative_data = makeHistoricalData(fixed_data, temporal_data, best_h, r, test_size, 'death', 'mrmr', 'country', 'cumulative', [])

	if mode == 0:   # all country
		print("Running the winner model with normal method")
		normal_model_result = lstm_model_all_country(normal_data, best_h, best_c, 0.2, default_model_type, 'regular')
		print("Running the winner model with moving average method")
		moving_avg_model_result = lstm_model_all_country(moving_avg_data, best_h, best_c, 0.2, default_model_type, 'weeklymovingaverage')
		print("Running the winner model with cumulative method")
		cumulative_model_result = lstm_model_all_country(cumulative_data, best_h, best_c, 0.2, default_model_type, 'cumulative')

		results = [{"normal": normal_model_result, "moving_avg": moving_avg_model_result, "cumulative": cumulative_model_result}]
		# tmp = list(itertools.chain.from_iterable(results))
		results_df = pd.DataFrame(results)
		results_df.to_csv('Results/winner_model_all_country.csv', index=False)

		temp_list = [normal_model_result, moving_avg_model_result, cumulative_model_result]

		best = temp_list.index(min(temp_list))

		if best == 0:
			return tuple((best_h, best_c, 0, mode))	# 0 shows best model should be implemented in normal method
		elif best == 1:
			return tuple((best_h, best_c, 1, mode))	# 1 shows best model should be implemented in moving average method
		elif best == 2:
			return tuple((best_h, best_c, 2, mode))	# 2 shows best model should be implemented in cumulative method

	elif mode == 1:     # per state
		print("Running the winner model with normal method")
		normal_model_result = lstm_model_per_state(normal_data, best_h, best_c, 0.2, default_model_type, 'regular')
		print("Running the winner model with moving average method")
		moving_avg_model_result = lstm_model_per_state(moving_avg_data, best_h, best_c, 0.2, default_model_type, 'weeklymovingaverage')
		print("Running the winner model with cumulative method")
		cumulative_model_result = lstm_model_per_state(cumulative_data, best_h, best_c, 0.2, default_model_type, 'cumulative')

		results = [{"normal": normal_model_result, "moving_avg": moving_avg_model_result, "cumulative": cumulative_model_result}]
		# tmp = list(itertools.chain.from_iterable(results))
		results_df = pd.DataFrame(results)
		results_df.to_csv('Results/winner_model_per_state.csv', index=False)

		temp_list = [normal_model_result, moving_avg_model_result, cumulative_model_result]

		best = temp_list.index(min(temp_list))

		if best == 0:
			return tuple((best_h, best_c, 0, mode))	# 0 shows best model should be implemented in normal method
		elif best == 1:
			return tuple((best_h, best_c, 1, mode))	# 1 shows best model should be implemented in moving average method
		elif best == 2:
			return tuple((best_h, best_c, 2, mode))	# 2 shows best model should be implemented in cumulative method

	elif mode == 2:     # per county
		print("Running the winner model with normal method")
		normal_model_result = lstm_model_per_county(normal_data, best_h, best_c, 0.2, default_model_type, 'regular')
		print("Running the winner model with moving average method")
		moving_avg_model_result = lstm_model_per_county(moving_avg_data, best_h, best_c, 0.2, default_model_type, 'weeklymovingaverage')
		print("Running the winner model with cumulative method")
		cumulative_model_result = lstm_model_per_county(cumulative_data, best_h, best_c, 0.2, default_model_type, 'cumulative')

		results = [{"normal": normal_model_result, "moving_avg": moving_avg_model_result, "cumulative": cumulative_model_result}]
		# tmp = list(itertools.chain.from_iterable(results))
		results_df = pd.DataFrame(results)
		results_df.to_csv('Results/winner_model_per_county.csv', index=False)

		temp_list = [normal_model_result, moving_avg_model_result, cumulative_model_result]

		best = temp_list.index(min(temp_list))

		if best == 0:
			return tuple((best_h, best_c, 0, mode))	# 0 shows best model should be implemented in normal method
		elif best == 1:
			return tuple((best_h, best_c, 1, mode))	# 1 shows best model should be implemented in moving average method
		elif best == 2:
			return tuple((best_h, best_c, 2, mode))	# 2 shows best model should be implemented in cumulative method


def choose_best_number_of_layers_and_dropout_value(h, c, method, mode):	# level 8 in flowchart

	data = pd.DataFrame()
	restore_method = ''

	if method == 0:	# normal method
		data = makeHistoricalData(fixed_data, temporal_data, h, r, test_size, 'death', 'mrmr', 'country', 'regular', [])
		restore_method = 'regular'

	elif method == 1:	# moving average method
		data = makeHistoricalData(fixed_data, temporal_data, h, r, test_size, 'death', 'mrmr', 'country', 'weeklymovingaverage', [])
		restore_method = 'weeklymovingaverage'
		# data = moving_avg_data = moving_average_data[h]

	elif method == 2:	#cumulative method
		data = makeHistoricalData(fixed_data, temporal_data, h, r, test_size, 'death', 'mrmr', 'country', 'cumulative', [])
		restore_method = 'cumulative'

	if mode == 0:	# all country mode
		models_best_results = []
		models_best_dropout_values = []

		for model_type in range(len(types)):		# loop over model types
			with mp.Pool(mp.cpu_count()) as pool:
				results = pool.starmap(lstm_model_all_country, [(data, h, c, d/10, model_type, restore_method) for d in range(1, 6)])

			best_result = min(results)	# getting min mase (mase for best dropout on this model type)
			models_best_results.append(best_result)
			
			dropout_value = results.index(min(results))
			models_best_dropout_values.append(dropout_value)

		best_type = models_best_results.index(min(models_best_results))
		best_dropout_value = models_best_dropout_values[best_type]

		return tuple((best_type, best_dropout_value))

	elif mode == 1:		# per state mode
		models_best_results = []
		models_best_dropout_values = []

		for model_type in range(len(types)):		# loop over model types
			with mp.Pool(mp.cpu_count()) as pool:
				results = pool.starmap(lstm_model_per_state, [(data, h, c, d/10, model_type, restore_method) for d in range(1, 6)])

			best_result = min(results)	# getting min mase (mase for best dropout on this model type)
			models_best_results.append(best_result)
			
			dropout_value = results.index(min(results))
			models_best_dropout_values.append(dropout_value)

		best_type = models_best_results.index(min(models_best_results))
		best_dropout_value = models_best_dropout_values[best_type]

		return tuple((best_type, best_dropout_value))

	elif mode == 2:		# per county mode
		models_best_results = []
		models_best_dropout_values = []

		for model_type in range(len(types)):		# loop over model types
			with mp.Pool(mp.cpu_count()) as pool:
				results = pool.starmap(lstm_model_per_county, [(data, h, c, d/10, model_type, restore_method) for d in range(1, 6)])

			best_result = min(results)	# getting min mase (mase for best dropout on this model type)
			models_best_results.append(best_result)
			
			dropout_value = results.index(min(results))
			models_best_dropout_values.append(dropout_value)

		best_type = models_best_results.index(min(models_best_results))
		best_dropout_value = models_best_dropout_values[best_type]

		return tuple((best_type, best_dropout_value))


def lstm_model_on_test_all_country(dataset, h, c, dropout_value, model_type, method):	# this function works with specific values for h and c

	global numberOfSelectedCounties

	dataset.reset_index(drop=True, inplace=True)
	numberOfSelectedCounties = len(dataset['county_fips'].unique())
	new_dataset = clean_data(dataset, numberOfSelectedCounties, 'country')
	X_train, X_val, X_test, y_train, y_val, y_test, y_train_date, y_test_date, _, _, _ = preprocess(new_dataset, 'country', 1)

	train_X, val_X, test_X, train_y, val_y, test_y = part_of_data(X_train, X_val, X_test, y_train, y_val, y_test, c)
	train_X, train_y, val_X, val_y, test_X, test_y, scaler = data_normalize(train_X, train_y, val_X, val_y, test_X, test_y)

	model = Sequential()
	model.add(LSTM(types[model_type][0], kernel_initializer=tf.keras.initializers.Identity(), return_sequences=True, input_shape=(1, train_X.shape[2])))
	model.add(LeakyReLU(alpha=0.3))
	for i in types[model_type][1:len(types[model_type])-1]:
		model.add(LSTM(i, dropout=dropout_value, return_sequences=True))
		model.add(LeakyReLU(alpha=0.3))
	model.add(LSTM(types[model_type][-1]))
	model.add(LeakyReLU(alpha=0.3))
	model.add(Dense(1))
	model.add(LeakyReLU(alpha=0.3))

	model.compile (
		loss=tf.keras.losses.MeanAbsoluteError(),
		optimizer=keras.optimizers.Adam(0.0001),
		metrics=[tf.keras.metrics.MeanAbsoluteError()]
	)

	early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1, patience=3)
	
	model.fit (
		train_X, train_y,
		epochs=500,
		batch_size=128,
		validation_split=0.2,
		verbose=1,
		callbacks=[early_stop], 
		shuffle=False
	)

	y_pred = model.predict(test_X)
	y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))
	
	regular_data = regular_data_list[h]
	numberOfSelectedCounties = len(regular_data['county_fips'].unique())
	new_dataset_2 = clean_data(regular_data, numberOfSelectedCounties, 'country')
	_, _, _, _, _, _, _, _, _, _, test_naive_pred_2 = preprocess(new_dataset_2, 'country', 1)

	y_true, y_pred = make_org_data(h, y_pred, y_test_date, y_train_date, regular_data, -1, method, 'val')
	y_pred[y_pred <= 0] = 0

	mase = MASE(y_true, y_pred, test_naive_pred_2)
	mae = mean_absolute_error(y_true, y_pred)
	mape = mean_absolute_percentage_error(y_true, y_pred)

	K.clear_session()

	result = [{"MAE": mae, "MAPE": mape, "MASE": mase}]
	
	return tuple((result, y_pred))


def lstm_model_on_test_per_state(dataset, h, c, dropout_value, model_type, method):
	
	global numberOfSelectedCounties

	dataset.reset_index(drop=True, inplace=True)
	states = dataset.state_fips.unique()

	test_naive_pred_array = np.array([])
	test_model_pred_array = np.array([])
	test_true_pred_array = np.array([])

	for state in states:
		state_dataset = dataset.loc[dataset['state_fips'] == state]
		numberOfSelectedCounties = len(state_dataset['county_fips'].unique())
		new_dataset = clean_data(state_dataset, numberOfSelectedCounties, 'country')
		X_train, X_val, X_test, y_train, y_val, y_test, y_train_date, y_test_date, _, _, _ = preprocess(new_dataset, 'country', 1)

		train_X, val_X, test_X, train_y, val_y, test_y = part_of_data(X_train, X_val, X_test, y_train, y_val, y_test, c)
		train_X, train_y, val_X, val_y, test_X, test_y, scaler = data_normalize(train_X, train_y, val_X, val_y, test_X, test_y)

		model = Sequential()
		model.add(LSTM(types[model_type][0], kernel_initializer=tf.keras.initializers.Identity(), return_sequences=True, input_shape=(1, train_X.shape[2])))
		model.add(LeakyReLU(alpha=0.3))
		for i in types[model_type][1:len(types[model_type])-1]:
			model.add(LSTM(i, dropout=dropout_value, return_sequences=True))
			model.add(LeakyReLU(alpha=0.3))
		model.add(LSTM(types[model_type][-1]))
		model.add(LeakyReLU(alpha=0.3))
		model.add(Dense(1))
		model.add(LeakyReLU(alpha=0.3))

		model.compile (
			loss=tf.keras.losses.MeanAbsoluteError(),
			optimizer=keras.optimizers.Adam(0.0001),
			metrics=[tf.keras.metrics.MeanAbsoluteError()]
		)

		early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1, patience=3)
	
		model.fit (
			train_X, train_y,
			epochs=500,
			batch_size=128,
			validation_split=0.2,
			verbose=1,
			callbacks=[early_stop], 
			shuffle=False
		)

		y_pred = model.predict(test_X)
		y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))

		regular_data = regular_data_list[h]
		regular_data = regular_data.loc[regular_data['state_fips'] == state]
		numberOfSelectedCounties = len(regular_data['county_fips'].unique())
		new_dataset_2 = clean_data(regular_data, numberOfSelectedCounties, 'country')
		_, _, _, _, _, _, _, _, _, _, test_naive_pred_2 = preprocess(new_dataset_2, 'country', 1)

		y_true, y_pred = make_org_data(h, y_pred, y_test_date, y_train_date, regular_data, -1, method, 'val')
		y_pred[y_pred <= 0] = 0

		test_true_pred_array = np.concatenate((test_true_pred_array, y_true))
		test_naive_pred_array = np.concatenate((test_naive_pred_array, test_naive_pred_2))
		test_model_pred_array = np.concatenate((test_model_pred_array, y_pred))

		K.clear_session()

	mase = MASE(test_true_pred_array, test_model_pred_array, test_naive_pred_array)
	mae = mean_absolute_error(test_true_pred_array, test_model_pred_array)
	mape = mean_absolute_percentage_error(test_true_pred_array, test_model_pred_array)

	result = [{"MAE": mae, "MAPE": mape, "MASE": mase}]
	
	return tuple((result, test_model_pred_array))


def lstm_model_on_test_per_county(dataset, h, c, dropout_value, model_type, method):
	
	global numberOfSelectedCounties

	dataset.reset_index(drop=True, inplace=True)
	counties = dataset.county_fips.unique()

	test_naive_pred_array = np.array([])
	test_model_pred_array = np.array([])
	test_true_pred_array = np.array([])

	for county in counties:
		county_dataset = dataset.loc[dataset['county_fips'] == county]
		numberOfSelectedCounties = len(county_dataset['county_fips'].unique())
		new_dataset = clean_data(county_dataset, numberOfSelectedCounties, 'country')
		X_train, X_val, X_test, y_train, y_val, y_test, y_train_date, y_test_date, _, _, _ = preprocess(new_dataset, 'country', 1)

		train_X, val_X, test_X, train_y, val_y, test_y = part_of_data(X_train, X_val, X_test, y_train, y_val, y_test, c)
		train_X, train_y, val_X, val_y, test_X, test_y, scaler = data_normalize(train_X, train_y, val_X, val_y, test_X, test_y)

		model = Sequential()
		model.add(LSTM(types[model_type][0], kernel_initializer=tf.keras.initializers.Identity(), return_sequences=True, input_shape=(1, train_X.shape[2])))
		model.add(LeakyReLU(alpha=0.3))
		for i in types[model_type][1:len(types[model_type])-1]:
			model.add(LSTM(i, dropout=dropout_value, return_sequences=True))
			model.add(LeakyReLU(alpha=0.3))
		model.add(LSTM(types[model_type][-1]))
		model.add(LeakyReLU(alpha=0.3))
		model.add(Dense(1))
		model.add(LeakyReLU(alpha=0.3))

		model.compile (
			loss=tf.keras.losses.MeanAbsoluteError(),
			optimizer=keras.optimizers.Adam(0.0001),
			metrics=[tf.keras.metrics.MeanAbsoluteError()]
		)

		early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1, patience=3)
	
		model.fit (
			train_X, train_y,
			epochs=500,
			batch_size=128,
			validation_split=0.2,
			verbose=1,
			callbacks=[early_stop], 
			shuffle=False
		)

		y_pred = model.predict(test_X)
		y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))

		regular_data = regular_data_list[h]
		regular_data = regular_data.loc[regular_data['county_fips'] == county]
		numberOfSelectedCounties = len(regular_data['county_fips'].unique())
		new_dataset_2 = clean_data(regular_data, numberOfSelectedCounties, 'country')
		_, _, _, _, _, _, _, _, _, _, test_naive_pred_2 = preprocess(new_dataset_2, 'country', 1)

		y_true, y_pred = make_org_data(h, y_pred, y_test_date, y_train_date, regular_data, -1, method, 'val')
		y_pred[y_pred <= 0] = 0
		
		test_true_pred_array = np.concatenate((test_true_pred_array, y_true))
		test_naive_pred_array = np.concatenate((test_naive_pred_array, test_naive_pred_2))
		test_model_pred_array = np.concatenate((test_model_pred_array, y_pred))

		K.clear_session()

	mase = MASE(test_true_pred_array, test_model_pred_array, test_naive_pred_array)
	mae = mean_absolute_error(test_true_pred_array, test_model_pred_array)
	mape = mean_absolute_percentage_error(test_true_pred_array, test_model_pred_array)

	result = [{"MAE": mae, "MAPE": mape, "MASE": mase}]

	return tuple((result, test_model_pred_array))


def run_final_model_on_test(h, c, method, mode, model_type, dropout_value):

	data = pd.DataFrame()
	restore_method = ''

	if method == 0:	# normal method
		data = makeHistoricalData(fixed_data, temporal_data, h, r, test_size, 'death', 'mrmr', 'country', 'regular', [])
		restore_method = 'regular'

	elif method == 1:	# moving average method
		data = makeHistoricalData(fixed_data, temporal_data, h, r, test_size, 'death', 'mrmr', 'country', 'weeklymovingaverage', [])
		restore_method = 'weeklymovingaverage'

	elif method == 2:	#cumulative method
		data = makeHistoricalData(fixed_data, temporal_data, h, r, test_size, 'death', 'mrmr', 'country', 'cumulative', [])
		restore_method = 'cumulative'

	if mode == 0:	# all country mode
		scores, result = lstm_model_on_test_all_country(data, h, c, dropout_value, model_type, restore_method)
		return tuple((scores, result))

	elif mode == 1:	# per state mode
		scores, result = lstm_model_on_test_per_state(data, h, c, dropout_value, model_type, restore_method)
		return tuple((scores, result))

	elif mode == 2:	# per county mode
		scores, result = lstm_model_on_test_per_county(data, h, c, dropout_value, model_type, restore_method)
		return tuple((scores, result))


def send_mail(mail_subject, mail_body, path, file_name):
	fromaddr = "lstm.covid19.server@gmail.com"
	toaddr = "m1998naderi@gmail.com"

	msg = MIMEMultipart()

	msg['From'] = fromaddr
	msg['To'] = toaddr

	msg['Subject'] = mail_subject

	body = mail_body
	msg.attach(MIMEText(body, 'plain'))

	filename = file_name
	filepath = path+"/"+file_name
	attachment = open(filepath, "rb")

	p = MIMEBase('application', 'octet-stream')
	p.set_payload((attachment).read())

	encoders.encode_base64(p)

	p.add_header('Content-Disposition', "attachment; filename= %s" % filename)

	msg.attach(p)

	s = smtplib.SMTP('smtp.gmail.com', 587)
	s.starttls()
	s.login(fromaddr, "4%f{h=W%m'f85cC7")

	text = msg.as_string()

	s.sendmail(fromaddr, toaddr, text)
	s.quit()


def main():
	# global numberOfSelectedCounties
	
	############################################################################### level 1, 2
	# constructing moving average data with 'h' from 1 to 21
	# parallelized on 'h'
	print("Started to make historical data (moving average method)")
	start_time = time.time()
	with mp.Pool(mp.cpu_count()) as pool:
		moving_average_data = pool.starmap(makeHistoricalData, 
			[(fixed_data, temporal_data, h, r, test_size, 'death', 'mrmr', 'country', 'weeklymovingaverage', []) for h in range(1, H+1)])

	# with mp.Pool(mp.cpu_count()) as pool:
	# 	regular_data_list = pool.starmap(makeHistoricalData, 
	# 		[(fixed_data, temporal_data, h, r, test_size, 'death', 'mrmr', 'country', 'regular', []) for h in range(1, H+1)])

	print("Making moving average historical data is done!")
	print("--- %s seconds to construct moving average data and regular data ---" % (time.time() - start_time))
	###############################################################################

	# save the entire session until now
	shelve_filename = 'level_1to2.out'
	s = shelve.open(shelve_filename, 'n')  # 'n' for new
	for key in dir():
		try:
			s[key] = locals()[key]
		except:
			print('ERROR shelving: {0}'.format(key))
	s.close()

	############################################################################### level 5
	# training model on all country with moving average data with 'h' from 1 to 21
	# parallelized on 'h'
	print("Started to train model on moving average data and on all country")
	print("--- level 5 in flowchart ---")
	start_time = time.time()
	
	with mp.Pool(mp.cpu_count()) as pool:
		moving_average_results_all_country = pool.starmap(fit_model_all_country, [(moving_average_data[h], h, c) for h in range(0, H) for c in range(1, 30+10*(h+1))])

	end_time = time.time()
	print("Finished training the model on moving average data and on all country")
	print("--- %s seconds to train the model on all country ---" % (end_time - start_time))

	tmp = list(itertools.chain.from_iterable(moving_average_results_all_country))
	moving_average_results_all_country_df = pd.DataFrame(tmp)

	# writing all country results into a file
	moving_average_results_all_country_df.to_csv('Results/moving_average_results_all_country.csv', index=False)

	with open('Results/moving_average_results_all_country_time.txt', 'w') as fp:
		fp.write("--- %s seconds to train the model on all country ---" % (end_time - start_time))

	# sending mail
	mail_subject = "lstm model - level 5 - all country"
	mail_body = "Finished training the model on moving average data and on all country --- took %s seconds" % (end_time - start_time)

	send_mail(mail_subject, mail_body, "Results", "moving_average_results_all_country.csv")
	###############################################################################

	# save the entire session until now
	shelve_filename = 'level_1to5_1.out'
	s = shelve.open(shelve_filename, 'n')  # 'n' for new
	for key in dir():
		try:
			s[key] = locals()[key]
		except:
			print('ERROR shelving: {0}'.format(key))
	s.close()

	############################################################################### level 5
	# training model per state with moving average data with 'h' from 1 to 21
	# parallelized on 'state'
	print("Started to train model on moving average data and per state")
	print("--- level 5 in flowchart ---")
	start_time = time.time()

	with mp.Pool(mp.cpu_count()) as pool:
		moving_average_results_per_state = pool.starmap(fit_model_per_state, [(moving_average_data[h], h, c) for h in range(0, H) for c in range(1, 30+10*(h+1))])

	end_time = time.time()
	print("Finished training the model on moving average data and per state")
	print("--- %s seconds to train the model per state ---" % (end_time - start_time))

	tmp = list(itertools.chain.from_iterable(moving_average_results_per_state))
	moving_average_results_per_state_df = pd.DataFrame(tmp)

	# writing per state results into a file
	moving_average_results_per_state_df.to_csv('Results/moving_average_results_per_state.csv', index=False)

	with open('Results/moving_average_results_per_state_time.txt', 'w') as fp:
		fp.write("--- %s seconds to train the model per state ---" % (end_time - start_time))

	# sending mail
	mail_subject = "lstm model - level 5 - per state"
	mail_body = "Finished training the model on moving average data and per state --- took %s seconds" % (end_time - start_time)

	send_mail(mail_subject, mail_body, "Results", "moving_average_results_per_state.csv")
	###############################################################################

	# save the entire session until now
	shelve_filename = 'level_1to5_2.out'
	s = shelve.open(shelve_filename, 'n')  # 'n' for new
	for key in dir():
		try:
			s[key] = locals()[key]
		except:
			print('ERROR shelving: {0}'.format(key))
	s.close()

	############################################################################### level 5
	# training model per county with moving average data with 'h' from 1 to 21
	# parallelized on 'h'
	print("Started to train model on moving average data and per county")
	print("--- level 5 in flowchart ---")
	start_time = time.time()

	with mp.Pool(mp.cpu_count()) as pool:
		moving_average_results_per_county = pool.starmap(fit_model_per_county, [(moving_average_data[h], h, c) for h in range(0, H) for c in range(1, 30+10*(h+1))])

	end_time = time.time()
	print("Finished training the model on moving average data and per county")
	print("--- %s seconds to train the model per county ---" % (time.time() - start_time))

	tmp = list(itertools.chain.from_iterable(moving_average_results_per_county))
	moving_average_results_per_county_df = pd.DataFrame(tmp)

	# writing per county results into a file
	moving_average_results_per_county_df.to_csv('Results/moving_average_results_per_county.csv', index=False)

	with open('Results/moving_average_results_per_county_time.txt', 'w') as fp:
		fp.write("--- %s seconds to train the model per county ---" % (end_time - start_time))

	# sending mail
	mail_subject = "lstm model - level 5 - per county"
	mail_body = "Finished training the model on moving average data and per county --- took %s seconds" % (time.time() - start_time)

	send_mail(mail_subject, mail_body, "Results", "moving_average_results_per_county.csv")
	###############################################################################

	# save the entire session until now
	shelve_filename = 'level_1to5_3.out'
	s = shelve.open(shelve_filename, 'n')  # 'n' for new
	for key in dir():
		try:
			s[key] = locals()[key]
		except:
			print('ERROR shelving: {0}'.format(key))
	s.close()

	############################################################################### level 6, 7, 8, 9
	# choosing a method for the best model
	print("Started to choose the best model")
	print("--- level 9 in flowchart ---")
	start_time = time.time()
	best_h, best_c, best_method, best_mode = choose_best_model(moving_average_results_all_country_df, moving_average_results_per_state_df,
																moving_average_results_per_county_df)
	end_time = time.time()
	print("Finished choosing the best model")
	print("--- %s seconds to train the models and choose the best model ---" % (end_time - start_time))
	methods = ['normal', 'moving average', 'cumulative']
	print("best method chosen for training is " + methods[best_method])

	with open('Results/choosing_best_model.txt', 'w') as fp:
		fp.write("best method chosen for training is " + methods[best_method])
		fp.write("\nbest_h = " + str(best_h))
		fp.write("\nbest_c = " + str(best_c))
		fp.write("\n--- %s seconds to train the models and choose the best model ---" % (end_time - start_time))

	# sending mail
	mail_subject = "lstm model - level 6, 7, 8, 9"
	mail_body = "Finished choosing the best model --- took %s seconds" % (end_time - start_time)

	send_mail(mail_subject, mail_body, "Results", "choosing_best_model.txt")
	###############################################################################

	# save the entire session until now
	shelve_filename = 'level_1to9.out'
	s = shelve.open(shelve_filename, 'n')  # 'n' for new
	for key in dir():
		try:
			s[key] = locals()[key]
		except:
			print('ERROR shelving: {0}'.format(key))
	s.close()

	############################################################################### run the model gained until now
	start_time = time.time()
	test_data_scores, test_data_preds = run_final_model_on_test(best_h, best_c, best_method, best_mode, default_model_type, 0.2)
	end_time = time.time()

	test_data_scores_df = pd.DataFrame(test_data_scores)
	test_data_scores_df.to_csv('Results/test_data_scores_level(9).csv', index=False)	# saving scores into a csv file

	np.savetxt('Results/test_data_preds_level(9).txt', test_data_preds, fmt='%f')	# saving predictions on test dataset

	# Create a ZipFile Object
	with ZipFile('Results/test_data_results_level(9).zip', 'w') as zipObj:
		# Add multiple files to the zip
		zipObj.write('Results/test_data_scores_level(9).csv')
		zipObj.write('Results/test_data_preds_level(9).txt')

	# sending mail
	mail_subject = "lstm model - results on test dataset - level(9)"
	mail_body = "Finished training the model and evaluating on test dataset in level 9 --- took %s seconds" % (end_time - start_time)

	send_mail(mail_subject, mail_body, "Results", "test_data_results_level(9).zip")
	###############################################################################

	############################################################################### level 10
	start_time = time.time()
	best_type, dropout_value = choose_best_number_of_layers_and_dropout_value(best_h, best_c, best_method, best_mode)
	end_time = time.time()

	print("Finished choosing the number of layers and the best dropout value")
	print("--- %s seconds to choose the number of layers and the best dropout value ---" % (end_time - start_time))
	dropout_values = [0.1, 0.2, 0.3, 0.4, 0.5]
	print("number of layers for best model is based on " + str(types[best_type]))
	print("best dropout value in the best model is " + str(dropout_values[dropout_value]))

	with open('Results/number_of_layers.txt', 'w') as fp:
		fp.write("number of layers for best model is based on " + str(types[best_type]))
		fp.write("\nbest dropout value in the best model is " + str(dropout_values[dropout_value]))
		fp.write("\n--- %s seconds to choose the number of layers and the best dropout value ---" % (end_time - start_time))

	# sending mail
	mail_subject = "lstm model - level 10"
	mail_body = "Finished choosing the number of layers and the best dropout value --- took %s seconds" % (end_time - start_time)

	send_mail(mail_subject, mail_body, "Results", "number_of_layers.txt")
	###############################################################################

	# save the entire session until now
	shelve_filename = 'level_1to9.out'
	s = shelve.open(shelve_filename, 'n')  # 'n' for new
	for key in dir():
		try:
			s[key] = locals()[key]
		except:
			print('ERROR shelving: {0}'.format(key))
	s.close()

	###############################################################################
	# and finally storing all the variables needed for implementing the final model in a txt file
	methods = ['normal', 'moving average', 'cumulative']
	modes = ['all country', 'per state', 'per county']
	dropout_values = [0.1, 0.2, 0.3, 0.4, 0.5]

	with open('Results/final_model.txt', 'w') as fp:
		fp.write("best method : " + methods[best_method])
		fp.write("\nbest mode : " + modes[best_mode])
		fp.write("\nbest value for h : " + str(best_h))
		fp.write("\nbest value for c : " + str(best_c))
		fp.write("\nnumber of layers is based on : " + str(types[best_type]))
		fp.write("\ndropout value : " + str(dropout_values[dropout_value]))

	# sending mail
	mail_subject = "lstm model - best parameters"
	mail_body = "\n\nbest model parameters\n\n"

	send_mail(mail_subject, mail_body, "Results", "final_model.txt")
	###############################################################################

	###############################################################################
	# running the best model type with best parameters on test dataset
	start_time = time.time()
	test_data_scores, test_data_preds = run_final_model_on_test(best_h, best_c, best_method, best_mode, best_type, dropout_values[dropout_value])
	end_time = time.time()

	print("Finished training the model and evaluating on test dataset")
	print("--- %s seconds for training the model and evaluating on test dataset ---" % (end_time - start_time))

	test_data_scores_df = pd.DataFrame(test_data_scores)
	test_data_scores_df.to_csv('Results/test_data_scores.csv', index=False)	# saving scores into a csv file

	np.savetxt('Results/test_data_preds.txt', test_data_preds, fmt='%f')	# saving predictions on test dataset

	# Create a ZipFile Object
	with ZipFile('Results/test_data_results.zip', 'w') as zipObj:
		# Add multiple files to the zip
		zipObj.write('Results/test_data_scores.csv')
		zipObj.write('Results/test_data_preds.txt')

	# sending mail
	mail_subject = "lstm model - results on test dataset"
	mail_body = "Finished training the model and evaluating on test dataset --- took %s seconds" % (end_time - start_time)

	send_mail(mail_subject, mail_body, "Results", "test_data_results.zip")
	###############################################################################

	# save the entire session until now
	shelve_filename = 'final.out'
	s = shelve.open(shelve_filename, 'n')  # 'n' for new
	for key in dir():
		try:
			s[key] = locals()[key]
		except:
			print('ERROR shelving: {0}'.format(key))
	s.close()

	print("\n\n**************** Code execution completed successfully! ****************\n")


if __name__ == "__main__":
	main()
