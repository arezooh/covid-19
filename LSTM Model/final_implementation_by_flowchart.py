import pandas as pd
import numpy as np
from makeHistoricalData import makeHistoricalData
import multiprocessing as mp
import matplotlib.pyplot as plt
from sklearn import preprocessing
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras import backend as K
from pathlib import Path
import time
import math
import statistics

# mkdir for saving the results in
Path("Results").mkdir(parents=True, exist_ok=True)

r = 14
H = 21
numberOfSelectedCounties = -1
fixed_data = pd.read_csv('data/fixed-data.csv')
temporal_data = pd.read_csv('data/temporal-data.csv')

# creating some dataframes to store the results in
moving_average_results_all_country_df = pd.DataFrame(columns=['h', 'c', 'MAE', 'MAPE', 'R2_SCORE'])
moving_average_results_per_state_df = pd.DataFrame(columns=['h', 'c', 'state_fips', 'MAE', 'MAPE', 'R2_SCORE'])
moving_average_results_per_county_df = pd.DataFrame(columns=['h', 'c', 'county_fips', 'MAE', 'MAPE', 'R2_SCORE'])

moving_average_data = []

def splitData(numberOfCounties, main_data, target):

	if numberOfCounties == -1:
		numberOfCounties = len(main_data['county_fips'].unique())

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
	numberOfDays = len(using_data['date of day t'].unique())

	return main_data


def preprocess(main_data):
	
	target = pd.DataFrame(main_data[['date of day t', 'county_fips', 'Target']])
	main_data = main_data.drop(['Target'], axis=1)
	# specify the size of train, validation and test sets
	# produce train, validation and test data in parallel

	X_train_train , X_train_val , X_test , y_train_train , y_train_val , y_test = splitData(numberOfSelectedCounties, main_data, target)
	return X_train_train, y_train_train, X_train_val, y_train_val, X_test, y_test


def data_normalize(X_train, y_train, X_val, y_val, X_test, y_test):
	scalar = preprocessing.MinMaxScaler()

	X_train = X_train.values
	X_train = scalar.fit_transform(X_train)

	X_val = X_val.values
	X_val = scalar.fit_transform(X_val)

	X_test = X_test.values
	X_test = scalar.fit_transform(X_test)

	y_train = y_train.values
	y_train = scalar.fit_transform(y_train.reshape(-1, 1))

	y_val = y_val.values
	y_val = scalar.fit_transform(y_val.reshape(-1, 1))

	# y_test = y_test.values
	# min_max_scaler = preprocessing.MinMaxScaler()
	# y_test = min_max_scaler.fit_transform(y_test.reshape(-1, 1))
	#############################################

	X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
	X_val = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))
	X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

	y_train = y_train.reshape((y_train.shape[0]), )
	y_val = y_val.reshape((y_val.shape[0]), )
	# y_test = y_test.reshape((y_test.shape[0]), )

	return X_train, y_train, X_val, y_val, X_test, y_test


def coeff_determination(y_true, y_pred):
	SS_res =  K.sum(K.square( y_true-y_pred ))
	SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
	return ( 1 - SS_res/(SS_tot + K.epsilon()) )


def MASE(training_series, testing_series, prediction_series):
	"""	
	parameters:
		training_series: the series used to train the model, 1d numpy array
		testing_series: the test series to predict, 1d numpy array or float
		prediction_series: the prediction of testing_series, 1d numpy array (same size as testing_series) or float
	
	"""
	n = training_series.shape[0]
	d = np.abs(np.diff(training_series)).sum()/(n-1)
	
	errors = np.abs(testing_series - prediction_series)
	return errors.mean()/d
	

def part_of_data(X_train, y_train, X_val, y_val, X_test, y_test, c):
	train_X = X_train.iloc[:, 0:c].copy()
	train_y = y_train.copy()

	val_X = X_val.iloc[:, 0:c].copy()
	val_y = y_val.copy()

	test_X = X_test.iloc[:, 0:c].copy()
	test_y = y_test.copy()

	return train_X, train_y, val_X, val_y, test_X, test_y


def default_model(n):
	model = Sequential()
	model.add(LSTM(8, return_sequences=True, input_shape=(1, n)))
	model.add(LSTM(256, dropout=0.2, return_sequences=True))
	model.add(LSTM(256, dropout=0.2, return_sequences=True))
	model.add(LSTM(128, dropout=0.2, return_sequences=True))
	model.add(LSTM(128))
	model.add(Dense(1, activation='selu'))

	# opt = tf.keras.optimizers.RMSprop(learning_rate=0.1)
	model.compile (
		loss='mean_squared_error', 
		optimizer=keras.optimizers.Adam(0.01), 
		# optimizer=keras.optimizers.SGD(lr=0.01, nesterov=True),
		metrics=['acc', tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanAbsolutePercentageError(), coeff_determination]
	)

	return model


def fit_model_all_country(dataset, h):

	results = []

	numberOfSelectedCounties = len(dataset['county_fips'].unique())
	new_dataset = clean_data(dataset, numberOfSelectedCounties)
	X_train, y_train, X_val, y_val, X_test, y_test = preprocess(new_dataset)

	X_train.drop('date of day t', axis=1, inplace=True)
	X_val.drop('date of day t', axis=1, inplace=True)
	X_test.drop('date of day t', axis=1, inplace=True)

	y_train.drop('date of day t', axis=1, inplace=True)
	y_train.drop('county_fips', axis=1, inplace=True)
	y_val.drop('date of day t', axis=1, inplace=True)
	y_val.drop('county_fips', axis=1, inplace=True)
	y_test.drop('date of day t', axis=1, inplace=True)
	y_test.drop('county_fips', axis=1, inplace=True)

	numberOfCovariates = X_train.shape[1]

	for c in range(1, numberOfCovariates+1):
		print('*************************************')
		print('c = ' + str(c) + ', h = ' + str(h+1))
		# print('c = ' + str(c))
		
		train_X, train_y, val_X, val_y, test_X, test_y = part_of_data(X_train, y_train, X_val, y_val, X_test, y_test, c)
		train_X, train_y, val_X, val_y, test_X, test_y = data_normalize(train_X, train_y, val_X, val_y, test_X, test_y)
		
		model = default_model(train_X.shape[2])

		history = model.fit (
			train_X, train_y, 
			epochs=30,
			batch_size=128,
			validation_data=(val_X, val_y),
			verbose=1,
			shuffle=False
		)

		loss, accuracy, mae, mape, r2score = model.evaluate(val_X, val_y, verbose=0)
		# mase = MASE(train_X, val_X, val_y)

		result_dict = {"h": h+1, "c": c, "MAE": mae, "MAPE": mape, "R2_SCORE": r2score}
		results.append(result_dict.copy())

		K.clear_session()
	
	return results


def fit_model_per_state(dataset, state_fips, h):

	results = []

	state_dataset = dataset.loc[dataset['state_fips'] == state_fips]
	state_dataset.reset_index(drop=True, inplace=True)

	numberOfSelectedCounties = len(state_dataset['county_fips'].unique())
	new_dataset = clean_data(state_dataset, numberOfSelectedCounties)
	X_train, y_train, X_val, y_val, X_test, y_test = preprocess(new_dataset)

	X_train.drop('date of day t', axis=1, inplace=True)
	X_val.drop('date of day t', axis=1, inplace=True)
	X_test.drop('date of day t', axis=1, inplace=True)

	y_train.drop('date of day t', axis=1, inplace=True)
	y_train.drop('county_fips', axis=1, inplace=True)
	y_val.drop('date of day t', axis=1, inplace=True)
	y_val.drop('county_fips', axis=1, inplace=True)
	y_test.drop('date of day t', axis=1, inplace=True)
	y_test.drop('county_fips', axis=1, inplace=True)

	numberOfCovariates = X_train.shape[1]

	for c in range(1, numberOfCovariates+1):
		print('*************************************')
		print('c = ' + str(c) + ', h = ' + str(h) + ', state_fips = ' + str(state_fips))
		# print('c = ' + str(c))
		
		train_X, train_y, val_X, val_y, test_X, test_y = part_of_data(X_train, y_train, X_val, y_val, X_test, y_test, c)
		train_X, train_y, val_X, val_y, test_X, test_y = data_normalize(train_X, train_y, val_X, val_y, test_X, test_y)
		
		model = default_model(train_X.shape[2])

		history = model.fit (
			train_X, train_y, 
			epochs=30,
			batch_size=128,
			validation_data=(val_X, val_y),
			verbose=1,
			shuffle=False
		)

		loss, accuracy, mae, mape, r2score = model.evaluate(val_X, val_y, verbose=0)
		# mase = MASE(train_X, val_X, val_y)

		result_dict = {"h": h+1, "c": c, "state_fips": state_fips, "MAE": mae, "MAPE": mape, "R2_SCORE": r2score}
		results.append(result_dict.copy())

		K.clear_session()

	return results


def fit_model_per_county(dataset, h):

	results = []

	counties_in_raw_data = fixed_data.county_fips.unique()
	counties_in_historical_data= dataset.county_fips.unique()

	for county in counties_in_raw_data:
		if county in counties_in_historical_data:
			county_dataset = dataset.loc[dataset['county_fips'] == county]
		else:
			continue
		
		numberOfSelectedCounties = len(county_dataset['county_fips'].unique())
		new_dataset = clean_data(county_dataset, numberOfSelectedCounties)
		X_train, y_train, X_val, y_val, X_test, y_test = preprocess(new_dataset)

		X_train.drop('date of day t', axis=1, inplace=True)
		X_val.drop('date of day t', axis=1, inplace=True)
		X_test.drop('date of day t', axis=1, inplace=True)

		y_train.drop('date of day t', axis=1, inplace=True)
		y_train.drop('county_fips', axis=1, inplace=True)
		y_val.drop('date of day t', axis=1, inplace=True)
		y_val.drop('county_fips', axis=1, inplace=True)
		y_test.drop('date of day t', axis=1, inplace=True)
		y_test.drop('county_fips', axis=1, inplace=True)

		numberOfCovariates = X_train.shape[1]


		for c in range(1, numberOfCovariates+1):
			print('*************************************')
			print('c = ' + str(c) + ', h = ' + str(h) + ', county_fips = ' + str(county))
			# print('c = ' + str(c))
			
			train_X, train_y, val_X, val_y, test_X, test_y = part_of_data(X_train, y_train, X_val, y_val, X_test, y_test, c)
			train_X, train_y, val_X, val_y, test_X, test_y = data_normalize(train_X, train_y, val_X, val_y, test_X, test_y)
			
			model = default_model(train_X.shape[2])

			history = model.fit (
				train_X, train_y, 
				epochs=30,
				batch_size=128,
				validation_data=(val_X, val_y),
				verbose=1,
				shuffle=False
			)

			loss, accuracy, mae, mape, r2score = model.evaluate(val_X, val_y, verbose=0)
			# mase = MASE(train_X, val_X, val_y)

			result_dict = {"h": h+1, "c": c, "county_fips": county, "MAE": mae, "MAPE": mape, "R2_SCORE": r2score}
			results.append(result_dict.copy())

			K.clear_session()

	return results


def get_best_model_all_country(moving_average_results_all_country_df):	# each element is a tuple of (h, c, R2_SCORE)
	best_r2score, best_h, best_c = 0, 0, 0

	moving_average_results_all_country_df['R2_SCORE'] = pd.to_numeric(moving_average_results_all_country_df['R2_SCORE'])
	index_of_max_r2score = moving_average_results_all_country_df['R2_SCORE'].idxmax()

	best_h = moving_average_results_all_country_df.iloc[index_of_max_r2score]['h']
	best_c = moving_average_results_all_country_df.iloc[index_of_max_r2score]['c']
	best_r2score = moving_average_results_all_country_df.iloc[index_of_max_r2score]['R2_SCORE']
	
	return tuple((best_h, best_c, best_r2score))


def get_best_model_per_state(moving_average_results_per_state_df):	# each element is a tuple of (h, c, state_fips, R2_SCORE)
	best_r2score, best_h, best_c = -math.inf, 0, 0

	unique_h = moving_average_results_per_state_df.h.unique()
	for h in unique_h:
		temp_df = moving_average_results_per_state_df.loc[moving_average_results_per_state_df['h'] == h]
		unique_c = temp_df.c.unique()
		for c in unique_c:
			temp_df2 = moving_average_results_per_state_df.loc[moving_average_results_per_state_df['c'] == c]
			mean_R2_SCORE = temp_df2["R2_SCORE"].mean()
			if mean_R2_SCORE >= best_r2score:
				best_r2score = mean_R2_SCORE
				best_h = h
				best_c = c

	# moving_average_results_per_state_df['mape'] = pd.to_numeric(moving_average_results_per_state_df['mape'])
	# index_of_max_mape = moving_average_results_per_state_df['mape'].idxmax()
	
	# best_h = moving_average_results_per_state_df.iloc[index_of_max_mape]['h']
	# best_c = moving_average_results_per_state_df.iloc[index_of_max_mape]['c']
	# best_mape = moving_average_results_per_state_df.iloc[index_of_max_mape]['mape']
	
	return tuple((best_h, best_c, best_r2score))


def get_best_model_per_county(moving_average_results_per_county_df):	# each element is a tuple of (h, c, county_fips, R2_SCORE)
	best_r2score, best_h, best_c = -math.inf, 0, 0

	unique_h = moving_average_results_per_county_df.h.unique()
	for h in unique_h:
		temp_df = moving_average_results_per_county_df.loc[moving_average_results_per_county_df['h'] == h]
		unique_c = temp_df.c.unique()
		for c in unique_c:
			temp_df2 = moving_average_results_per_county_df.loc[moving_average_results_per_county_df['c'] == c]
			mean_R2_SCORE = temp_df2["R2_SCORE"].mean()
			if mean_R2_SCORE >= best_r2score:
				best_r2score = mean_R2_SCORE
				best_h = h
				best_c = c

	# moving_average_results_per_county_df['mape'] = pd.to_numeric(moving_average_results_per_county_df['mape'])
	# index_of_max_mape = moving_average_results_per_county_df['mape'].idxmax()
	
	# best_h = moving_average_results_per_county_df.iloc[index_of_max_mape]['h']
	# best_c = moving_average_results_per_county_df.iloc[index_of_max_mape]['c']
	# best_mape = moving_average_results_per_county_df.iloc[index_of_max_mape]['mape']
	
	return tuple((best_h, best_c, best_r2score))


def choose_winner_model(moving_average_results_all_country_df, moving_average_results_per_state_df, moving_average_results_per_county_df):

	best_h_all_country, best_c_all_country, r2score_all_country = get_best_model_all_country(moving_average_results_all_country_df)
	best_h_per_state, best_c_per_state, r2score_per_state = get_best_model_per_state(moving_average_results_per_state_df)
	best_h_per_county, best_c_per_county, r2score_per_county = get_best_model_per_county(moving_average_results_per_county_df)

	temp_list = [r2score_all_country, r2score_per_state, r2score_per_county]

	best = temp_list.index(max(temp_list))

	if best == 0:
		return tuple((int(best_h_all_country), int(best_c_all_country), 0))	# 0 shows winner model should be implemented in all country mode
	elif best == 1:
		return tuple((int(best_h_per_state), int(best_c_per_state), 1))	# 1 shows winner model should be implemented in per state mode
	elif best == 2:
		return tuple((int(best_h_per_county), int(best_c_per_county), 2))	# 2 shows winner model should be implemented in per county mode


def winner_model_all_country(dataset, c):	# when mode = 0 in 'choose_winner_model' function

	numberOfSelectedCounties = len(dataset['county_fips'].unique())
	new_dataset = clean_data(dataset, numberOfSelectedCounties)
	X_train, y_train, X_val, y_val, X_test, y_test = preprocess(new_dataset)

	X_train.drop('date of day t', axis=1, inplace=True)
	X_val.drop('date of day t', axis=1, inplace=True)
	X_test.drop('date of day t', axis=1, inplace=True)

	y_train.drop('date of day t', axis=1, inplace=True)
	y_train.drop('county_fips', axis=1, inplace=True)
	y_val.drop('date of day t', axis=1, inplace=True)
	y_val.drop('county_fips', axis=1, inplace=True)
	y_test.drop('date of day t', axis=1, inplace=True)
	y_test.drop('county_fips', axis=1, inplace=True)

	numberOfCovariates = X_train.shape[1]
		
	train_X, train_y, val_X, val_y, test_X, test_y = part_of_data(X_train, y_train, X_val, y_val, X_test, y_test, c)
	train_X, train_y, val_X, val_y, test_X, test_y = data_normalize(train_X, train_y, val_X, val_y, test_X, test_y)
	
	model = default_model(train_X.shape[2])

	history = model.fit (
		train_X, train_y, 
		epochs=30,
		batch_size=128,
		validation_data=(val_X, val_y),
		verbose=1,
		shuffle=False
	)

	loss, accuracy, mae, mape, r2score = model.evaluate(val_X, val_y, verbose=0)
	# mase = MASE(train_X, val_X, val_y)
	# result = [{"MAE": mae, "MAPE": mape}]
	# result = np.asarray(result)
	# result = result.flatten()
	# result = np.array(result).tolist()
	# result_df = pd.DataFrame(result)
	# result_df.to_csv('winner_model_all_country_result.csv', index=False)

	K.clear_session()

	return r2score


def winner_model_per_state(dataset, c):	# when mode = 1 in 'choose_winner_model' function
	
	results_list = []

	############################################################################
	def fit_model_on_state(dataset, state_fips, c):
		state_dataset = dataset.loc[dataset['state_fips'] == state_fips]
		state_dataset.reset_index(drop=True, inplace=True)

		numberOfSelectedCounties = len(state_dataset['county_fips'].unique())
		new_dataset = clean_data(state_dataset, numberOfSelectedCounties)
		X_train, y_train, X_val, y_val, X_test, y_test = preprocess(new_dataset)

		X_train.drop('date of day t', axis=1, inplace=True)
		X_val.drop('date of day t', axis=1, inplace=True)
		X_test.drop('date of day t', axis=1, inplace=True)

		y_train.drop('date of day t', axis=1, inplace=True)
		y_train.drop('county_fips', axis=1, inplace=True)
		y_val.drop('date of day t', axis=1, inplace=True)
		y_val.drop('county_fips', axis=1, inplace=True)
		y_test.drop('date of day t', axis=1, inplace=True)
		y_test.drop('county_fips', axis=1, inplace=True)

		numberOfCovariates = X_train.shape[1]
			
		train_X, train_y, val_X, val_y, test_X, test_y = part_of_data(X_train, y_train, X_val, y_val, X_test, y_test, c)
		train_X, train_y, val_X, val_y, test_X, test_y = data_normalize(train_X, train_y, val_X, val_y, test_X, test_y)
		
		model = default_model(train_X.shape[2])

		history = model.fit (
			train_X, train_y, 
			epochs=30,
			batch_size=128,
			validation_data=(val_X, val_y),
			verbose=1,
			shuffle=False
		)

		loss, accuracy, mae, mape, r2score = model.evaluate(val_X, val_y, verbose=0)
		# mase = MASE(train_X, val_X, val_y)

		temp_dict = {"MAE": mae, "MAPE": mape, "R2_SCORE": r2score}
		results_list.append(temp_dict)
		
		K.clear_session()

		return r2score
	############################################################################

	states = dataset.state_fips.unique()
	with mp.Pool(mp.cpu_count()) as pool:
		results = pool.starmap(fit_model_on_state, [(dataset, state_fips, c) for state_fips in states])

	# results_list = np.asarray(results_list)
	# results_list = results_list.flatten()
	# results_list = np.array(results_list).tolist()
	# results_list_df = pd.DataFrame(results_list)
	# results_list_df.to_csv('winner_model_all_per_state.csv', index=False)

	return results


def winner_model_per_county(dataset, c):	# when mode = 2 in 'choose_winner_model' function
	
	results = []
	results_list = []

	counties_in_raw_data = fixed_data.county_fips.unique()
	counties_in_historical_data= dataset.county_fips.unique()

	for county in counties_in_raw_data:
		if county in counties_in_historical_data:
			county_dataset = dataset.loc[dataset['county_fips'] == county]
		else:
			continue
		
		numberOfSelectedCounties = len(county_dataset['county_fips'].unique())
		new_dataset = clean_data(county_dataset, numberOfSelectedCounties)
		X_train, y_train, X_val, y_val, X_test, y_test = preprocess(new_dataset)

		X_train.drop('date of day t', axis=1, inplace=True)
		X_val.drop('date of day t', axis=1, inplace=True)
		X_test.drop('date of day t', axis=1, inplace=True)

		y_train.drop('date of day t', axis=1, inplace=True)
		y_train.drop('county_fips', axis=1, inplace=True)
		y_val.drop('date of day t', axis=1, inplace=True)
		y_val.drop('county_fips', axis=1, inplace=True)
		y_test.drop('date of day t', axis=1, inplace=True)
		y_test.drop('county_fips', axis=1, inplace=True)

		numberOfCovariates = X_train.shape[1]
			
		train_X, train_y, val_X, val_y, test_X, test_y = part_of_data(X_train, y_train, X_val, y_val, X_test, y_test, c)
		train_X, train_y, val_X, val_y, test_X, test_y = data_normalize(train_X, train_y, val_X, val_y, test_X, test_y)
		
		model = default_model(train_X.shape[2])

		history = model.fit (
			train_X, train_y, 
			epochs=30,
			batch_size=128,
			validation_data=(val_X, val_y),
			verbose=1,
			shuffle=False
		)

		loss, accuracy, mae, mape, r2score = model.evaluate(val_X, val_y, verbose=0)
		# mase = MASE(train_X, val_X, val_y)

		results.append(r2score)

		temp_dict = {"MAE": mae, "MAPE": mape, "R2_SCORE": r2score}
		results_list.append(temp_dict)

		K.clear_session()

	# results_list = np.asarray(results_list)
	# results_list = results_list.flatten()
	# results_list = np.array(results_list).tolist()
	# results_list_df = pd.DataFrame(results_list)
	# results_list_df.to_csv('winner_model_all_per_county.csv', index=False)

	return results


def choose_best_model(moving_average_results_all_country_df, moving_average_results_per_state_df, moving_average_results_per_county_df):

	global moving_average_data

	best_h, best_c, mode = choose_winner_model(moving_average_results_all_country_df, moving_average_results_per_state_df, 
												moving_average_results_per_county_df)
	
	print("*********** best_h = " + str(best_h) + " ***********")

	normal_data = makeHistoricalData(fixed_data, temporal_data, best_h, r, 'death', 'mrmr', 'country', 'regular')
	# moving_avg_data = moving_average_data[best_h]
	moving_avg_data = makeHistoricalData(fixed_data, temporal_data, best_h, r, 'death', 'mrmr', 'country', 'weeklymovingaverage')
	cumulative_data = makeHistoricalData(fixed_data, temporal_data, best_h, r, 'death', 'mrmr', 'country', 'cumulative')

	if mode == 0:   # all country
		print("Running the winner model with normal method")
		nonrmal_model_result = winner_model_all_country(normal_data, best_c)
		print("Running the winner model with moving average method")
		moving_avg_model_result = winner_model_all_country(moving_avg_data, best_c)
		print("Running the winner model with cumulative method")
		cumulatove_model_result = winner_model_all_country(cumulative_data, best_c)

		results = [{"nonrmal": nonrmal_model_result, "moving_avg": moving_avg_model_result, "cumulative": cumulatove_model_result}]
		results = np.asarray(results)
		results = results.flatten()
		results = np.array(results).tolist()
		results_df = pd.DataFrame(results)
		results_df.to_csv('Results/winner_model_all_country.csv', index=False)

		temp_list = [nonrmal_model_result, moving_avg_model_result, cumulatove_model_result]

		best = temp_list.index(max(temp_list))

		if best == 0:
			return tuple((best_h, best_c, 0, mode))	# 0 shows best model should be implemented in normal method
		elif best == 1:
			return tuple((best_h, best_c, 1, mode))	# 1 shows best model should be implemented in moving average method
		elif best == 2:
			return tuple((best_h, best_c, 2, mode))	# 2 shows best model should be implemented in cumulative method

	elif mode == 1:     # per state
		print("Running the winner model with normal method")
		nonrmal_model_result = winner_model_per_state(normal_data, best_c)
		print("Running the winner model with moving average method")
		moving_avg_model_result = winner_model_per_state(moving_avg_data, best_c)
		print("Running the winner model with cumulative method")
		cumulatove_model_result = winner_model_per_state(cumulative_data, best_c)

		results = [{"nonrmal": statistics.mean(nonrmal_model_result), 
					"moving_avg": statistics.mean(moving_avg_model_result), 
					"cumulatove": statistics.mean(cumulatove_model_result)}]

		results = np.asarray(results)
		results = results.flatten()
		results = np.array(results).tolist()
		results_df = pd.DataFrame(results)
		results_df.to_csv('Results/winner_model_per_state.csv', index=False)

		temp_list = [statistics.mean(nonrmal_model_result), statistics.mean(moving_avg_model_result), 
						statistics.mean(cumulatove_model_result)]

		best = temp_list.index(max(temp_list))

		if best == 0:
			return tuple((best_h, best_c, 0, mode))	# 0 shows best model should be implemented in normal method
		elif best == 1:
			return tuple((best_h, best_c, 1, mode))	# 1 shows best model should be implemented in moving average method
		elif best == 2:
			return tuple((best_h, best_c, 2, mode))	# 2 shows best model should be implemented in cumulative method

	elif mode == 2:     # per county
		print("Running the winner model with normal method")
		nonrmal_model_result = winner_model_per_county(normal_data, best_c)
		print("Running the winner model with moving average method")
		moving_avg_model_result = winner_model_per_county(moving_avg_data, best_c)
		print("Running the winner model with cumulative method")
		cumulatove_model_result = winner_model_per_county(cumulative_data, best_c)

		temp_list = [statistics.mean(nonrmal_model_result), statistics.mean(moving_avg_model_result), 
						statistics.mean(cumulatove_model_result)]

		results = [{"nonrmal": statistics.mean(nonrmal_model_result), 
					"moving_avg": statistics.mean(moving_avg_model_result), 
					"cumulatove": statistics.mean(cumulatove_model_result)}]

		results = np.asarray(results)
		results = results.flatten()
		results = np.array(results).tolist()
		results_df = pd.DataFrame(results)
		results_df.to_csv('Results/winner_model_per_county.csv', index=False)

		best = temp_list.index(max(temp_list))

		if best == 0:
			return tuple((best_h, best_c, 0, mode))	# 0 shows best model should be implemented in normal method
		elif best == 1:
			return tuple((best_h, best_c, 1, mode))	# 1 shows best model should be implemented in moving average method
		elif best == 2:
			return tuple((best_h, best_c, 2, mode))	# 2 shows best model should be implemented in cumulative method


def model_type_one(dataset, c, dropout_value):
	dataset.reset_index(drop=True, inplace=True)
	numberOfSelectedCounties = len(dataset['county_fips'].unique())
	new_dataset = clean_data(dataset, numberOfSelectedCounties)
	X_train, y_train, X_val, y_val, X_test, y_test = preprocess(new_dataset)

	X_train.drop('date of day t', axis=1, inplace=True)
	X_val.drop('date of day t', axis=1, inplace=True)
	X_test.drop('date of day t', axis=1, inplace=True)

	y_train.drop('date of day t', axis=1, inplace=True)
	y_train.drop('county_fips', axis=1, inplace=True)
	y_val.drop('date of day t', axis=1, inplace=True)
	y_val.drop('county_fips', axis=1, inplace=True)
	y_test.drop('date of day t', axis=1, inplace=True)
	y_test.drop('county_fips', axis=1, inplace=True)

	numberOfCovariates = X_train.shape[1]
		
	train_X, train_y, val_X, val_y, test_X, test_y = part_of_data(X_train, y_train, X_val, y_val, X_test, y_test, c)
	train_X, train_y, val_X, val_y, test_X, test_y = data_normalize(train_X, train_y, val_X, val_y, test_X, test_y)
	
	model = Sequential()
	model.add(LSTM(4, return_sequences=True, input_shape=(1, train_X.shape[2])))
	model.add(LSTM(256, dropout=dropout_value, return_sequences=True))
	model.add(LSTM(128, dropout=dropout_value, return_sequences=True))
	model.add(LSTM(128))
	model.add(Dense(1, activation='selu'))

	# opt = tf.keras.optimizers.RMSprop(learning_rate=0.1)
	model.compile (
		loss='mean_squared_error', 
		optimizer=keras.optimizers.Adam(0.01), 
		# optimizer=keras.optimizers.SGD(lr=0.01, nesterov=True),
		metrics=['acc', tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanAbsolutePercentageError(), coeff_determination]
	)

	history = model.fit (
		train_X, train_y, 
		epochs=30,
		batch_size=128,
		validation_data=(val_X, val_y),
		verbose=1,
		shuffle=False
	)

	loss, accuracy, mae, mape, r2score = model.evaluate(val_X, val_y, verbose=0)

	K.clear_session()

	return r2score


def model_type_two(dataset, c, dropout_value):
	dataset.reset_index(drop=True, inplace=True)
	numberOfSelectedCounties = len(dataset['county_fips'].unique())
	new_dataset = clean_data(dataset, numberOfSelectedCounties)
	X_train, y_train, X_val, y_val, X_test, y_test = preprocess(new_dataset)

	X_train.drop('date of day t', axis=1, inplace=True)
	X_val.drop('date of day t', axis=1, inplace=True)
	X_test.drop('date of day t', axis=1, inplace=True)

	y_train.drop('date of day t', axis=1, inplace=True)
	y_train.drop('county_fips', axis=1, inplace=True)
	y_val.drop('date of day t', axis=1, inplace=True)
	y_val.drop('county_fips', axis=1, inplace=True)
	y_test.drop('date of day t', axis=1, inplace=True)
	y_test.drop('county_fips', axis=1, inplace=True)

	numberOfCovariates = X_train.shape[1]
		
	train_X, train_y, val_X, val_y, test_X, test_y = part_of_data(X_train, y_train, X_val, y_val, X_test, y_test, c)
	train_X, train_y, val_X, val_y, test_X, test_y = data_normalize(train_X, train_y, val_X, val_y, test_X, test_y)
	
	model = Sequential()
	model.add(LSTM(4, return_sequences=True, input_shape=(1, train_X.shape[2])))
	model.add(LSTM(256, dropout=dropout_value, return_sequences=True))
	model.add(LSTM(256, dropout=dropout_value, return_sequences=True))
	model.add(LSTM(128))
	model.add(Dense(1, activation='selu'))

	# opt = tf.keras.optimizers.RMSprop(learning_rate=0.1)
	model.compile (
		loss='mean_squared_error', 
		optimizer=keras.optimizers.Adam(0.01), 
		# optimizer=keras.optimizers.SGD(lr=0.01, nesterov=True),
		metrics=['acc', tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanAbsolutePercentageError(), coeff_determination]
	)

	history = model.fit (
		train_X, train_y, 
		epochs=30,
		batch_size=128,
		validation_data=(val_X, val_y),
		verbose=1,
		shuffle=False
	)

	loss, accuracy, mae, mape, r2score = model.evaluate(val_X, val_y, verbose=0)

	K.clear_session()

	return r2score


def model_type_three(dataset, c, dropout_value):
	dataset.reset_index(drop=True, inplace=True)
	numberOfSelectedCounties = len(dataset['county_fips'].unique())
	new_dataset = clean_data(dataset, numberOfSelectedCounties)
	X_train, y_train, X_val, y_val, X_test, y_test = preprocess(new_dataset)

	X_train.drop('date of day t', axis=1, inplace=True)
	X_val.drop('date of day t', axis=1, inplace=True)
	X_test.drop('date of day t', axis=1, inplace=True)

	y_train.drop('date of day t', axis=1, inplace=True)
	y_train.drop('county_fips', axis=1, inplace=True)
	y_val.drop('date of day t', axis=1, inplace=True)
	y_val.drop('county_fips', axis=1, inplace=True)
	y_test.drop('date of day t', axis=1, inplace=True)
	y_test.drop('county_fips', axis=1, inplace=True)

	numberOfCovariates = X_train.shape[1]
		
	train_X, train_y, val_X, val_y, test_X, test_y = part_of_data(X_train, y_train, X_val, y_val, X_test, y_test, c)
	train_X, train_y, val_X, val_y, test_X, test_y = data_normalize(train_X, train_y, val_X, val_y, test_X, test_y)
	
	model = Sequential()
	model.add(LSTM(4, return_sequences=True, input_shape=(1, train_X.shape[2])))
	model.add(LSTM(256, dropout=dropout_value, return_sequences=True))
	model.add(LSTM(256, dropout=dropout_value, return_sequences=True))
	model.add(LSTM(128, dropout=dropout_value, return_sequences=True))
	model.add(LSTM(128))
	model.add(Dense(1, activation='selu'))

	# opt = tf.keras.optimizers.RMSprop(learning_rate=0.1)
	model.compile (
		loss='mean_squared_error', 
		optimizer=keras.optimizers.Adam(0.01), 
		# optimizer=keras.optimizers.SGD(lr=0.01, nesterov=True),
		metrics=['acc', tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanAbsolutePercentageError(), coeff_determination]
	)

	history = model.fit (
		train_X, train_y, 
		epochs=30,
		batch_size=128,
		validation_data=(val_X, val_y),
		verbose=1,
		shuffle=False
	)

	loss, accuracy, mae, mape, r2score = model.evaluate(val_X, val_y, verbose=0)

	K.clear_session()

	return r2score


def model_type_four(dataset, c, dropout_value):
	dataset.reset_index(drop=True, inplace=True)
	numberOfSelectedCounties = len(dataset['county_fips'].unique())
	new_dataset = clean_data(dataset, numberOfSelectedCounties)
	X_train, y_train, X_val, y_val, X_test, y_test = preprocess(new_dataset)

	X_train.drop('date of day t', axis=1, inplace=True)
	X_val.drop('date of day t', axis=1, inplace=True)
	X_test.drop('date of day t', axis=1, inplace=True)

	y_train.drop('date of day t', axis=1, inplace=True)
	y_train.drop('county_fips', axis=1, inplace=True)
	y_val.drop('date of day t', axis=1, inplace=True)
	y_val.drop('county_fips', axis=1, inplace=True)
	y_test.drop('date of day t', axis=1, inplace=True)
	y_test.drop('county_fips', axis=1, inplace=True)

	numberOfCovariates = X_train.shape[1]
		
	train_X, train_y, val_X, val_y, test_X, test_y = part_of_data(X_train, y_train, X_val, y_val, X_test, y_test, c)
	train_X, train_y, val_X, val_y, test_X, test_y = data_normalize(train_X, train_y, val_X, val_y, test_X, test_y)
	
	model = Sequential()
	model.add(LSTM(4, return_sequences=True, input_shape=(1, train_X.shape[2])))
	model.add(LSTM(256, dropout=dropout_value, return_sequences=True))
	model.add(LSTM(256, dropout=dropout_value, return_sequences=True))
	model.add(LSTM(128, dropout=dropout_value, return_sequences=True))
	model.add(LSTM(128, dropout=dropout_value, return_sequences=True))
	model.add(LSTM(64))
	model.add(Dense(1, activation='selu'))

	# opt = tf.keras.optimizers.RMSprop(learning_rate=0.1)
	model.compile (
		loss='mean_squared_error', 
		optimizer=keras.optimizers.Adam(0.01), 
		# optimizer=keras.optimizers.SGD(lr=0.01, nesterov=True),
		metrics=['acc', tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanAbsolutePercentageError(), coeff_determination]
	)

	history = model.fit (
		train_X, train_y, 
		epochs=30,
		batch_size=128,
		validation_data=(val_X, val_y),
		verbose=1,
		shuffle=False
	)

	loss, accuracy, mae, mape, r2score = model.evaluate(val_X, val_y, verbose=0)

	K.clear_session()

	return r2score


def model_type_five(dataset, c, dropout_value):
	dataset.reset_index(drop=True, inplace=True)
	numberOfSelectedCounties = len(dataset['county_fips'].unique())
	new_dataset = clean_data(dataset, numberOfSelectedCounties)
	X_train, y_train, X_val, y_val, X_test, y_test = preprocess(new_dataset)

	X_train.drop('date of day t', axis=1, inplace=True)
	X_val.drop('date of day t', axis=1, inplace=True)
	X_test.drop('date of day t', axis=1, inplace=True)

	y_train.drop('date of day t', axis=1, inplace=True)
	y_train.drop('county_fips', axis=1, inplace=True)
	y_val.drop('date of day t', axis=1, inplace=True)
	y_val.drop('county_fips', axis=1, inplace=True)
	y_test.drop('date of day t', axis=1, inplace=True)
	y_test.drop('county_fips', axis=1, inplace=True)

	numberOfCovariates = X_train.shape[1]
		
	train_X, train_y, val_X, val_y, test_X, test_y = part_of_data(X_train, y_train, X_val, y_val, X_test, y_test, c)
	train_X, train_y, val_X, val_y, test_X, test_y = data_normalize(train_X, train_y, val_X, val_y, test_X, test_y)
	
	model = Sequential()
	model.add(LSTM(4, return_sequences=True, input_shape=(1, train_X.shape[2])))
	model.add(LSTM(256, dropout=dropout_value, return_sequences=True))
	model.add(LSTM(256, dropout=dropout_value, return_sequences=True))
	model.add(LSTM(128, dropout=dropout_value, return_sequences=True))
	model.add(LSTM(128, dropout=dropout_value, return_sequences=True))
	model.add(LSTM(64, dropout=dropout_value, return_sequences=True))
	model.add(LSTM(64))
	model.add(Dense(1, activation='selu'))

	# opt = tf.keras.optimizers.RMSprop(learning_rate=0.1)
	model.compile (
		loss='mean_squared_error', 
		optimizer=keras.optimizers.Adam(0.01), 
		# optimizer=keras.optimizers.SGD(lr=0.01, nesterov=True),
		metrics=['acc', tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanAbsolutePercentageError(), coeff_determination]
	)

	history = model.fit (
		train_X, train_y, 
		epochs=30,
		batch_size=128,
		validation_data=(val_X, val_y),
		verbose=1,
		shuffle=False
	)

	loss, accuracy, mae, mape, r2score = model.evaluate(val_X, val_y, verbose=0)

	K.clear_session()

	return r2score


def choose_number_of_layers(h, c, method, mode):	# level 8 in flowchart
	# method is in ['normal', 'mvg avg', 'cumulative']
	# mode is in ['all country', 'per state', 'per county']
	global moving_average_data

	best = 0	# return value of function that says which type of model is better
	data = pd.DataFrame()

	if method == 0:	# normal method
		data = makeHistoricalData(fixed_data, temporal_data, h, r, 'death', 'mrmr', 'country', 'regular')
	elif method == 1:	# moving average method
		data = makeHistoricalData(fixed_data, temporal_data, h, r, 'death', 'mrmr', 'country', 'weeklymovingaverage')
		# data = moving_avg_data = moving_average_data[h]
	elif method == 2:	#cumulative method
		data = makeHistoricalData(fixed_data, temporal_data, h, r, 'death', 'mrmr', 'country', 'cumulative')

	if mode == 0:	# all country mode
		result_type_one = model_type_one(data, c, 0.2)
		result_type_two = model_type_two(data, c, 0.2)
		result_type_three = model_type_three(data, c, 0.2)
		result_type_four = model_type_four(data, c, 0.2)
		result_type_five = model_type_five(data, c, 0.2)

		results_list = {"result_type_one": result_type_one, 
						"result_type_two": result_type_two, 
						"result_type_three": result_type_three, 
						"result_type_four": result_type_four, 
						"result_type_five": result_type_five}
		results_list = np.asarray(results_list)
		results_list = results_list.flatten()
		results_list = np.array(results_list).tolist()
		results_list_df = pd.DataFrame(results_list)
		results_list_df.to_csv('Results/model_types_score(all_country).csv', index=False)

		temp_list = [result_type_one, result_type_two, result_type_three, result_type_four, result_type_five]
		best = temp_list.index(max(temp_list))

	if mode == 1:	# per state mode
		states = data.state_fips.unique()

		with mp.Pool(mp.cpu_count()) as pool:
			results_type_one = pool.starmap(model_type_one, [(data.loc[data['state_fips'] == i], c, 0.2) for i in states])

		with mp.Pool(mp.cpu_count()) as pool:
			results_type_two = pool.starmap(model_type_two, [(data.loc[data['state_fips'] == i], c, 0.2) for i in states])

		with mp.Pool(mp.cpu_count()) as pool:
			results_type_three = pool.starmap(model_type_three, [(data.loc[data['state_fips'] == i], c, 0.2) for i in states])

		with mp.Pool(mp.cpu_count()) as pool:
			results_type_four = pool.starmap(model_type_four, [(data.loc[data['state_fips'] == i], c, 0.2) for i in states])

		with mp.Pool(mp.cpu_count()) as pool:
			results_type_five = pool.starmap(model_type_five, [(data.loc[data['state_fips'] == i], c, 0.2) for i in states])

		results_list = {"result_type_one": statistics.mean(results_type_one), 
						"result_type_two": statistics.mean(results_type_two), 
						"result_type_three": statistics.mean(results_type_three), 
						"result_type_four": statistics.mean(results_type_four), 
						"result_type_five": statistics.mean(results_type_five)}
		results_list = np.asarray(results_list)
		results_list = results_list.flatten()
		results_list = np.array(results_list).tolist()
		results_list_df = pd.DataFrame(results_list)
		results_list_df.to_csv('Results/model_types_score(per_state).csv', index=False)

		temp_list = [statistics.mean(results_type_one), statistics.mean(results_type_two), statistics.mean(results_type_three), 
						statistics.mean(results_type_four), statistics.mean(results_type_five)]

		best = temp_list.index(max(temp_list))

	if mode == 2:	# per county mode
		counties = data.county_fips.unique()

		with mp.Pool(mp.cpu_count()) as pool:
			results_type_one = pool.starmap(model_type_one, [(data.loc[data['county_fips'] == i], c, 0.2) for i in counties])

		with mp.Pool(mp.cpu_count()) as pool:
			results_type_two = pool.starmap(model_type_two, [(data.loc[data['county_fips'] == i], c, 0.2) for i in counties])

		with mp.Pool(mp.cpu_count()) as pool:
			results_type_three = pool.starmap(model_type_three, [(data.loc[data['county_fips'] == i], c, 0.2) for i in counties])

		with mp.Pool(mp.cpu_count()) as pool:
			results_type_four = pool.starmap(model_type_four, [(data.loc[data['county_fips'] == i], c, 0.2) for i in counties])

		with mp.Pool(mp.cpu_count()) as pool:
			results_type_five = pool.starmap(model_type_five, [(data.loc[data['county_fips'] == i], c, 0.2) for i in counties])

		temp_list = [statistics.mean(results_type_one), statistics.mean(results_type_two), statistics.mean(results_type_three), 
						statistics.mean(results_type_four), statistics.mean(results_type_five)]

		results_list = {"result_type_one": statistics.mean(results_type_one), 
						"result_type_two": statistics.mean(results_type_two), 
						"result_type_three": statistics.mean(results_type_three), 
						"result_type_four": statistics.mean(results_type_four), 
						"result_type_five": statistics.mean(results_type_five)}
		results_list = np.asarray(results_list)
		results_list = results_list.flatten()
		results_list = np.array(results_list).tolist()
		results_list_df = pd.DataFrame(results_list)
		results_list_df.to_csv('Results/model_types_score(per_county).csv', index=False)

		best = temp_list.index(max(temp_list))

	return best


def choose_dropout_value(h, c, method, mode, number_of_layers):
	# method is in ['normal', 'mvg avg', 'cumulative']
	# mode is in ['all country', 'per state', 'per county']
	global moving_average_data

	data = pd.DataFrame()

	if method == 0:	# normal method
		data = makeHistoricalData(fixed_data, temporal_data, h, r, 'death', 'mrmr', 'country', 'regular')
	elif method == 1:	# moving average method
		data = makeHistoricalData(fixed_data, temporal_data, h, r, 'death', 'mrmr', 'country', 'weeklymovingaverage')
		# data = moving_avg_data = moving_average_data[h]
	elif method == 2:	#cumulative method
		data = makeHistoricalData(fixed_data, temporal_data, h, r, 'death', 'mrmr', 'country', 'cumulative')

	if mode == 0:	# all country mode
		results = []
		
		if number_of_layers == 0:
			with mp.Pool(mp.cpu_count()) as pool:
				results = pool.starmap(model_type_one, [(data, c, d/10) for d in range(1, 5)])

		if number_of_layers == 1:
			with mp.Pool(processes = mp.cpu_count()) as pool:
				results = pool.starmap(model_type_two, [(data, c, d/10) for d in range(1, 5)])

		if number_of_layers == 2:
			with mp.Pool(processes = mp.cpu_count()) as pool:
				results = pool.starmap(model_type_three, [(data, c, d/10) for d in range(1, 5)])

		if number_of_layers == 3:
			with mp.Pool(processes = mp.cpu_count()) as pool:
				results = pool.starmap(model_type_four, [(data, c, d/10) for d in range(1, 5)])

		if number_of_layers == 4:
			with mp.Pool(processes = mp.cpu_count()) as pool:
				results = pool.starmap(model_type_five, [(data, c, d/10) for d in range(1, 5)])

		best_index = results.index(max(results))
		return best_index

	if mode == 1:	# per state mode
		results = []
		states = data.state_fips.unique()

		if number_of_layers == 0:
			with mp.Pool(mp.cpu_count()) as pool:
				results = pool.starmap(model_type_one, [(data.loc[data['state_fips'] == i], c, d/10) 
															for i in states for d in range(1, 5)])

		if number_of_layers == 1:
			with mp.Pool(mp.cpu_count()) as pool:
				results = pool.starmap(model_type_two, [(data.loc[data['state_fips'] == i], c, d/10) 
															for i in states for d in range(1, 5)])

		if number_of_layers == 2:
			with mp.Pool(mp.cpu_count()) as pool:
				results = pool.starmap(model_type_three, [(data.loc[data['state_fips'] == i], c, d/10) 
															for i in states for d in range(1, 5)])

		if number_of_layers == 3:
			with mp.Pool(mp.cpu_count()) as pool:
				results = pool.starmap(model_type_four, [(data.loc[data['state_fips'] == i], c, d/10) 
															for i in states for d in range(1, 5)])

		if number_of_layers == 4:
			with mp.Pool(mp.cpu_count()) as pool:
				results = pool.starmap(model_type_five, [(data.loc[data['state_fips'] == i], c, d/10) 
															for i in states for d in range(1, 5)])

		best_index = results.index(max(results)) % 4
		return best_index

	if mode == 2:	# per county mode
		results = []
		counties = data.county_fips.unique()

		if number_of_layers == 0:
			with mp.Pool(mp.cpu_count()) as pool:
				results = pool.starmap(model_type_one, [(data.loc[data['county_fips'] == i], c, d/10) 
															for i in counties for d in range(1, 5)])

		if number_of_layers == 1:
			with mp.Pool(mp.cpu_count()) as pool:
				results = pool.starmap(model_type_two, [(data.loc[data['county_fips'] == i], c, d/10) 
															for i in counties for d in range(1, 5)])

		if number_of_layers == 2:
			with mp.Pool(mp.cpu_count()) as pool:
				results = pool.starmap(model_type_three, [(data.loc[data['county_fips'] == i], c, d/10) 
															for i in counties for d in range(1, 5)])

		if number_of_layers == 3:
			with mp.Pool(mp.cpu_count()) as pool:
				results = pool.starmap(model_type_four, [(data.loc[data['county_fips'] == i], c, d/10) 
															for i in counties for d in range(1, 5)])

		if number_of_layers == 4:
			with mp.Pool(mp.cpu_count()) as pool:
				results = pool.starmap(model_type_five, [(data.loc[data['county_fips'] == i], c, d/10) 
															for i in counties for d in range(1, 5)])

		best_index = results.index(max(results)) % 4
		return best_index


def main():	
	############################################################################### level 1, 2
	# constructing moving average data with 'h' from 1 to 21
	# parallelized on 'h'
	print("Started to make historical data (moving average method)")
	start_time = time.time()
	with mp.Pool(mp.cpu_count()) as pool:
		moving_average_data = pool.starmap(makeHistoricalData, 
			[(fixed_data, temporal_data, h, r, 'death', 'mrmr', 'country', 'weeklymovingaverage') for h in range(1, H+1)])

	print("Making historical data is done!")
	print("--- %s seconds to construct moving average data ---" % (time.time() - start_time))
	###############################################################################

	############################################################################### level 3, 4
	# training model on all country with moving average data with 'h' from 1 to 21
	# parallelized on 'h'
	print("Started to train model on moving average data and on all country")
	print("--- level 3 in flowchart ---")
	start_time = time.time()
	with mp.Pool(mp.cpu_count()) as pool:
		moving_average_results_all_country = pool.starmap(fit_model_all_country, [(moving_average_data[h], h) for h in range(0, H)])

	end_time = time.time()
	print("Finished training hte model on moving average data and on all country")
	print("--- %s seconds to train the model on all country ---" % (end_time - start_time))

	moving_average_results_all_country = np.asarray(moving_average_results_all_country)
	moving_average_results_all_country = moving_average_results_all_country.flatten()
	moving_average_results_all_country = np.array(moving_average_results_all_country).tolist()
	moving_average_results_all_country_df = pd.DataFrame(moving_average_results_all_country)

	# writing all country results into a file
	moving_average_results_all_country_df.to_csv('Results/moving_average_results_all_country.csv', index=False)
	with open('Results/moving_average_results_all_country_time.txt', 'w') as fp:
		fp.write("--- %s seconds to train the model on all country ---" % (end_time - start_time))
	###############################################################################

	############################################################################### level 3, 4
	# training model per state with moving average data with 'h' from 1 to 21
	# parallelized on 'state'
	print("Started to train model on moving average data and per state")
	print("--- level 3 in flowchart ---")
	start_time = time.time()
	with mp.Pool(mp.cpu_count()) as pool:
		moving_average_results_per_state = pool.starmap(fit_model_per_state, [(moving_average_data[h], state_fips, h) 
														for h in range(0, H) for state_fips in moving_average_data[h].state_fips.unique()])
	end_time = time.time()
	print("Finished training hte model on moving average data and per state")
	print("--- %s seconds to train the model per state ---" % (end_time - start_time))

	moving_average_results_per_state = np.asarray(moving_average_results_per_state)
	moving_average_results_per_state = moving_average_results_per_state.flatten()
	moving_average_results_per_state = np.array(moving_average_results_per_state).tolist()
	moving_average_results_per_state_df = pd.DataFrame(moving_average_results_per_state)

	# writing per state results into a file
	moving_average_results_per_state_df.to_csv('Results/moving_average_results_per_state.csv', index=False)
	with open('Results/moving_average_results_per_state_time.txt', 'w') as fp:
		fp.write("--- %s seconds to train the model per state ---" % (end_time - start_time))
	###############################################################################

	############################################################################### level 3, 4
	# training model per county with moving average data with 'h' from 1 to 21
	# parallelized on 'h'
	print("Started to train model on moving average data and per county")
	print("--- level 3 in flowchart ---")
	start_time = time.time()
	with mp.Pool(mp.cpu_count()) as pool:
		moving_average_results_per_county = pool.starmap(fit_model_per_county, [(moving_average_data[h], h) for h in range(0, H)])

	end_time = time.time()
	print("Finished training hte model on moving average data and per county")
	print("--- %s seconds to train the model per county ---" % (time.time() - start_time))

	moving_average_results_per_county = np.asarray(moving_average_results_per_county)
	moving_average_results_per_county = moving_average_results_per_county.flatten()
	moving_average_results_per_county = np.array(moving_average_results_per_county).tolist()
	moving_average_results_per_county_df = pd.DataFrame(moving_average_results_per_county)
	
	# writing per county results into a file
	moving_average_results_per_county_df.to_csv('Results/moving_average_results_per_county.csv', index=False)
	with open('Results/moving_average_results_per_county_time.txt', 'w') as fp:
		fp.write("--- %s seconds to train the model per county ---" % (end_time - start_time))
	###############################################################################

	############################################################################### level 5, 6, 7
	# choosing a method for the best model
	print("Started to choose the best model")
	print("--- level 7 in flowchart ---")
	start_time = time.time()
	best_h, best_c, best_method, best_mode = choose_best_model(moving_average_results_all_country_df, moving_average_results_per_state_df, moving_average_results_per_county_df)
	end_time = time.time()
	print("Finished choosing the best model")
	print("--- %s seconds to train the models and choose the best model ---" % (end_time - start_time))
	methods = ['normal', 'moving average', 'cumulative']
	print("best method chosen for training is " + methods[best_method])

	with open('Results/choosing_best_model.txt', 'w') as fp:
		fp.write("best method chosen for training is " + methods[best_method])
		fp.write("\n--- %s seconds to train the models and choose the best model ---" % (end_time - start_time))
	###############################################################################

	############################################################################### level 8
	# choosing best number of layers
	start_time = time.time()
	model_type = choose_number_of_layers(best_h, best_c, best_method, best_mode)
	end_time = time.time()
	
	print("Finished choosing the number of layers in best model")
	print("--- %s seconds to choose the number of layers in best model ---" % (end_time - start_time))
	types = ['type_one', 'type_two', 'type_three', 'type_four', 'type_five']
	print("number of layers for best model is based on " + types[model_type])

	with open('Results/number_of_layers.txt', 'w') as fp:
		fp.write("number of layers for best model is based on " + types[model_type])
		fp.write("\n--- %s seconds to choose the number of layers in best model ---" % (end_time - start_time))
	###############################################################################

	############################################################################### level 9
	# choosing best value for dropout of model layers in the range of [0.1, 0.4]
	start_time = time.time()
	best_dropout_value = choose_dropout_value(best_h, best_c, best_method, best_mode, model_type)
	end_time = time.time()

	print("Finished choosing the best value for drouput in best model")
	print("--- %s seconds to choose the dropout value in best model ---" % (end_time - start_time))
	dropout_values = [0.1, 0.2, 0.3, 0.4]
	print("best dropout value in the best model is " + str(dropout_values[best_dropout_value]))

	with open('Results/dropout_value.txt', 'w') as fp:
		fp.write("best value for dropout in the best model is " + str(dropout_values[best_dropout_value]))
		fp.write("\n--- %s seconds to choose the best dropout value in best model ---" % (end_time - start_time))
	###############################################################################

	###############################################################################
	# and finally storing all the variables needed for implementing the final model in a txt file
	methods = ['normal', 'moving average', 'cumulative']
	modes = ['all country', 'per state', 'per county']
	types = ['type_one', 'type_two', 'type_three', 'type_four', 'type_five']
	dropout_values = [0.1, 0.2, 0.3, 0.4]

	with open('Results/final_model.txt', 'w') as fp:
		fp.write("best method : " + methods[best_method])
		fp.write("\nbest mode : " + modes[best_mode])
		fp.write("\nbest value for h : " + str(best_h))
		fp.write("\nbest value for c : " + str(best_c))
		fp.write("\nnumber of layers is based on : " + types[model_type])
		fp.write("\ndropout value : " + str(dropout_values[best_dropout_value]))
	###############################################################################

	print("\n\n**************** Code execution completed successfully! ****************\n\n")


if __name__ == "__main__":
	main()