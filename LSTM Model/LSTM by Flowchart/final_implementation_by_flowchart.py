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
import time
import math
import statistics

r = 14
numberOfSelectedCounties = -1
temporal_data = pd.read_csv('temporal-data.csv')
fixed_data = pd.read_csv('fixed-data.csv')

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
	from keras import backend as K
	SS_res =  K.sum(K.square( y_true-y_pred ))
	SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
	return ( 1 - SS_res/(SS_tot + K.epsilon()) )


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
		metrics=['acc', coeff_determination]
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
		print('c = ' + str(c) + ', h = ' + str(h))
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

		loss, accuracy, R2_SCORE = model.evaluate(val_X, val_y, verbose=0)

		results.append(tuple((h, c, R2_SCORE)))

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

		loss, accuracy, R2_SCORE = model.evaluate(val_X, val_y, verbose=0)

		results.append(tuple((h, c, state_fips, R2_SCORE)))

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

			loss, accuracy, R2_SCORE = model.evaluate(val_X, val_y, verbose=0)

			results.append(tuple((h, c, county, R2_SCORE)))

	return results


def get_best_model_all_country(moving_average_results_all_country):	# each element is a tuple of (h, c, R2_SCORE)
	best_R2SCORE = -math.inf
	best_h = 0
	best_c = 0

	for element in moving_average_results_all_country:
		h, c, R2_SCORE = element
		if R2_SCORE > best_R2SCORE:
			best_R2SCORE = R2_SCORE
			best_h = h
			best_c = c

	return tuple((best_h, best_c, best_R2SCORE))


def get_best_model_per_state(moving_average_results_per_state):	# each element is a tuple of (h, c, state_fips, R2_SCORE)
	best_R2SCORE = -math.inf
	best_h = 0
	best_c = 0

	new_results = []

	for element in moving_average_results_per_state:
		h, c, state_fips, R2_SCORE = element
		if R2_SCORE > best_R2SCORE:
			best_R2SCORE = R2_SCORE
			best_h = h
			best_c = c

	return tuple((best_h, best_c, best_R2SCORE))


def get_best_model_per_county(moving_average_results_per_county):	# each element is a tuple of (h, c, county_fips, R2_SCORE)
	best_R2SCORE = -math.inf
	best_h = 0
	best_c = 0

	for element in moving_average_results_per_county:
		h, c, state_fips, R2_SCORE = element
		if R2_SCORE > best_R2SCORE:
			best_R2SCORE = R2_SCORE
			best_h = h
			best_c = c

	return tuple((best_h, best_c, best_R2SCORE))


def choose_winner_model(moving_average_results_all_country, moving_average_results_per_state, moving_average_results_per_county):

	best_h_all_country, best_c_all_country, R2_SCORE_all_country = get_best_model_all_country(moving_average_results_all_country)
	best_h_per_state, best_c_per_state, R2_SCORE_per_state = get_best_model_per_state(moving_average_results_per_state)
	best_h_per_county, best_c_per_county, R2_SCORE_per_county = get_best_model_per_county(moving_average_results_per_county)

	temp_list = [R2_SCORE_all_country, R2_SCORE_per_state, R2_SCORE_per_county]

	best = temp_list.index(max(temp_list))

	if best == 0:
		return tuple((best_h_all_country, best_c_all_country, 0)) # 0 shows winner model should be implemented in all country mode
	elif best == 1:
		return tuple((best_h_per_state, best_c_per_state, 1))   # 1 shows winner model should be implemented in per state mode
	elif best == 2:
		return tuple((best_h_per_county, best_c_per_county, 2)) # 2 shows winner model should be implemented in per county mode


def winner_model_all_country(dataset, c):	# when mode = 0 in 'choose_winner_model' function
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

	loss, accuracy, R2_SCORE = model.evaluate(val_X, val_y, verbose=0)

	results.append(tuple((R2_SCORE)))

	return results


def winner_model_per_state(dataset, c):	# when mode = 1 in 'choose_winner_model' function

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

		loss, accuracy, R2_SCORE = model.evaluate(val_X, val_y, verbose=0)

		return R2_SCORE
	############################################################################

	states = dataset.state_fips.unique()
	with mp.Pool(mp.cpu_count()) as pool:
		results = pool.starmap(fit_model_on_state, [(dataset, state_fips, c) for state_fips in states])

	return results


def winner_model_per_county(dataset, c):	# when mode = 2 in 'choose_winner_model' function
	
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

		loss, accuracy, R2_SCORE = model.evaluate(val_X, val_y, verbose=0)

		results.append(R2_SCORE)

	return results


def choose_best_model(moving_average_results_all_country, moving_average_results_per_state, moving_average_results_per_county):

	best_h, best_c, mode = choose_winner_model(moving_average_results_all_country, moving_average_results_per_state, 
												moving_average_results_per_county)

	normal_data = makeHisttoricalData(fixed_data, temporal_data, best_h, r, 'death', 'mrmr', 'country', 'regular')
	moving_avg_data = moving_average_data[best_h]
	cumulative_data = makeHisttoricalData(fixed_data, temporal_data, best_h, r, 'death', 'mrmr', 'country', 'cumulative')

	if mode == 0:   # all country
		nonrmal_model_result = winner_model_all_country(normal_data, best_c)
		moving_avg_model_result = winner_model_all_country(moving_avg_data, c)
		cumulatove_model_result = winner_model_all_country(cumulative_data, c)

		temp_list = [nonrmal_model_result, moving_avg_model_result, cumulatove_model_result]

		best = temp_list.index(max(temp_list))

		if best == 0:
			return 0        # 0 shows best model should be implemented in normal method
		elif best == 1:
			return 1        # 1 shows best model should be implemented in moving average method
		elif best == 2:
			return 2        # 2 shows best model should be implemented in cumulative method

	elif mode == 1:     # per state
		nonrmal_model_result = winner_model_per_state(normal_data, best_c)
		moving_avg_model_result = winner_model_per_state(moving_avg_data, c)
		cumulatove_model_result = winner_model_per_state(cumulative_data, c)

		temp_list = [statistics.mean(nonrmal_model_result), statistics.mean(moving_avg_model_result), 
						statistics.mean(cumulatove_model_result)]

		best = temp_list.index(max(temp_list))

		if best == 0:
			return 0        # 0 shows best model should be implemented in normal method
		elif best == 1:
			return 1        # 1 shows best model should be implemented in moving average method
		elif best == 2:
			return 2        # 2 shows best model should be implemented in cumulative method

	elif mode == 2:     # per state
		nonrmal_model_result = winner_model_per_county(normal_data, best_c)
		moving_avg_model_result = winner_model_per_county(moving_avg_data, c)
		cumulatove_model_result = winner_model_per_county(cumulative_data, c)

		temp_list = [statistics.mean(nonrmal_model_result), statistics.mean(moving_avg_model_result), 
						statistics.mean(cumulatove_model_result)]

		best = temp_list.index(max(temp_list))

		if best == 0:
			return 0        # 0 shows best model should be implemented in normal method
		elif best == 1:
			return 1        # 1 shows best model should be implemented in moving average method
		elif best == 2:
			return 2        # 2 shows best model should be implemented in cumulative method


def main():
	# constructing moving average data with 'h' from 1 to 21
	# parallelized on 'h'
	start_time = time.time()
	with mp.Pool(mp.cpu_count()) as pool:
	moving_average_data = pool.starmap(makeHistoricalData, 
		[(fixed_data, temporal_data, h, r, 'death', 'mrmr', 'country', 'weeklymovingaverage') for h in range(1, 22)])

	print("--- %s seconds to construct moving average data ---" % (time.time() - start_time))

	# training model on all country with moving average data with 'h' from 1 to 21
	# parallelized on 'h'
	start_time = time.time()
	with mp.Pool(mp.cpu_count()) as pool:
		moving_average_results_all_country = pool.starmap(fit_model_all_country, [(moving_average_data[h], h) for h in range(0, 21)])

	end_time = time.time()
	print("--- %s seconds to train the model on all country ---" % (end_time - start_time))

	# writing all country results into a file
	with open('moving_average_results_per_state.txt', 'w') as fp:
		fp.write('h\tc\tR2_SCORE\n')
		fp.write('\n'.join('%s\t%s\t%s' % x for x in moving_average_results_all_country))
		fp.write("\n\n--- %s seconds to train the model on all country ---" % (end_time - start_time))

	# training model per state with moving average data with 'h' from 1 to 21
	# parallelized on 'state'
	start_time = time.time()
	for h in range(0, 21):
		dataset = moving_average_data[h]
		states = dataset.state_fips.unique()
		with mp.Pool(mp.cpu_count()) as pool:
			moving_average_results_per_state = pool.starmap(fit_model_per_state, [(dataset, state_fips, h) for state_fips in states])

	end_time = time.time()
	print("--- %s seconds to train the model per state ---" % (end_time - start_time))

	# writing per state results into a file
	with open('moving_average_results_per_state.txt', 'w') as fp:
		fp.write('h\tc\tstate_fips\tR2_SCORE\n')
		fp.write('\n'.join('%s\t%s\t%s\t%s' % x for x in moving_average_results_per_state))
		fp.write("\n\n--- %s seconds to train the model per state ---" % (end_time - start_time))
	
	# training model per county with moving average data with 'h' from 1 to 21
	# parallelized on 'h'
	start_time = time.time()
	with mp.Pool(mp.cpu_count()) as pool:
		moving_average_results_per_county = pool.starmap(fit_model_per_county, [(moving_average_data[h], h) for h in range(0, 21)])

	end_time = time.time()
	print("--- %s seconds to train the model per county ---" % (time.time() - start_time))

	# writing per county resluts into a file
	with open('moving_average_results_per_county.txt', 'w') as fp:
		fp.write('h\tc\tcounty_fips\tR2_SCORE')
		fp.write('\n'.join('%s\t%s\t%s\t%s' % x for x in moving_average_results_per_county))
		fp.write("\n\n--- %s seconds to train the model per county ---" % (end_time - start_time))

	# choosing a method for the best model
	start_time = time.time()
	best_method_for_training = choose_best_model(moving_average_results_all_country, moving_average_results_per_state, 
													moving_average_results_per_county)
	end_time = time.time()
	print("--- %s seconds to train the models and choose the best model ---" % (end_time - start_time))


if __name__ == "__main__":
	main()