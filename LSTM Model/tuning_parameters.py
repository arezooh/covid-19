# Tuning Parameters of LSTM Model using Keras
# Requirements : pip install keras-tuner

import pandas as pd
import numpy as np
from makeHistoricalData import makeHistoricalData
from sklearn import preprocessing
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras import backend as K
from kerastuner import HyperModel

r = 14
h = 5
numberOfSelectedCounties = -1
fixed_data = pd.read_csv('fixed-data.csv')
temporal_data = pd.read_csv('temporal-data.csv')

def splitData(numberOfCounties, main_data, target):

	if numberOfCounties == -1:
		numberOfCounties = len(main_data['county_fips'].unique())

	main_data = main_data.sort_values(by=['date of day t' , 'county_fips'])
	target = target.sort_values(by=['date of day t' , 'county_fips'])
	X_train_train = main_data.iloc[:-2*(r*numberOfCounties),:].sort_values(by=['county_fips' , 'date of day t'])
	X_train_val = main_data.iloc[-2*(r*numberOfCounties):-(r*numberOfCounties),:].sort_values(by=['county_fips' , 'date of day t'])
	X_test = main_data.tail(r*numberOfCounties).sort_values(by=['county_fips' , 'date of day t'])

	y_train_train = target.iloc[:-2*(r*numberOfCounties),:].sort_values(by=['county_fips' , 'date of day t'])
	y_train_val = target.iloc[-2*(r*numberOfCounties):-(r*numberOfCounties),:].sort_values(by=['county_fips' , 'date of day t'])
	y_test = target.tail(r*numberOfCounties).sort_values(by=['county_fips' , 'date of day t'])

	val_naive_pred = target.iloc[-3*(r*numberOfCounties):-2*(r*numberOfCounties),:].sort_values(by=['county_fips' , 'date of day t'])
	test_naive_pred = target.iloc[-2*(r*numberOfCounties):-(r*numberOfCounties),:].sort_values(by=['county_fips' , 'date of day t'])

	return X_train_train , X_train_val , X_test , y_train_train , y_train_val , y_test, val_naive_pred, test_naive_pred


def clean_data(data, numberOfSelectedCounties):
	global numberOfDays
	data = data.sort_values(by=['county_fips', 'date of day t'])
	# select the number of counties we want to use
	# numberOfSelectedCounties = numberOfCounties
	if numberOfSelectedCounties == -1:
		numberOfSelectedCounties = len(data['county_fips'].unique())

	using_data = data[(data['county_fips'] <= data['county_fips'].unique()[numberOfSelectedCounties - 1])]
	using_data = using_data.reset_index(drop=True)
	main_data = using_data.drop(['county_name', 'state_fips', 'state_name'], axis=1)  # , 'date of day t'
	numberOfDays = len(using_data['date of day t'].unique())

	return main_data


def preprocess(main_data):

	target = pd.DataFrame(main_data[['date of day t', 'county_fips', 'Target']])
	main_data = main_data.drop(['Target'], axis=1)
	# specify the size of train, validation and test sets
	# produce train, validation and test data in parallel

	X_train_train , X_train_val , X_test , y_train_train , y_train_val , y_test, val_naive_pred, test_naive_pred = splitData(numberOfSelectedCounties, main_data, target)

	y_train_date = y_train_train.copy()
	y_test_date = y_test.copy()
	y_val_date = y_train_val.copy()

	X_train_train.drop('date of day t', axis=1, inplace=True)
	X_train_val.drop('date of day t', axis=1, inplace=True)
	X_test.drop('date of day t', axis=1, inplace=True)

	y_train_train.drop(['date of day t', 'county_fips'], axis=1, inplace=True)
	y_train_val.drop(['date of day t', 'county_fips'], axis=1, inplace=True)
	y_test.drop(['date of day t', 'county_fips'], axis=1, inplace=True)

	val_naive_pred.drop(['date of day t', 'county_fips'], axis=1, inplace=True)
	test_naive_pred.drop(['date of day t', 'county_fips'], axis=1, inplace=True)

	val_naive_pred = val_naive_pred.values.flatten()
	test_naive_pred = test_naive_pred.values.flatten()

	return X_train_train, y_train_train, X_train_val, y_train_val, X_test, y_test, y_train_date, y_test_date, y_val_date, val_naive_pred, test_naive_pred


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

	return X_train, y_train, X_val, y_val, X_test, y_test, scalar


def part_of_data(X_train, y_train, X_val, y_val, X_test, y_test, c):
	train_X = X_train.iloc[:, 0:c].copy()
	train_y = y_train.copy()

	val_X = X_val.iloc[:, 0:c].copy()
	val_y = y_val.copy()

	test_X = X_test.iloc[:, 0:c].copy()
	test_y = y_test.copy()

	return train_X, train_y, val_X, val_y, test_X, test_y


class LSTMHyperModel(HyperModel):
	def __init__(self, n):	# n is the second parameter of input_shape
		self.n = n
		
	def build(self, hp):
		model = Sequential()
		
		# first layer
		model.add(
			LSTM(
				units=hp.Int('units_0', min_value=8, max_value=64, step=8, default=8),
				kernel_initializer=hp.Choice('kernel_initializer', 
											 values=['uniform', 'lecun_uniform', 'normal', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'], 
											 default="glorot_uniform"), 
				return_sequences=True, 
				input_shape=(1, self.n)
			)
		)
		
		# hidden layers (also the number of hidden layers will be suggested)
		for i in range(hp.Int('num_layers', 1, 8)):
			model.add(
				LSTM(
					units=hp.Int('units_'+str(i+1), min_value=16, max_value=256, step=16, default=32), 
					dropout=hp.Float('dropout_'+str(i+1), 0, 0.5, step=0.1, default=0.2), 
					return_sequences=True
				)
			)
		
		# last hidden layer
		model.add(
			LSTM(
				units=hp.Int('units_before_last_layer', min_value=16, max_value=256, step=16, default=32)
			)
		)
		
		# output layer
		model.add(
			Dense(
				units=1, 
				activation=hp.Choice(
					'dense_activation', 
					values=["softplus", "softsign", "tanh", "relu", "selu", "elu", "exponential"], 
					default="selu"
				)
			)
		)
		
		# compiling the model with choices for optimizer and loss function
		model.compile(
			optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', [1e-3, 1e-2, 1e-1, 1.0, 2.5, 5.0])), 
			loss=hp.Choice('loss_function', values=["mean_squared_error", "mean_squared_logarithmic_error", "mean_absolute_percentage_error", "cosine_similarity", 
													"huber_loss"], 
											default="mean_squared_error"), 
			metrics=['mse']
		)
		
		return model


from kerastuner import HyperModel
from kerastuner.tuners import RandomSearch

def main():

	dataset = makeHistoricalData(fixed_data, temporal_data, h, r, 'death', 'mrmr', 'country', 'regular')

	numberOfSelectedCounties = len(dataset['county_fips'].unique())
	new_dataset = clean_data(dataset, numberOfSelectedCounties)
	X_train, y_train, X_val, y_val, X_test, y_test, y_train_date, y_test_date, y_val_date, val_naive_pred, test_naive_pred = preprocess(new_dataset)
	X_train, y_train, X_val, y_val, X_test, y_test, scalar = data_normalize(X_train, y_train, X_val, y_val, X_test, y_test)

	hypermodel = LSTMHyperModel(n=X_train.shape[2])

	tuner = RandomSearch(
		hypermodel,
		objective='mse',
		seed=1,
		max_trials=60,
		executions_per_trial=4,
		directory='parameter_tuning',
		project_name='lstm_model_tuning'
	)

	tuner.search_space_summary()

	print()
	input("Press Enter to continue...")
	print()

	N_EPOCH_SEARCH = 50
	tuner.search(X_train, y_train, epochs=N_EPOCH_SEARCH, validation_split=0.2)

	print()
	input("Press Enter to show the summary of search...")
	print()

	# Show a summary of the search
	tuner.results_summary()

	print()
	input("Press Enter to retrive the best model...")
	print()

	# Retrieve the best model.
	best_model = tuner.get_best_models(num_models=1)[0]

	print()
	input("Press Enter to show best model summary...")
	print()

	best_model.summary()

	print()
	input("Press Enter to run the best model on test dataset...")
	print()

	# Evaluate the best model.
	loss, accuracy = best_model.evaluate(X_test, y_test)
	print("loss = " + str(loss) + ", acc = " + str(accuracy))


if __name__ == "__main__":
	main()
