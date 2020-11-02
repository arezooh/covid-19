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


from numpy.random import seed
seed(84156)
tf.random.set_seed(102487)

# mkdir for saving the results in
Path("Results").mkdir(parents=True, exist_ok=True)
Path("Step1").mkdir(parents=True, exist_ok=True)

r = 4
H = 16
target = 'confirmed'
test_size = None
fixed_data = pd.read_csv('fixed-data.csv')
temporal_data = pd.read_csv('integer-test-temporal-data.csv')
comparison_criteria = 'MAPE'

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

# def mean_absolute_percentage_error(y_true, y_pred):
# 	y_true, y_pred = np.array(y_true), np.array(y_pred)
# 	sumOfAbsoluteError = sum(abs(y_true - y_pred))
# 	mape = (sumOfAbsoluteError / sum(y_true)) * 100
# 	return mape

def mean_absolute_percentage_error(y_true, y_pred):
	return np.mean((abs(y_true - y_pred)/y_true)*100)


def MASE(y_true, y_pred, y_naive):
	mae_on_pred = mean_absolute_error(y_true, y_pred)
	mae_on_naive = mean_absolute_error(y_true, y_naive)

	return mae_on_pred/mae_on_naive


def normalize(X_train, y_train, X_test, y_test):
	scaler = preprocessing.StandardScaler()

	X_train = X_train.values
	X_train = scaler.fit_transform(X_train)

	# X_val = X_val.values
	# X_val = scaler.fit_transform(X_val)

	X_test = X_test.values
	X_test = scaler.fit_transform(X_test)

	y_train = y_train.values
	y_train = scaler.fit_transform(y_train.reshape(-1, 1))

	# y_val = y_val.values
	# y_val = scaler.fit_transform(y_val.reshape(-1, 1))

	y_test = y_test.values
	# y_test = scaler.fit_transform(y_test.reshape(-1, 1))

	X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
	# X_val = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))
	X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

	y_train = y_train.reshape((y_train.shape[0]), )
	# y_val = y_val.reshape((y_val.shape[0]), )
	y_test = y_test.reshape((y_test.shape[0]), )

	return X_train, y_train, X_test, y_test, scaler


def part_of_data(X_train, X_test, y_train, y_test, c):
	train_X = X_train.iloc[:, 0:c].copy()
	train_y = y_train.copy()

	# val_X = X_val.iloc[:, 0:c].copy()
	# val_y = y_val.copy()

	test_X = X_test.iloc[:, 0:c].copy()
	test_y = y_test.copy()

	return train_X, test_X, train_y,test_y


################################################ my new activation function :))		( activation = sin(x)/cosh(x) )
from keras.layers import Activation
from keras.utils.generic_utils import get_custom_objects

def custom1(x):
	return 2*K.sin(x) / (K.exp(x)+K.exp(-1*x))

get_custom_objects().update({'custom1': Activation(custom1)})
################################################


def default_model(n):
	model = Sequential()
	model.add(LSTM(8, kernel_initializer=tf.keras.initializers.Identity(), return_sequences=True, input_shape=(1, n)))
	model.add(Activation(custom1))

	model.add(LSTM(256, dropout=0.2, return_sequences=True))
	model.add(Activation(custom1))

	model.add(LSTM(256, dropout=0.2, return_sequences=True))
	model.add(Activation(custom1))

	model.add(LSTM(128, dropout=0.2, return_sequences=True))
	model.add(Activation(custom1))

	model.add(LSTM(128))
	model.add(Activation(custom1))

	model.add(Dense(1))
	model.add(Activation(custom1))


	model.compile (
	   loss=tf.keras.losses.MeanAbsoluteError(),
	   optimizer=keras.optimizers.Adam(0.00005),
	   metrics=[tf.keras.metrics.MeanAbsoluteError()]
	)

	return model


def fit_model(dataset, h):

	results = []

	# dataset.drop('county_fips', axis=1, inplace=True)
	# dataset.drop(dataset.tail(4).index, inplace=True)

	########################################### split data
	train = dataset.head(len(dataset)-7)
	test = dataset.tail(7)

	X_train = train.drop(['date of day t', 'Target'], axis=1)
	y_train = train['Target']

	X_test = test.drop(['date of day t', 'Target'], axis=1)
	y_test = test['Target']

	naive_pred = train['Target'].tail(4).append(test['Target'].head(3))
	########################################### end of split data

	limit = len(X_train)

	for c in range(1, limit):
		train_X, test_X, train_y, test_y = part_of_data(X_train, X_test, y_train, y_test, c)
		train_X, train_y, test_X, test_y, scaler = normalize(train_X, train_y, test_X, test_y)

		es = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1, patience=5)

		model = default_model(train_X.shape[2])

		model.fit (
			train_X, train_y,
			epochs=1000,
			batch_size=8,
			validation_split=0.4,
			verbose=0,
			# callbacks=[es], 
			shuffle=False
		)

		y_pred = model.predict(test_X)
		y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))
		
		y_pred = y_pred.reshape(-1)
		test_y = test_y.reshape(-1)
		naive = naive_pred.values.reshape(-1)

		# y_pred = np.round(y_pred.astype(np.double))
		# y_test = np.round(y_test.astype(np.double))
		# naive = np.round(naive.astype(np.double))
		
		mase = MASE(test_y, y_pred, naive)
		mae = mean_absolute_error(test_y, y_pred)
		mape = mean_absolute_percentage_error(test_y, y_pred)

		print("\n###############################################")
		print("h = " + str(h+1) + ", c = " + str(c) + " ( mae = " + str(mae) + ", mape = " + str(mape) + ", mase = " + str(mase) + " )")
		print("###############################################\n")

		result_dict = {"h": h+1, "c": c, "MAE": mae, "MAPE": mape, "MASE": mase}
		results.append(result_dict.copy())

		dates = list(test['date of day t'].to_numpy())
		df = pd.DataFrame()
		df['date'] = dates
		df['actual'] = list(y_test)
		df['predicted'] = list(y_pred)
		df.to_csv('Step1/results(h='+str(int(h+1))+'-c='+str(int(c))+'-mape='+str(mape)+').csv', index = False)

		K.clear_session()

	return results


def get_best_historical_parameters(results_dataframe):	# each element is a tuple of (h, c, R2_SCORE)
	best_score, best_h, best_c = 0, 0, 0

	results_dataframe[comparison_criteria] = pd.to_numeric(results_dataframe[comparison_criteria])
	index_of_min_score = results_dataframe[comparison_criteria].idxmin()

	best_h = results_dataframe.iloc[index_of_min_score]['h']
	best_c = results_dataframe.iloc[index_of_min_score]['c']
	best_score = results_dataframe.iloc[index_of_min_score][comparison_criteria]

	return tuple((best_h, best_c, best_score))


def lstm_model(dataset, h, c, dropout_value, model_type):	# this function works with specific values for h and c

	# dataset.drop('county_fips', axis=1, inplace=True)
	# dataset.drop(dataset.tail(4).index, inplace=True)

	########################################### split data
	train = dataset.head(len(dataset)-7)
	test = dataset.tail(7)

	X_train = train.drop(['date of day t', 'Target'], axis=1)
	y_train = train['Target']

	X_test = test.drop(['date of day t', 'Target'], axis=1)
	y_test = test['Target']

	naive_pred = train['Target'].tail(4).append(test['Target'].head(3))
	########################################### end of split data

	X_train, X_test, y_train, y_test = part_of_data(X_train, X_test, y_train, y_test, c)
	X_train, y_train, X_test, y_test, scaler = normalize(X_train, y_train, X_test, y_test)

	model = Sequential()
	model.add(LSTM(types[model_type][0], kernel_initializer=tf.keras.initializers.Identity(), return_sequences=True, input_shape=(1, X_train.shape[2])))
	model.add(Activation(custom1))
	for i in types[model_type][1:len(types[model_type])-1]:
		model.add(LSTM(i, dropout=dropout_value, return_sequences=True))
		model.add(Activation(custom1))
	model.add(LSTM(types[model_type][-1]))
	model.add(Activation(custom1))
	model.add(Dense(1))
	model.add(Activation(custom1))

	model.compile (
		loss=tf.keras.losses.MeanAbsoluteError(),
		optimizer=keras.optimizers.Adam(0.00005),
		metrics=[tf.keras.metrics.MeanAbsoluteError()]
	)

	early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1, patience=5)
	
	model.fit (
		X_train, y_train,
		epochs=1000,
		batch_size=8,
		validation_split=0.4,
		verbose=0,
		# callbacks=[early_stop], 
		shuffle=False
	)

	y_pred = model.predict(X_test)
	y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))
	
	y_pred = y_pred.reshape(-1)
	y_test = y_test.reshape(-1)
	naive = naive_pred.values.reshape(-1)

	# y_pred = np.round(y_pred.astype(np.double))
	# y_test = np.round(y_test.astype(np.double))
	# naive = np.round(naive.astype(np.double))
	
	mase = MASE(y_test, y_pred, naive)
	mae = mean_absolute_error(y_test, y_pred)
	mape = mean_absolute_percentage_error(y_test, y_pred)

	print("\n###############################################")
	print("model_type = " + str(model_type) + ", d = " + str(dropout_value) + ", mase = " + str(mase))
	print("###############################################\n")

	dates = list(test['date of day t'].to_numpy())
	df = pd.DataFrame()
	df['date'] = dates
	df['actual'] = list(y_test)
	df['predicted'] = list(y_pred)
	df.to_csv('Results/results(model_type='+str(int(model_type))+'-d='+str(dropout_value)+'-mape='+str(mape)+').csv', index = False)

	K.clear_session()

	return mape


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

	print("Started to make historical data")

	with mp.Pool(mp.cpu_count()) as pool:
		weeklyavg_data_list = pool.starmap(makeHistoricalData, 
			[(h, r, test_size, target, 'mrmr', 'country', 'weeklyaverage', '', [], 'country') 
			for h in range(1, H+1)])

	for i in range(len(weeklyavg_data_list)):
		weeklyavg_data_list[i].drop('county_fips', axis=1, inplace=True)
	
	print("Historical data is made\n")

	################################################### STEP 1
	print("--- STEP 1 ---")
	print("Started to train oh all possible values for 'h' and 'c'")
	start_time = time.time()
	
	with mp.Pool(mp.cpu_count()) as pool:
		default_model_results = pool.starmap(fit_model, [(weeklyavg_data_list[h], h) for h in range(0, H)])

	end_time = time.time()
	print("\nSTEP 1 FINISHED --- %s seconds for step 1 ---" % (end_time - start_time))

	tmp = list(itertools.chain.from_iterable(default_model_results))
	default_model_results_df = pd.DataFrame(tmp)

	# writing all country results into a file
	default_model_results_df.to_csv('Results/default_model_results.csv', index=False)

	tmp_df = pd.read_csv('Results/default_model_results.csv')

	print("\n--- Choosing best historical parameters ---")
	best_h, best_c, best_score = get_best_historical_parameters(tmp_df)
	
	with open('Results/historical_parameters.txt', 'w') as fp:
		fp.write("\nbest_h = " + str(int(best_h)))
		fp.write("\nbest_c = " + str(int(best_c)))
	
	# sending mail
	mail_subject = "unacast old rank confirmed - STEP 1"
	mail_body = "Finished STEP 1 --- took %s seconds --- best_h = %s and best_c = %s" % (end_time - start_time, int(best_h), int(best_c))

	send_mail(mail_subject, mail_body, "Results", "default_model_results.csv")
	################################################### END OF STEP 1

	# save the entire session until now
	shelve_filename = 'step1.out'
	s = shelve.open(shelve_filename, 'n')  # 'n' for new
	for key in dir():
		try:
			s[key] = locals()[key]
		except:
			print('ERROR shelving: {0}'.format(key))
	s.close()

	################################################### STEP 2
	print("\nChoosing best model type and best value for dropout")
	start_time = time.time()

	models_best_results = []
	models_best_dropout_values = []

	data = makeHistoricalData(int(best_h), r, test_size, target, 'mrmr', 'country', 'weeklyaverage', '', [], 'country')
	# data = weeklyavg_data_list[best_h-1]

	for model_type in range(len(types)):	# loop over model types
		with mp.Pool(mp.cpu_count()) as pool:
			results = pool.starmap(lstm_model, [(data, int(best_h), int(best_c), d/10, model_type) for d in range(1, 6)])

		best_result = min(results)	# getting min mase (mase for best dropout on this model type)
		models_best_results.append(best_result)
		
		dropout_value = results.index(min(results))
		models_best_dropout_values.append(dropout_value)

	best_type = models_best_results.index(min(models_best_results))
	best_dropout_value = models_best_dropout_values[best_type]

	end_time = time.time()

	with open('Results/model_types_results.txt', 'w') as f:
		for item in results:
			f.write("%s\n" % item)

	print("\n--- %s seconds to train the models --- best dropout value = %s and best model type = %s" \
		% (end_time - start_time, best_dropout_value, best_type))

	with open('Results/best_type_and_dropout_value.txt', 'w') as fp:
		fp.write("\nbest dropout value = " + str(int(best_dropout_value)))
		fp.write("\nbest type = " + str(int(best_type)))
	
	# sending mail
	mail_subject = "unacast old rank confirmed - STEP 2"
	mail_body = "Finished choosing the best model type and dropout value --- took %s seconds --- d = %s and type = %s" \
				% (end_time - start_time, int(best_dropout_value), int(best_type))

	send_mail(mail_subject, mail_body, "Results", "model_types_results.txt")
	################################################### END OF STEP 2

	# save the entire session until now
	shelve_filename = 'step2.out'
	s = shelve.open(shelve_filename, 'n')  # 'n' for new
	for key in dir():
		try:
			s[key] = locals()[key]
		except:
			print('ERROR shelving: {0}'.format(key))
	s.close()


if __name__ == "__main__":
	main()
