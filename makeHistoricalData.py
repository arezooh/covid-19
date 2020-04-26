import subprocess
import sys
# Installing required packages (pandas and numpy)
subprocess.check_call([sys.executable, "-m", "pip", "install", 'pandas'])
subprocess.check_call([sys.executable, "-m", "pip", "install", 'numpy'])
import pandas as pd
import numpy as np


'''	This function gets a dataframe and removes first row of each group (gropus are distinguished by FIPS code)
	we use it for shifting dataframes'''
def using_mask(df):
	def mask_first(x):
		result = np.ones_like(x)
		result[0] = 0
		return result

	mask = df.groupby(['FIPS code'])['FIPS code'].transform(mask_first).astype(bool)
	return df.loc[mask]


def makeHistoricalData(h, r):
	staticDataPath = 'final-fixed-data.csv'	# Static Data is the data independant of time
	dataFrameOfStaticCovariates = pd.read_csv(staticDataPath)

	dynamicData = pd.read_excel('confirnew2.xlsx')	# Dynaimc Data is the data that is time dependant
	nameOfDynamicCovariates = dynamicData.columns.values.tolist()	# Getting name of time dependant covariates

	nameOfDynamicCovariates.remove('FIPS code')
	nameOfDynamicCovariates.remove('date')

	newDataFrame = pd.DataFrame()	# We store historical dynamic covariates in this dataframe
	totalNumberOfDays = len(dynamicData['date'].unique())
	# In this loop we make historical data from time dependant covariates
	for name in nameOfDynamicCovariates:	# name would be name of a time dependant covariate
		temporalDataFrame = dynamicData[['FIPS code', name]]
		threshold = 0
		while threshold != h:
			# By the line bellow, we get first (totalNumberOfDays-h-r+1) rows of each group distinguished by FIPS code
			temp = temporalDataFrame.groupby('FIPS code').head(totalNumberOfDays-h-r+1).reset_index(drop=True)
			temp.rename(columns={name: (name+' '+str(threshold))}, inplace = True)
			newDataFrame = pd.concat([newDataFrame, temp], axis=1)
			temporalDataFrame = using_mask(temporalDataFrame)	# Shifting each group one time
			threshold += 1

	# By the lines bellow we concatenate target variable to historical datas were made before		
	backupData = dynamicData.copy()
	for i in range(r+h-1):
		backupData = using_mask(backupData)
	backupData.rename(columns = {'confirmed':'Target'}, inplace = True)
	backupData = backupData.reset_index(drop=True)
	backupData = backupData['Target']
	newDataFrame = pd.concat([newDataFrame, backupData], axis=1)
	newDataFrame = newDataFrame.iloc[:,~newDataFrame.columns.duplicated()]

	# By the lines bellow we concatenate date of day (t) to historical dataframe
	dates = dynamicData[['FIPS code', 'date']]
	# Deleting first (h-1) dates from each group
	for i in range(h-1):
		dates = using_mask(dates)
	dates = dates.reset_index(drop=True)
	dates = dates.groupby('FIPS code').head(totalNumberOfDays-h-r+1).reset_index(drop=True)	# Get first (totalNumberOfDays-h-r+1) dates of each group
	dates.rename(columns={'date': ('date of day t')}, inplace = True)
	newDataFrame = pd.concat([newDataFrame, dates['date of day t']], axis=1)

	# Merging historical data and independant of time dataframes to each other
	dataFrameOfStaticCovariates.rename(columns={'county_fips': ('FIPS code')}, inplace = True)
	result = pd.merge(dataFrameOfStaticCovariates, newDataFrame, on='FIPS code')
	
	# Storing the result in a csv file
	reslut = result.sort_values(by=['FIPS code'])
	result.to_csv('dataset_h='+str(h)+'.csv')


def main():
	makeHistoricalData(3, 14)


if __name__ == "__main__":
	main()
