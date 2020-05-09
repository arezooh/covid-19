import pandas as pd
import numpy as np

# h is the number of days before day (t)
# r indicates how many days after day (t) --> target-day = day(t+r)
# target could be number of deaths or number of confirmed 
def makeHistoricalData(h, r, target):
	''' in this code when h is 1, it means there is no history and we have just one column for each covariate
	so when h is 0, we put h equal to 1, because when h is 0 that means there no history (as when h is 1) '''
	if h == 0:
		h = 1
	
	independantOfTimeData = pd.read_csv('csvFiles/fixed-data.csv')
	timeDeapandantData = pd.read_csv('csvFiles/temporal-data.csv')

	allData = pd.merge(independantOfTimeData, timeDeapandantData, on='county_fips')
	allData = allData.sort_values(by=['date', 'county_fips'])
	allData = allData.reset_index(drop=True)
	# this columns are not numercal and wouldn't be included in correlation matrix, we store them to concatenate them later
	notNumericlData = allData[['county_name', 'state_name', 'date']]

	# next 2 lines arranges columns in order of correlations with target
	ix = allData.corr().sort_values(target, ascending=False).index
	allData = allData.loc[:, ix]
	allData = pd.concat([allData, notNumericlData], axis=1)

	nameOfTimeDependantCovariates = timeDeapandantData.columns.values.tolist()
	nameOfAllCovariates = allData.columns.values.tolist()

	result = pd.DataFrame()  # we store historical data in this dataframe
	totalNumberOfCounties = len(allData['county_fips'].unique())
	totalNumberOfDays = len(allData['date'].unique())

	# in this loop we make historical data
	for name in nameOfAllCovariates:
		# if covariate is time dependant
		if name in nameOfTimeDependantCovariates and name not in ['date', 'county_fips']:
			temporalDataFrame = allData[[name]] # selecting column of the covariate that is being processed
			threshold = 0
			while threshold != h:
				# get value of covariate that is being processed in first (totalNumberOfDays-h-r+1) days
				temp = temporalDataFrame.head((totalNumberOfDays-h-r+1)*totalNumberOfCounties).copy().reset_index(drop=True)
				temp.rename(columns={name: (name + ' t-' + str(h-threshold-1))}, inplace=True) # renaming column
				result = pd.concat([result, temp], axis=1)
				# deleting the values in first day in temporalDataFrame dataframe (similiar to shift)
				temporalDataFrame = temporalDataFrame.iloc[totalNumberOfCounties:]
				threshold += 1
		# if covariate is independant of time
		elif name not in nameOfTimeDependantCovariates and name not in ['date', 'county_fips']:
			temporalDataFrame = allData[[name]]
			temp = temporalDataFrame.head((totalNumberOfDays-h-r+1)*totalNumberOfCounties).copy().reset_index(drop=True)
			result = pd.concat([result, temp], axis=1)

	# next 3 lines is for adding FIPS code to final dataframe
	temporalDataFrame = allData[['county_fips']]
	temp = temporalDataFrame.head((totalNumberOfDays-h-r+1)*totalNumberOfCounties).copy().reset_index(drop=True)
	result.insert(0, 'county_fips', temp)

	# next 3 lines is for adding date of day (t) to final dataframe
	temporalDataFrame = allData[['date']]
	temporalDataFrame = temporalDataFrame[totalNumberOfCounties*(h-1):]
	temp = temporalDataFrame.head((totalNumberOfDays-h-r+1)*totalNumberOfCounties).copy().reset_index(drop=True)
	result.insert(1, 'date of day t', temp)

	# next 3 lines is for adding target to final dataframe
	temporalDataFrame = allData[[target]]
	temporalDataFrame = temporalDataFrame.tail((totalNumberOfDays-h-r+1)*totalNumberOfCounties).reset_index(drop=True)
	result.insert(1, 'Target', temporalDataFrame)
	for i in result.columns:
            if i.endswith('t-0'):
                result.rename(columns={i: i[:-2]}, inplace=True)

	return result


def main():
	h = 0
	r = 14
	target = 'confirmed'
	result = makeHistoricalData(h, r, target)
	# Storing the result in a csv file
	result.to_csv('dataset_h=' + str(h) + '.csv', mode='w', index=False)


if __name__ == "__main__":
	main()
