import pandas as pd
import numpy as np
import datetime

'''	This function gets a dataframe and removes first row of each group (gropus are distinguished by FIPS code)
	we use it for shifting dataframes'''


def using_mask(df):
    def mask_first(x):
        result = np.ones_like(x)
        result[0] = 0
        return result

    mask = df.groupby(['county_fips'])['county_fips'].transform(mask_first).astype(bool)
    return df.loc[mask]


# h is the number of days before day (t)
# r indicates how many days after day (t) --> target-day = day(t+r)
def makeHistoricalData(h, r):
    independantOfTimeDataPath = 'csvFiles/fixed-data.csv'  # Static Data is the data independant of time
    independantOfTimeData = pd.read_csv(independantOfTimeDataPath)

    timeDependantData = pd.read_csv('csvFiles/temporal-data.csv')  # Dynaimc Data is the data that is time dependant
    nameOfTimeDependantCovariates = timeDependantData.columns.values.tolist()  # Getting name of time dependant covariates

    nameOfTimeDependantCovariates.remove('county_fips')
    nameOfTimeDependantCovariates.remove('date')

    newDataFrame = pd.DataFrame()  # We store historical dynamic covariates in this dataframe
    totalNumberOfDays = len(timeDependantData['date'].unique())
    # In this loop we make historical data from time dependant covariates
    for name in nameOfTimeDependantCovariates:  # name would be name of a time dependant covariate
        temporalDataFrame = timeDependantData[['county_fips', name]]
        threshold = 0
        while threshold != h:
            # By the line bellow, we get first (totalNumberOfDays-h-r+1) rows of each group distinguished by FIPS code
            temp = temporalDataFrame.groupby('county_fips').head(totalNumberOfDays - h - r + 1).reset_index(drop=True)
            temp.rename(columns={name: (name + ' ' + str(threshold))}, inplace=True)
            newDataFrame = pd.concat([newDataFrame, temp], axis=1)
            temporalDataFrame = using_mask(temporalDataFrame)  # Shifting each group one time
            threshold += 1

    # By the lines bellow we concatenate target variable to historical datas were made before
    backupData = timeDependantData.copy()
    for i in range(r + h - 1):
        backupData = using_mask(backupData)
    backupData.rename(columns={'confirmed': 'Target'}, inplace=True)
    backupData = backupData.reset_index(drop=True)
    backupData = backupData['Target']
    newDataFrame = pd.concat([newDataFrame, backupData], axis=1)
    newDataFrame = newDataFrame.iloc[:, ~newDataFrame.columns.duplicated()]

    # By the lines bellow we concatenate date of day (t) to historical dataframe
    dates = timeDependantData[['county_fips', 'date']]
    # Deleting first (h-1) dates from each group
    for i in range(h - 1):
        dates = using_mask(dates)
    dates = dates.reset_index(drop=True)
    dates = dates.groupby('county_fips').head(totalNumberOfDays - h - r + 1).reset_index(
        drop=True)  # Get first (totalNumberOfDays-h-r+1) dates of each group
    dates.rename(columns={'date': ('date of day t')}, inplace=True)
    newDataFrame = pd.concat([newDataFrame, dates['date of day t']], axis=1)

    # Merging historical data and independant of time dataframes to each other

    independantOfTimeData.rename(columns={'county_fips': ('county_fips')}, inplace=True)
    result = pd.merge(independantOfTimeData, newDataFrame, on='county_fips')

    # Convert type of date column values from string to datetime
    result['date of day t'] = pd.to_datetime(result['date of day t'])
    result = result.sort_values(by=['date of day t', 'county_fips'])

    return result


def main():
    result = makeHistoricalData(3, 14)
    # Storing the result in a csv file
    result.to_csv('dataset_h=' + str(h) + '.csv', mode='w', index=False)


if __name__ == "__main__":
    main()
