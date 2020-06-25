import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# h is the number of days before day (t)
# r indicates how many days after day (t) --> target-day = day(t+r)
# target could be number of deaths or number of confirmed 
def makeHistoricalData(h, r, target, feature_selection, mode, address):
    ''' in this code when h is 1, it means there is no history and we have just one column for each covariate
    so when h is 0, we put h equal to 1, because when h is 0 that means there no history (as when h is 1) '''
    if h == 0:
        h = 1


    ##################################################################### imputation

    independantOfTimeData = pd.read_csv(address + 'fixed-data.csv')
    timeDeapandantData = pd.read_csv(address + 'temporal-data.csv')

    # impute missing values for tests in first days with min
    timeDeapandantData.loc[timeDeapandantData['daily-state-test']<0,'daily-state-test']=abs(timeDeapandantData.loc[timeDeapandantData['daily-state-test']<0,'daily-state-test'])

    if mode=='country':

        # for country mode we impute test variable
        # Next 3 lines create dataframe contains only daily-state-test to impute this feature
        temp=pd.DataFrame(index=timeDeapandantData['county_fips'].unique().tolist(),columns=timeDeapandantData['date'].unique().tolist())
        for i in timeDeapandantData['date'].unique():
            temp[i]=timeDeapandantData.loc[timeDeapandantData['date']==i,'daily-state-test'].tolist()

        # Next line find min daily-state-test performed in each county for impute first days missing values with min
        county_min_test=temp.replace(0,np.NaN).T.min()

        # impute missing tests for first days with min test performed in each county
        for i in temp.columns:
            temp.loc[pd.isna(temp[i]),i]=county_min_test[pd.isna(temp[i])]

        #replace imputed values in timeDeapandantData
        for i in timeDeapandantData['date'].unique():
            timeDeapandantData.loc[timeDeapandantData['date']==i,'daily-state-test']=temp[i].tolist()




    #Next 12 lines remove counties with all missing values for some features (counties with partly missing values have been imputed)
    independantOfTime_features_with_nulls=['ventilator_capacity','icu_beds','deaths_per_100000']

    for i in independantOfTime_features_with_nulls:
        nullind=independantOfTimeData.loc[pd.isnull(independantOfTimeData[i]),'county_fips'].unique()
        timeDeapandantData=timeDeapandantData[~timeDeapandantData['county_fips'].isin(nullind)]
        independantOfTimeData=independantOfTimeData[~independantOfTimeData['county_fips'].isin(nullind)]

    timeDeapandant_features_with_nulls=['social-distancing-travel-distance-grade','social-distancing-visitation-grade',
                                                'temperature','precipitation']

    for i in timeDeapandant_features_with_nulls:
        nullind=timeDeapandantData.loc[pd.isnull(timeDeapandantData[i]),'county_fips'].unique()
        timeDeapandantData=timeDeapandantData[~timeDeapandantData['county_fips'].isin(nullind)]
        independantOfTimeData=independantOfTimeData[~independantOfTimeData['county_fips'].isin(nullind)]


    #####################################################################



    allData = pd.merge(independantOfTimeData, timeDeapandantData, on='county_fips')
    allData = allData.sort_values(by=['date', 'county_fips'])
    allData = allData.reset_index(drop=True)
    # this columns are not numercal and wouldn't be included in correlation matrix, we store them to concatenate them later
    notNumericlData = allData[['county_name', 'state_name', 'county_fips', 'state_fips', 'date']]
    allData=allData.drop(['county_name', 'state_name', 'county_fips', 'state_fips', 'date'],axis=1)

    # next 19 lines ranking columns with mRMR
    cor=allData.corr().abs()
    valid_feature=cor.index.drop([target])
    overall_rank_df=pd.DataFrame(index=cor.index,columns=['mrmr_rank'])
    for i in cor.index:
        overall_rank_df.loc[i,'mrmr_rank']=cor.loc[i,target]-cor.loc[i,valid_feature].mean()
    overall_rank_df=overall_rank_df.sort_values(by='mrmr_rank',ascending=False)
    overall_rank=overall_rank_df.index.tolist()
    final_rank=[]
    final_rank=overall_rank[0:2]
    overall_rank=overall_rank[2:]
    while len(overall_rank)>0:
        temp=pd.DataFrame(index=overall_rank,columns=['mrmr_rank'])
        for i in overall_rank:
            temp.loc[i,'mrmr_rank']=cor.loc[i,target]-cor.loc[i,final_rank[1:]].mean()
        temp=temp.sort_values(by='mrmr_rank',ascending=False)
        final_rank.append(temp.index[0])
        overall_rank.remove(temp.index[0])

    # next 6 lines arranges columns in order of correlations with target or by mRMR rank
    if(feature_selection=='mrmr'):
        ix=final_rank
    else:
        ix = allData.corr().abs().sort_values(target, ascending=False).index

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

    result.dropna(inplace=True)

    result=result.sort_values(by=['county_fips','date of day t']).reset_index(drop=True)
    totalNumberOfDays=len(result['date of day t'].unique())
    county_end_index=0
    overall_non_zero_index=list()
    for i in result['county_fips'].unique():
        county_data = result[result['county_fips']==i]#.reset_index(drop=True)
        county_end_index = county_end_index+len(result[result['county_fips']==i])

        # we dont use counties with zero values for target variable in all history dates
        if (county_data[target+' t'].sum()>0):
            if h==1:
                # find first row index with non_zero values for target variable in all history dates when history length<7 
                first_non_zero_date_index = county_data[target+' t'].ne(0).idxmax()
            elif h<7:
                # find first row index with non_zero values for target variable in all history dates when history length<7 
                first_non_zero_date_index = county_data[target+' t-'+str(h-1)].ne(0).idxmax()
            else:
                # find first row index with non_zero values for target variable in 7 last days of history when history length>7 
                first_non_zero_date_index = county_data[target+' t'].ne(0).idxmax()+7

            zero_removed_county_index=[i for i in range(first_non_zero_date_index,county_end_index)]
            
            # we choose r days for test and r days for validation so at least we must have r days for train -> 3*r
            if (len(zero_removed_county_index) >= 3*r):
                    overall_non_zero_index = overall_non_zero_index+zero_removed_county_index
   

    
    zero_removed_data=result.loc[overall_non_zero_index,:]

    return zero_removed_data


def main():
    h = 0
    r = 14
    target = 'confirmed'
    feature_selection='mrmr'
    spatial_mode='country'
    address = './'
#     result = makeHistoricalData(h, r, target, feature_selection, spatial_mode, address)
    # Storing the result in a csv file
    # result.to_csv('dataset_h=' + str(h) + '.csv', mode='w', index=False)


if __name__ == "__main__":
    main()
