import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
import os
import requests
import matplotlib.pyplot as plt
import subprocess as cmd

############################################## getting updated real values from data source
def get_csv(web_addres,file_address):
    url=web_addres
    print(url)
    req = requests.get(url)
    url_content = req.content
    csv_file = open(file_address, 'wb')
    csv_file.write(url_content)
    csv_file.close


def get_updated_covid_data(address):
    get_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
    ,address + 'international-covid-death-data.csv')
    get_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
    ,address + 'international-covid-confirmed-data.csv')

    

def generate_updated_temporal_data(address):
    death = pd.read_csv(address+'international-covid-death-data.csv')
    confirmed = pd.read_csv(address+'international-covid-confirmed-data.csv')
    death=death[death['Country/Region']=='US'].T
    death=death.iloc[4:,:]
    death.iloc[1:]=(death.iloc[1:].values-death.iloc[:-1].values)
    death = death.reset_index()
    death.columns = ['date','death']
    death['death']= death['death'].astype(int)
    confirmed=confirmed[confirmed['Country/Region']=='US'].T
    confirmed=confirmed.iloc[4:,:]
    confirmed.iloc[1:]=(confirmed.iloc[1:].values-confirmed.iloc[:-1].values)
    confirmed = confirmed.reset_index()
    confirmed.columns = ['date','confirmed']
    confirmed['confirmed']= confirmed['confirmed'].astype(int)
    confirmed_death = pd.merge(death,confirmed)
    confirmed_death['county_fips']=1
    confirmed_death['date'] = confirmed_death['date'].apply(lambda x:datetime.datetime.strptime(x,'%m/%d/%y'))
    confirmed_death=confirmed_death.sort_values(by=['date'])
    return(confirmed_death)

###################################################### make data weekly
def make_weekly(dailydata):
    dailydata.sort_values(by=['date','county_fips'],inplace=True)
    numberofcounties=len(dailydata['county_fips'].unique())
    numberofweeks=len(dailydata['date'].unique())//7

    weeklydata=pd.DataFrame(columns=dailydata.columns)

    for i in range(numberofweeks):
        temp_df=dailydata.head(numberofcounties*7) # weekly average of last week for all counties
        date=temp_df.tail(1)['date'].iloc[0]
        dailydata=dailydata.iloc[(numberofcounties*7):,:]
        temp_df=temp_df.drop(['date'],axis=1)
        temp_df=temp_df.groupby(['county_fips']).mean().reset_index()
        temp_df['date']=date # last day of week 
        weeklydata=weeklydata.append(temp_df)
    weeklydata.sort_values(by=['county_fips','date'],inplace=True)
    weeklydata=weeklydata.reset_index(drop=True)
    return(weeklydata)


def add_real_values(data,address,temporal_mode,target_name,weekly_r,daily_r,scenario_flag):
    data['prediction datetime'] = data['Prediction Date'].apply(lambda x : datetime.datetime.strptime(x,"%Y-%m-%d"))
    data['datetime'] = data['Date'].apply(lambda x : datetime.datetime.strptime(x[-11:],"%d %b %Y"))
    temporal_data = generate_updated_temporal_data(address)

    for date in data['Prediction Date'].unique():
      if pd.isna(data[data['Prediction Date']==date].tail(1)['Real'].values):
        previous_prediction_date = datetime.datetime.strptime(date,"%Y-%m-%d")
        
        
        temp = temporal_data[['county_fips','date',target_name]]

        real_values_df = pd.DataFrame(columns=['date',target_name])
        if temporal_mode == 'daily':
            for r in daily_r:
                first_day = previous_prediction_date
                last_day = first_day + datetime.timedelta(days=14)
                daily_part = temp[(temp['date']>first_day) & (temp['date']<=last_day)]
                if len(daily_part)>0 : last_day = max(daily_part['date'])
                daily_part['run_code'] = 'd'+str(r)
                real_values_df = real_values_df.append(daily_part)


        if temporal_mode == 'weekly':
            for r in weekly_r:
                first_day = previous_prediction_date
                print(first_day)
                last_day = first_day + datetime.timedelta(days=28)
                print(last_day)
                weekly_part = temp[(temp['date']>first_day) & (temp['date']<=last_day)]
                print(weekly_part)
                if len(weekly_part)>0 : last_day = max(weekly_part['date'])
                weekly_part = make_weekly(weekly_part)
                if scenario_flag:
                  weekly_part['run_code'] = 's'+str(r*7)
                else:
                  weekly_part['run_code'] = 'w'+str(r*7)
                real_values_df = real_values_df.append(weekly_part)

        data = pd.merge(data, real_values_df, how='left', left_on=['datetime','run_code'],right_on=['date','run_code'])
        print(data)
        new_ind=~pd.isnull(data[target_name])
        # add real values
        data.loc[new_ind,'Real'] = data.loc[~pd.isnull(data[target_name]),target_name]
        # calculate daily errors
        data.loc[new_ind,'Absolute Error']=abs(data.loc[new_ind,'Real']-data.loc[new_ind,'Prediction'])

        # calculate average errors
        for i in data[new_ind].index.values:

            # mean absolute error
            past_data = data.loc[:i,:]
            run_code = data.loc[i,:]['run_code']
            past_data = past_data[past_data['run_code'] == run_code]
            # only rows with 'Absolute Error' are the future rows
            future_days_index = past_data[~pd.isna(past_data['Absolute Error'])].index
            if temporal_mode == 'daily' :
              past_number = 21
            # for weekly mode we consider 8 weeks for test 
            if temporal_mode == 'weekly' :
              past_number = 8
            past_sum_absolute_error = past_number*(past_data.head(1)['MAE (test)'].iloc[0])
            present_sum_absolute_error = past_data['Absolute Error'].sum()
            present_number = len(future_days_index)
            sum_of_absolute_error = past_sum_absolute_error + present_sum_absolute_error
            sum_of_numbers = past_number + present_number
            data.loc[i,'Daily MAE'] = sum_of_absolute_error/sum_of_numbers

            # mean absolute precentage error
            past_MAPE_error = (past_data.head(1)['MAPE (test)'].iloc[0])/100
            sum_of_past_real_values = past_sum_absolute_error/past_MAPE_error
            sum_of_present_real_values = past_data.loc[future_days_index]['Real'].sum()
            sum_of_real_values = sum_of_past_real_values + sum_of_present_real_values
            data.loc[i,'Daily MAPE'] = (sum_of_absolute_error/sum_of_real_values)*100

        data = data.drop([target_name,'date'],axis=1)

    data = data.drop(['prediction datetime','datetime'],axis=1)
    return(data)

################################################### plot


def plot(data,address):
    plt.rc('font', size=100)
    data=data[~pd.isna(data['Real'])].drop_duplicates(subset=['the day of the target variable'])
    dates=data['the day of the target variable'].unique().tolist()[40:]
    plot_with = len(dates) + 2
    fig, ax = plt.subplots(figsize=(plot_with, 40))
    plt.plot(dates,data['Real'].tolist()[40:],label='Real number of deaths', linewidth=5.0)
    plt.plot(dates,data['Prediction'].tolist()[40:],label='Predicted number of deaths', linewidth=5.0)
    line_position = dates.index('25 Sep 2020')
    plt.axvline(x=line_position, color='k', linestyle='--')
    plt.ylabel('number of deaths')
    plt.xlabel('date')
    plt.legend()
    plt.xticks(rotation=65)
    locs, labels = plt.xticks()
    weeks = (len(dates)//7)
    plt.xticks([0+(i*7) for i in range(weeks)], np.array(dates)[[0+(i*7) for i in range(weeks)]])
    plt.tight_layout()
    plt.savefig(address + 'US_real_prediction_values.pdf')
    plt.close()
    
########################################################### push results to github
def push(message):
    
    cmd.run("git pull", check=True, shell=True)
    print("everything has been pulled")
    cmd.run("git add .", check=True, shell=True)
    cmd.call(["git", "commit", "-m", "'{}'".format(message), "--allow-empty"], shell=True)
    cmd.run("git push", check=True, shell=True)
    print('pushed.')


def main():
    
    address = './'
    
    get_updated_covid_data(address)
    target_name = 'death'
    weekly_r = [4]
    daily_r = [14]

    weekly_output_csv = pd.read_csv(address + "weekly_backup.csv")
    daily_output_csv = pd.read_csv(address + "daily_backup.csv")
    scenarios_output_csv = pd.read_csv(address + "scenarios_backup.csv")

    # add real values of previous days
    weekly_output_csv = add_real_values(weekly_output_csv,address,'weekly',
                                        target_name,weekly_r,daily_r,0)
    daily_output_csv = add_real_values(daily_output_csv,address,'daily',
                                       target_name,weekly_r,daily_r,0)
    scenarios_output_csv = add_real_values(scenarios_output_csv,address,'weekly',
                                           target_name,weekly_r,daily_r,1)


    # remove duplaicates
    for col in ['MASE (test)', 'MAPE (test)', 'MAE (test)','RMSE (test)']:
           daily_output_csv.loc[daily_output_csv[col].duplicated(), col]=np.nan
           weekly_output_csv.loc[weekly_output_csv[col].duplicated(), col]=np.nan
           scenarios_output_csv.loc[scenarios_output_csv[col].duplicated(), col]=np.nan

    daily_output_csv.to_csv('daily_backup.csv',index = False)
    weekly_output_csv.to_csv('weekly_backup.csv',index = False)
    scenarios_output_csv.to_csv('scenarios_backup.csv',index = False)


    ############################### prepare output csv file

    daily_output_csv=daily_output_csv.drop(['run_code'],axis=1)
    weekly_output_csv=weekly_output_csv.drop(['run_code'],axis=1)
    scenarios_output_csv = scenarios_output_csv.drop(['run_code'],axis=1)

    daily_output_csv['Absolute Error'] = abs(daily_output_csv['Real']-daily_output_csv['Prediction'])
    weekly_output_csv['Absolute Error'] = abs(weekly_output_csv['Real']-weekly_output_csv['Prediction'])
    scenarios_output_csv['Absolute Error'] = abs(scenarios_output_csv['Real']-scenarios_output_csv['Prediction'])

    numerical_cols = ['Real', 'Prediction','Absolute Error', 'MASE (test)',\
                                   'MAPE (test)', 'MAE (test)', 'RMSE (test)', 
                                   'Daily MAE', 'Daily MAPE']
    numerical_scenarios_cols = ['Real', 'Prediction','Absolute Error', 
                                 'future-social-distancing-encounters-grade','future-social-distancing-total-grade',
                  'future-social-distancing-travel-distance-grade', 'MASE (test)',\
                                   'MAPE (test)', 'MAE (test)', 'RMSE (test)', 
                                   'Daily MAE', 'Daily MAPE']
    for col in numerical_cols:
      daily_output_csv[col]=daily_output_csv[col].apply(lambda x : np.round(x,2))
      weekly_output_csv[col]=weekly_output_csv[col].apply(lambda x : np.round(x,2))
    for col in numerical_scenarios_cols:
      scenarios_output_csv[col]=scenarios_output_csv[col].apply(lambda x : np.round(x,2))


    ordered_scenarios_cols = ['Prediction Date','Date', 'Scenario', 'Real', 'Prediction','Absolute Error', 'Model', 
                      'future-social-distancing-encounters-grade','future-social-distancing-total-grade',
                  'future-social-distancing-travel-distance-grade','MASE (test)',\
      'MAPE (test)', 'MAE (test)', 'RMSE (test)', 'Daily MAE', 'Daily MAPE']

    ordered_cols = ['Prediction Date','Date', 'Real', 'Prediction','Absolute Error', 'Model', 'MASE (test)',\
      'MAPE (test)', 'MAE (test)', 'RMSE (test)', 'Daily MAE', 'Daily MAPE']     

    daily_output_csv = daily_output_csv[ordered_cols]
    weekly_output_csv = weekly_output_csv[ordered_cols]
    scenarios_output_csv = scenarios_output_csv[ordered_scenarios_cols]

    daily_output_csv = daily_output_csv.rename(columns={'Date':'the day of the target variable', 'Prediction Date':'the day the prediction is made',\
                                 'Absolute Error':'difference'})
    weekly_output_csv = weekly_output_csv.rename(columns={'Date':'the week of the target variable', 'Prediction Date':'the day the prediction is made',\
                                 'Absolute Error':'difference', 'Daily MAE':'Weekly MAE','Daily MAPE':'Weekly MAPE'})
    scenarios_output_csv = scenarios_output_csv.rename(columns={'Date':'the week of the target variable', 'Prediction Date':'the day the prediction is made',\
                                 'Absolute Error':'difference', 'Daily MAE':'Weekly MAE','Daily MAPE':'Weekly MAPE'})

    # save plot of real and predicted values
    plot(daily_output_csv,address)

    daily_output_csv.to_csv("US-Daily-Deaths-Prediction.csv")
    weekly_output_csv.to_csv("US-Weekly-Deaths-Prediction.csv")
    scenarios_output_csv.to_csv("Different-Scenarios.csv", index=False)
    
    push("Predictions updated")

if __name__ == "__main__":
    main()