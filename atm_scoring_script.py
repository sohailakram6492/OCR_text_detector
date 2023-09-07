import base64
import pandas as pd
import numpy as np
from fbprophet import Prophet
from sklearn.metrics import mean_squared_error,mean_absolute_error
from statsmodels.tsa.stattools import acf, pacf, adfuller
from datetime import datetime, timedelta
import math
from math import sqrt
from sklearn.metrics import mean_squared_error
import warnings
import itertools
import functools
import logging 
import multiprocessing as mp
from multiprocessing import pool
import time
import os
from numpy import vectorize
import sys
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import cx_Oracle
from pyhive import hive
import pyodbc
import ipywidgets as widgets
import sqlalchemy
from subprocess import call
import config
from sqlalchemy import create_engine
import os
import datetime
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import sys
import io
from io import StringIO
import importlib
import traceback
importlib.reload(config)
warnings.filterwarnings('ignore')
logging.getLogger('pyhive').setLevel(logging.CRITICAL)


def generate_HTML(message):
    HTML = """
    <!DOCTYPE html>
    <html>
      <head>
        <meta charset="utf-8" />
        <style type="text/css">
      </head>
      <body>
        {}
      </body>
    </html>
    """.format(message)
    return HTML

def _generate_message(HTML,SUBJECT,SENDER_EMAIL ,SENDER_PASSWORD,RECEIVER_EMAIL) -> MIMEMultipart:
    message = MIMEMultipart("alternative", None, [MIMEText(HTML, 'html')])
    message['Subject'] = SUBJECT
    message['From'] = SENDER_EMAIL
    message['To'] = ",".join(RECEIVER_EMAIL)
    return message


def send_message(SENDER_EMAIL,SENDER_PASSWORD, SERVER, RECEIVER_EMAIL, SUBJECT, message):
    HTML = generate_HTML(message)
    message = _generate_message(HTML,SUBJECT,SENDER_EMAIL ,SENDER_PASSWORD,RECEIVER_EMAIL)
    server = smtplib.SMTP(SERVER)
    SERVER.ehlo()
    SERVER.starttls()
    SERVER.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, message.as_string())
    SERVER.quit()
    print("email sent.")

#compress stdout/stderr so that log files don't have junk
class suppress_stdout_stderr(object):
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])
        
#This function extract column name as hive by default append table names with columns
def extract_column_names(admfactdata_columnnames):
    try:
        columnsnamelist = []
        for item in admfactdata_columnnames:
            name = item.split(".")
            columnname = name[1]
            columnsnamelist.append(columnname)
        return columnsnamelist
    except:
        logging.info('Exception while extracting column names')
        logging.exception('Exception in function extract_column_names')
        sys.exit(1)
        
#This function query on hive and load fact data
def load_fact_data(conn_hive_t24, test_atms, schame_name, table_name): 
    try:
        test_atms = list(map(str,test_atms))
        filter_atms = ','.join(["'"+i+"'" for i in test_atms])
        datest = str(pd.datetime.now().date().year - 4) + str(1).zfill(2) + str(1).zfill(2)
        _sql_fact = 'SELECT * FROM {}.{} WHERE \
                    to_date(from_unixtime(unix_timestamp({}.cal_date,\'yyyyMMdd\'))) >= \
                    to_date(from_unixtime(unix_timestamp(\'{}\',\'yyyyMMdd\'))) and atm_id in ({})'.format(schame_name
                                                                                                           ,table_name,
                                                                                                           table_name,datest,
                                                                                                           filter_atms)
        df1 = []
        admfactdata = pd.DataFrame()
        for chunk in pd.read_sql(_sql_fact,conn_hive_t24,chunksize=100000):
            df1.append(chunk)
        admfactdata = pd.concat(df1,ignore_index=True)
        return admfactdata
    except:
        logging.info('Exception while loading fact data')
        logging.exception('Exception in function load_fact_data')
        sys.exit(1)
        
#This function query on hive and load calender dimenion data
def load_date_dimension(conn_hive_t24, schame_name): 
    try:
        datest = str(pd.datetime.now().date().year - 4) + str(1).zfill(2) + str(1).zfill(2)
        _sql_dimension = 'SELECT * FROM {}.calendar_date WHERE \
                        to_date(from_unixtime(unix_timestamp(calendar_date.time_period_id,\'yyyyMMdd\'))) >= \
                        to_date(from_unixtime(unix_timestamp(\'{}\',\'yyyyMMdd\')))'.format(schame_name,datest)
        df1 = []
        admdimensiondata = pd.DataFrame()
        for chunk in pd.read_sql(_sql_dimension,conn_hive_t24,chunksize=100000):
            df1.append(chunk)
        admdimensiondata = pd.concat(df1,ignore_index=True)
        return admdimensiondata
    except:
        logging.info('Exception while loading date dimension data')
        logging.exception('Exception in function load_date_dimension')
        sys.exit(1)
        
        
#This function prepare data to be fed to model
def prepare_data(conn_hive_t24,test_start_date,delta,no_of_days,test_atms,schema_name,table_name,fact_table):    
    try:    
        #load fact data
        admfactdata = load_fact_data(conn_hive_t24,test_atms,schema_name,fact_table)
        admfactdata_columnnames = admfactdata.columns
        extracted_fact_columns = extract_column_names(admfactdata_columnnames)
        admfactdata.rename(columns={i:j for i,j in zip(admfactdata_columnnames,extracted_fact_columns)},inplace=True)
        admfactdata['time_period_id'] = pd.to_datetime(admfactdata['cal_date']).dt.date

        #load dimension data
        datedimensiondata = load_date_dimension(conn_hive_t24,schema_name)
        datedimensiondata_columnnames = datedimensiondata.columns
        extracted_dimension_columns = extract_column_names(datedimensiondata_columnnames)
        datedimensiondata.rename(columns={i:j for i,j in zip(datedimensiondata_columnnames,extracted_dimension_columns)},inplace=True)
        datedimensiondata['time_period_id'] = pd.to_datetime(datedimensiondata['time_period_id']).dt.date

        #join data
        joineddata = pd.merge(admfactdata,datedimensiondata, on=['time_period_id'], how = 'left')
        
        #Creating dates of training end date, forecasted start date and forecasted end date
        if test_start_date == '1-1-2014':
            test_date_start = pd.datetime.now().date() + timedelta(days=delta)
        else:
            test_start_date = test_start_date.split(sep='-')
            test_date_start = pd.datetime(int(test_start_date[2]),int(test_start_date[1]),int(test_start_date[0])).date()
            
            
        test_end_date = test_date_start + timedelta(days=no_of_days-1)
        train_end_date = test_date_start + timedelta(days=-1)
        
        print(test_date_start)
        print(test_end_date)
        print(train_end_date)
        
        #preparing testing dataframe
        testdata = datedimensiondata.loc[datedimensiondata.time_period_id >= pd.datetime(test_date_start.year,test_date_start.month,test_date_start.day).date()]
        testdata = testdata.loc[testdata.time_period_id <= pd.datetime(test_end_date.year,test_end_date.month,test_end_date.day).date()]
        return joineddata,testdata,datedimensiondata,test_end_date,test_date_start
    except:
        logging.info('Exception while loading preparing data')
        logging.exception('Exception in function prepare_data')
        sys.exit(1)
        
#This function is used to get training start date for each atm model as this varies across atms
def get_training_dates(conn_hive_t24,schema_name,table_name):
    try:
        _sql_hyperparamters = 'SELECT atm_id, train_date_start, percentage_error, timestamp_ FROM {}.{}'.format(schema_name,table_name)
        traindata = pd.DataFrame()
        traindata =  pd.read_sql(_sql_hyperparamters,conn_hive_t24)
        traindata['train_date_start'] = pd.to_datetime(traindata['train_date_start']).dt.date
        return traindata
    except:
        logging.info('Exception while reading training dates for atm')
        logging.exception('Exception in function get_training_dates')
        sys.exit(1)
        
#This function is used to read hyperparamters for all atms
def read_hyperparamters(conn_hive_t24,schema_name,table_name):
    try:
        _sql_hyperparamters = 'SELECT * FROM {}.{}'.format(schema_name,table_name)
        df1 = []
        hyperparamterdata = pd.DataFrame()
        for chunk in pd.read_sql(_sql_hyperparamters,conn_hive_t24,chunksize=100000):
            df1.append(chunk)     
        hyperparamterdata = pd.concat(df1,ignore_index=True)
        hyperparamterdata_columnnames = hyperparamterdata.columns
        extracted_hyperparamterdata_columns = extract_column_names(hyperparamterdata_columnnames)
        hyperparamterdata.rename(columns={i:j for i,j in zip(hyperparamterdata_columnnames,extracted_hyperparamterdata_columns)},inplace=True)     
        return hyperparamterdata
    except:
        logging.info('Exception while reading hyperparamters data')
        logging.exception('Exception in function read_hyperparamters')
        sys.exit(1)
        
#This function is used to get training start date for each atm model as this varies across atms        
def get_margin_data(conn_hive_t24,schema_name, table_name,test_date_start,past_data_for_margin,test_atms):
    try:
     
        margin_st_dt = test_date_start + timedelta(days=-past_data_for_margin)
        test_atms = list(map(str,test_atms))
        filter_atms = ','.join(["'"+i+"'" for i in test_atms])
        marginst = str(margin_st_dt.year) + '-' + str(margin_st_dt.month).zfill(2) + '-' + str(margin_st_dt.day).zfill(2) 
        _sql_margin = 'SELECT * FROM {}.{} where \
        to_date(from_unixtime(unix_timestamp({}.forecasted_date,\'yyyy-MM-dd\'))) \
        >= to_date(from_unixtime(unix_timestamp(\'{}\',\'yyyy-MM-dd\'))) and atm_id in ({})' \
        .format(schema_name,table_name,table_name,marginst,filter_atms)
        df1 = []
        margindata = pd.DataFrame(columns=['forecasted_date','forecasted_amount','marginal_amount',
                                           'day_of_month','atm_id','region_id','region_name','branch_id',
                                           'branch_name','timestamp_'])
        
        for chunk in pd.read_sql(_sql_margin,conn_hive_t24,chunksize=100000):
            df1.append(chunk)  
        if len(df1) > 0:
            margindata = pd.concat(df1,ignore_index=True)
            margindata_columnnames = margindata.columns
            extracted_margindata_columns = extract_column_names(margindata_columnnames)
            margindata.rename(columns={i:j for i,j in zip(margindata_columnnames,extracted_margindata_columns)},inplace=True) 
            margindata['forecasted_date'] = pd.to_datetime(margindata['forecasted_date']).dt.date
            margindata = margindata.sort_values(by='forecasted_date')
            margindata = margindata.loc[margindata.forecasted_date < test_date_start]  
        return margindata
    except:
        logging.info('Exception while reading data for margin calculation')
        logging.exception('Exception in function get_margin_data')
        sys.exit(1)
            
#Get Branch id and name        
def get_branch(conn_hive_t24,schema_name,table_name): 
    try:
        _sql_branches_fact = 'select brcd, brcdname from {}.{}'.format(schema_name,table_name)
        df1 = []
        admfactdata = pd.DataFrame()
        for chunk in pd.read_sql(_sql_branches_fact,conn_hive_t24,chunksize=100000):
            df1.append(chunk)
        admfactdata = pd.concat(df1,ignore_index=True)
        admfactdata['branch_id'] = admfactdata['brcd']
        return admfactdata
    except:
        logging.info('Exception in extracting branches name')
        logging.exception('Exception in function get_branch')
        sys.exit(1)

#Get region id and name  
def get_region(conn_hive_t24,schema_name,table_name): 
    try:
        _sql_region_fact = 'select regioncode,regionfullname from {}.{}'.format(schema_name,table_name)
        df1 = []
        admfactdata = pd.DataFrame()
        for chunk in pd.read_sql(_sql_region_fact,conn_hive_t24,chunksize=100000):
            df1.append(chunk)
        admfactdata = pd.concat(df1,ignore_index=True)
        admfactdata['region_id'] = admfactdata['regioncode']
        return admfactdata
    except:
        logging.info('Exception in extracting regions name')
        logging.exception('Exception in function get_region')
        sys.exit(1)

#This function is used to read list of atms from database
def get_live_atms(conn_hive_t24,schema_name,table_name):
    try:
        _live_atms = 'SELECT atm_id FROM {}.{} where status = 1'.format(schema_name,table_name)
        list_of_live_atms = pd.DataFrame()
        list_of_live_atms =  pd.read_sql(_live_atms,conn_hive_t24)
        return list_of_live_atms
    except:
        logging.info('Exception while getting list of live atms')
        logging.exception('Exception in function get_live_atms')
        sys.exit(1)
        
def salary_days_func(start,end,ds):
    if (ds >= 1) and (ds <=end):
        return 1
    elif (ds >= start) and (ds <=31):
        return 1
    else:
        return 0
        
#Fbprophet fitting and scoring function
def prophet_execution(params,atmid,traindata,testdata,holidaysdf,train_start,forecast_end,margin_data,weekday_margin,weekend_margin):  
    try:
        if (traindata.shape[0] > 90):
            params.sort_values(by=['timestamp_'],inplace=True, ascending=False)
            params = params.iloc[0]
            traindata['salary_days'] = list(map(functools.partial(salary_days_func, int(float(params.salary_start)), \
                                                                  int(float(params.salary_end))), \
                                                                  traindata.day_of_month.astype(int).tolist()))
            testdata['salary_days'] = list(map(functools.partial(salary_days_func, int(float(params.salary_start)), \
                                                                 int(float(params.salary_end))), \
                                                                 testdata.day_of_month.astype(int).tolist()))
            if (holidaysdf.shape[0] > 0) :
                if traindata.y.notnull().sum() < 360:

                    model_with_holidays = Prophet(growth="linear",changepoint_prior_scale=float(params.changepoint_prior_scale),
                                                  interval_width = float(0.99), 
                                seasonality_mode = "additive",holidays=holidaysdf, daily_seasonality=False,
                                weekly_seasonality=False,yearly_seasonality=False,
                               ).add_seasonality(
                                    name = 'monthly',
                                    period=30.5,
                                    fourier_order= int(float(params.fmonthly[0])),
                                    prior_scale=10
                                ).add_seasonality(
                                    name= "weekly",
                                    period= 7,
                                    fourier_order= int(float(params.fweekly[0])),
                                    prior_scale=10
                                )
                else:
                    model_with_holidays = Prophet(growth="linear",changepoint_prior_scale=float(params.changepoint_prior_scale),
                                                  interval_width = float(0.99), 
                                seasonality_mode = "additive",holidays=holidaysdf, daily_seasonality=False,
                                weekly_seasonality=False,yearly_seasonality=False,
                               ).add_seasonality(
                                    name = 'monthly',
                                    period=30.5,
                                    fourier_order= int(float(params.fmonthly[0])),
                                    prior_scale=10
                                ).add_seasonality(
                                    name= "weekly",
                                    period= 7,
                                    fourier_order= int(float(params.fweekly[0])),
                                    prior_scale=10
                                ).add_seasonality(name='yearly', period=365.25, fourier_order=int(10), prior_scale=0.01);


            else:
                 model_with_holidays = Prophet(growth="linear",changepoint_prior_scale=float(params.changepoint_prior_scale),
                                               interval_width = float(0.99), 
                            seasonality_mode = "additive", daily_seasonality=False,
                            weekly_seasonality=False,yearly_seasonality=False,
                           ).add_seasonality(
                                name= "weekly",
                                period= 7,
                                fourier_order= int(float(params.fweekly[0])),
                                prior_scale=10
                            )
            model_with_holidays.add_regressor('salary_days',mode='multiplicative')       
            #fitting model
            with suppress_stdout_stderr():
                model_with_holidays.fit(traindata)
            testdata = testdata.sort_values(by='ds') 
        
            #forecasting
            future = model_with_holidays.make_future_dataframe(periods=testdata.shape[0])
            templist = []
            templist.append(list((traindata.day_of_month)))
            templist.append(list((testdata.day_of_month)))
            flatlist = [item for sublist in templist for item in sublist]
            future['day_of_month'] = flatlist
            future['salary_days'] = list(map(functools.partial(salary_days_func, int(float(params.salary_start)), 
                                                               int(float(params.salary_end))), 
                                                               future.day_of_month.astype(int).tolist()))
            forecast_with_holidays = model_with_holidays.predict(future)
            forecast_with_holidays = remove_negs(forecast_with_holidays,traindata.y)
            forecast_yhat = forecast_with_holidays.yhat.iloc[-testdata.shape[0]:]
            forecast_yhat_upper = forecast_with_holidays.yhat_upper.iloc[-testdata.shape[0]:]   
            testdata['forecasted_amount'] = forecast_yhat.values
            margin_factor_weekday,margin_factor_weekend = margincalculation(atmid,traindata,margin_data,testdata,weekday_margin,weekend_margin)
            #print(atmid,margin_factor_weekday,margin_factor_weekend)
            
            if margin_factor_weekday > 0.60:
                margin_factor_weekday = 0.60
                
            if margin_factor_weekend > 0.60:
                margin_factor_weekend = 0.60

            testdata['day_of_week'] = pd.to_datetime(testdata['ds']).dt.weekday

            testdata_weekday = testdata[testdata.day_of_week.isin([0,1,2,3,4])]
            testdata_weekend = testdata[testdata.day_of_week.isin([5,6])]

            testdata_weekday['marginal_amount'] = (testdata_weekday['forecasted_amount'] * margin_factor_weekday) + \
                                                    testdata_weekday['forecasted_amount']
            testdata_weekend['marginal_amount'] = (testdata_weekend['forecasted_amount'] * margin_factor_weekend) + \
                                                    testdata_weekend['forecasted_amount']
        
            frames_cat=[testdata_weekday,testdata_weekend]
            testdata = pd.concat(frames_cat)
            table_temp = pd.DataFrame()  
            table_temp['forecasted_date'] = testdata['ds']
            table_temp['forecasted_amount'] = testdata['forecasted_amount'].astype(float) \
                                                                    .apply(lambda x:math.ceil(x*(1.0/500))/(1.0/500))
            table_temp['marginal_amount'] = fixedmargin(testdata['marginal_amount'],traindata.y.mean())
            table_temp['marginal_amount'] =  table_temp['marginal_amount'].astype(float) \
                                                                    .apply(lambda x:math.ceil(x*(1.0/500))/(1.0/500))
            
            table_temp['day_of_month'] = testdata['day_of_month']
            table_temp['atm_id'] = atmid
            table_temp['region_id'] = traindata.region_id.unique()[0]
            table_temp['region_name'] = traindata.regionfullname.unique()[0]
            table_temp['branch_id'] = traindata.branch_id.unique()[0]
            table_temp['branch_name'] = traindata.brcdname.unique()[0]  
            table_temp['timestamp_'] = datetime.datetime.now()
            table_temp.sort_values(by=['forecasted_date'],inplace=True)
            return table_temp
        else:
            return
        
    except:
        logging.info('Exception while executing prophet')
        logging.exception('Exception in function prophet_execution')
        sys.exit(1)
        
#Dynamic margin calculation
def margincalculation(atmid,traindata,margin_data,testdata,weekday_margin,weekend_margin):
    
    try:
        #get margin data minimum and maximum date
        minimum_margin_date = margin_data.forecasted_date.min()
        maximum_margin_date = margin_data.forecasted_date.max()
        
        #extract data from traindata of those dates
        extracted_data = traindata.loc[traindata.ds <= maximum_margin_date]
        extracted_data = extracted_data.loc[extracted_data.ds >= minimum_margin_date]
        
        #compare actual and forecasted and calculate %error
        actualdata = pd.DataFrame()
        actualdata['actual'] = extracted_data.y.astype(float)
        actualdata['date'] = extracted_data.ds
        
        forecasteddata = pd.DataFrame()
        forecasteddata['forecasted'] = margin_data.forecasted_amount.astype(float)
        forecasteddata['date'] = margin_data.forecasted_date 
        forecasteddata['day_of_month'] = margin_data['day_of_month'].values
        
        big_final_data = pd.merge(forecasteddata,actualdata, on=['date'], how = 'left')
        big_final_data.dropna(inplace=True)
        
       
        big_final_data['day_of_week'] = pd.to_datetime(big_final_data['date']).dt.weekday
        
        big_final_data['percentage_error'] = (abs(big_final_data['actual']- big_final_data['forecasted']) \
                                              /big_final_data['actual'])*100

        current_week_days_list = list(testdata['day_of_month'].values)
        
        if big_final_data.shape[0] > 0:
            new = big_final_data['day_of_month'].isin(current_week_days_list) 
            big_final_data = big_final_data[new]
            big_final_data.dropna(inplace=True)
            
            filter_ = big_final_data['actual'] > big_final_data['forecasted']
            big_final_data.where(filter_, inplace = True) 
            big_final_data.dropna(inplace=True)  
      
            if big_final_data.shape[0] > 0:
          
                weekday_frame = big_final_data[big_final_data.day_of_week.isin([0,1,2,3,4])]
                weekend_frame = big_final_data[big_final_data.day_of_week.isin([5,6])]
                
                if weekday_frame.shape[0] > 0:
                    maximum_error_weekdays = (weekday_frame['percentage_error'].nlargest(3).mean())/100
                    if  maximum_error_weekdays < weekday_margin:
                        maximum_error_weekdays = weekday_margin
                else:
                    maximum_error_weekdays = weekday_margin
                    
                if weekend_frame.shape[0] > 0:
                    maximum_error_weekend = (weekend_frame['percentage_error'].nlargest(3).mean())/100
                    if maximum_error_weekend < weekend_margin:
                        maximum_error_weekend = weekend_margin
                    
                else:
                    maximum_error_weekend = weekend_margin
                    
                return maximum_error_weekdays, maximum_error_weekend
            else:
                return weekday_margin,weekend_margin
        else:
            return weekday_margin,weekend_margin
        
    except:
        logging.info('Exception while margin calculation')
        logging.exception('Exception in function margincalculation')
        sys.exit(1)
        
def fixedmargin(marginamount, AMT_mean):
    if (AMT_mean <= 100000):
        margiamt = marginamount + 170000
    elif (AMT_mean > 100000) & (AMT_mean <= 200000):
         margiamt = marginamount + 200000
    elif (AMT_mean > 200000) & (AMT_mean <= 400000):
         margiamt = marginamount + 200000
    elif (AMT_mean > 400000) & (AMT_mean <= 500000):
         margiamt = marginamount + 250000
    elif (AMT_mean > 500000) & (AMT_mean <= 700000):
         margiamt = marginamount + 250000
    elif (AMT_mean > 700000) & (AMT_mean <= 800000):
         margiamt = marginamount + 300000
    elif (AMT_mean > 800000) & (AMT_mean <= 1000000):
         margiamt = marginamount + 300000
    elif (AMT_mean > 1000000) & (AMT_mean <= 1200000):
         margiamt = marginamount + 300000
    elif (AMT_mean > 1200000) & (AMT_mean <= 1500000):
         margiamt = marginamount + 300000
    elif (AMT_mean > 1500000) & (AMT_mean <= 2500000):
         margiamt = marginamount + 300000
    else:
         margiamt = marginamount + 350000
    return margiamt
        
#removing negatives if any in forecast
def remove_negs(ts,train_ds):
    try:
        ts['yhat'] = ts['yhat'].clip_lower(0)
        ts['yhat_lower'] = ts['yhat_lower'].clip_lower(0)
        ts['yhat_upper'] = ts['yhat_upper'].clip_lower(0)
        ts.replace(0,train_ds.mean(),inplace=True) 
        return ts
    except:
        logging.info('Exception while removing negs')
        logging.exception('Exception in function remove_negs')
        sys.exit(1)

#removing outliers from data
def remove_outliers(ATM_Data):
    try:
        mean = np.mean(ATM_Data.transactions_amount.values, axis=0)
        sd = np.std(ATM_Data.transactions_amount.values, axis=0)
        final_list = [x for x in ATM_Data.transactions_amount.values if (x > mean - 2 * sd)]
        ATM_Data = ATM_Data.loc[(ATM_Data['transactions_amount'].isin(final_list))] 
        return ATM_Data
    except:
        logging.info('Exception while removing outliers')
        logging.exception('Exception in function remove_outliers')
        sys.exit(1)
        
#Make Holiday dataframe for fbprophet
def holiday_dataframe(date_matrix,ATM_one_MACHINE_data,train_start,forecast_end):
   
    try:
        
        date_matrix = date_matrix.sort_values(by='time_period_id') 
        date_matrix.index = pd.to_datetime(date_matrix.time_period_id, dayfirst=False)
        date_matrix.drop("time_period_id",axis=1,inplace=True)
        date_matrix = date_matrix.loc[train_start:forecast_end]
        ATM_one_MACHINE_data.index = pd.to_datetime(ATM_one_MACHINE_data.time_period_id, dayfirst=False)
        holidays = pd.DataFrame()
        for column in ["labour_day","new_years_day","new_years_eve","day_before_eid_ul_fitr","eid_ul_fitr_1", 
                       "eid_ul_fitr_2", "eid_ul_fitr_3", "eid_ul_azha_1", "eid_ul_azha_2", "eid_ul_azha_3", 
                       "eid_ul_azha_4","day_before_eid_ul_azha","kashmir_day","pakistan_day","shab_e_meraj",
                       "shab_e_barat","first_july_bank_holiday","independence_day","defence_day","first_day_of_ashura",
                       "chelum","iqbal_day","eid_milad_un_nabi","giarhwin_sharief","christmas_eve","quaid_e_azam_day",
                       "day_after_christmas"]:  
            if pd.Series(date_matrix[date_matrix[column] == '1'].index).isin(ATM_one_MACHINE_data.index).sum() > 0:
                current_holiday = pd.DataFrame({
                        "holiday": column,
                        "ds": date_matrix.loc[date_matrix[column] == '1'].index.tolist()
                        })
                holidays = pd.concat((holidays, current_holiday))
        if holidays.shape[0] > 0:        
            holidays.loc[holidays['holiday'] == 'Eid ul-Azha 1', 'lower_window'] = -9
            holidays.loc[holidays['holiday'] == 'Eid ul-Azha 2', 'lower_window'] = -9
            holidays.loc[holidays['holiday'] == 'Eid ul-Azha 3', 'lower_window'] = -9
            holidays.loc[holidays['holiday'] == 'Eid ul-Azha 4', 'lower_window'] = -9
            holidays.loc[holidays['holiday'] == 'Day before Eid ul-Azha', 'lower_window'] = -9
            holidays.loc[holidays['holiday'] == 'Eid ul-Azha 4', 'upper_window'] = 20
            holidays.loc[holidays['holiday'] == 'Eid ul-Azha 1', 'upper_window'] = 2
            holidays.loc[holidays['holiday'] == 'Eid ul-Fitr 1', 'upper_window'] = 20
            holidays.loc[holidays['holiday'] == 'Eid ul-Fitr 1', 'lower_window'] = -7
            holidays = holidays.fillna(0)
        return holidays  
    except:
        logging.info('Exception while creating holiday dataframe')
        logging.exception('Exception in function holiday_dataframe')
        sys.exit(1)
        
#create insert query for writing in oracle
def create_insert_query_oracle(df, schema_name, table_name):
    try:
        insert_query = 'INSERT ALL '
        for index, row in df.iterrows():
            temp_query = 'INTO {}.{} ({},{},{},{},{},{},{},{},{},{}) values'.format(schema_name, 
                                                                                    table_name,df.columns[0],
                                                                                    df.columns[1],df.columns[2],
                                                                                    df.columns[3],df.columns[4],
                                                                                    df.columns[5],df.columns[6],
                                                                                    df.columns[7],df.columns[8],
                                                                                    df.columns[9])
            temp_query = temp_query+'(\''
            for k in range(0,df.columns.shape[0]):
                val = str(row[(df.columns[k])]).replace("\'","\''")
                temp_query += str(val).replace('%', '%%')
                temp_query += '\'' 
                if k != df.columns.shape[0]-1:
                    temp_query += ','
                    temp_query += '\''      
            temp_query += ')'
            insert_query += temp_query 
        insert_query += ' SELECT 1 FROM DUAL'
        
        return insert_query 
    except:
        logging.info('Exception while creating oracle insert query')
        logging.exception('Exception in function create_insert_query_oracle')
        sys.exit(1)

#Create insert query for writing in hive
def create_insert_query_hive(df, schema_name, table_name):
    try:
        insert_query = 'insert into {}.{} values'.format(schema_name, table_name)
        for index, row in df.iterrows():
            temp_query = '(\''
            for k in range(0,df.columns.shape[0]):
                val = str(row[(df.columns[k])]).replace("\'","\''")
                temp_query += str(val).replace('%', '%%')
                temp_query += '\'' 
                if k != df.columns.shape[0]-1:
                    temp_query += ','
                    temp_query += '\''      
            temp_query += ')'
            insert_query += temp_query + ','
        insert_query = insert_query[:-1]  
        return insert_query
    except:
        logging.info('Exception while creating hive insert query')
        logging.exception('Exception in function create_insert_query_hive')
        sys.exit(1)
        
def atm_wise_execution(atmid,traindata,testdata,datematrix,forecast_end,forecast_start,hyperparamterslist,
                       margin_data,trainingdates,weekday_margin,weekend_margin ):
   
    try:
        atmid = atmid[0]
        ATM_one_MACHINE_data = traindata.loc[traindata.atm_id.astype(int) == int(atmid)]
        ATM_one_MACHINE_data = ATM_one_MACHINE_data.sort_values(by='time_period_id')   
        forecasted_results_previous = margin_data.loc[margin_data.atm_id.astype(int) == int(atmid)]
        forecasted_results_previous = forecasted_results_previous.sort_values(by='forecasted_date')  
        
        #removing outliers
        ATM_one_MACHINE_data = remove_outliers(ATM_one_MACHINE_data)
        ATM_one_MACHINE_data['ds'] = ATM_one_MACHINE_data['time_period_id'].values
        ATM_one_MACHINE_data['y'] = ATM_one_MACHINE_data['transactions_amount'].values    
        
        #reading hyperparamters
        hyperparam = hyperparamterslist.loc[hyperparamterslist.atm_id.astype(float).astype(int) == int(atmid)]
            
        if hyperparam.shape[0] > 0:  
            trainingdate = trainingdates.loc[trainingdates.atm_id.astype(int) == int(atmid)]
            if trainingdate.shape[0] > 0:
                trainingdate.sort_values(by=['timestamp_'],inplace=True, ascending=True)
                #print(trainingdate)
                trainingdate = trainingdate.iloc[-1]
                #print(trainingdate)
                train_start = pd.datetime(trainingdate.train_date_start.year,
                                          trainingdate.train_date_start.month,
                                          trainingdate.train_date_start.day).date()
            else:
                train_start = pd.datetime(int(pd.datetime.now().date().year - 3),1,1).date()
           
            ATM_one_MACHINE_data = ATM_one_MACHINE_data.loc[ATM_one_MACHINE_data.ds >= train_start]
            ATM_one_MACHINE_data = ATM_one_MACHINE_data.loc[ATM_one_MACHINE_data.ds <= (forecast_start+timedelta(days=-1))]
            testdata['ds'] = testdata['time_period_id']
            holidays = holiday_dataframe(datematrix,ATM_one_MACHINE_data,train_start,forecast_end)
            results = prophet_execution(hyperparam,atmid,ATM_one_MACHINE_data,testdata,holidays,train_start,forecast_end,
                                        forecasted_results_previous,weekday_margin,weekend_margin) 
            return results
        else:
            return
    except:
        logging.info('Exception while executing atm_wise function')
        logging.exception('Exception in function atm_wise_execution')
        sys.exit(1)
        
def extractatmids(lst): 
    return [[el] for el in lst] 
        
def main():
    
    try:
        print('Reading Configurations')

        start_time = time.time()
        SENDER_EMAIL  = config.SENDER_EMAIL 
        SENDER_PASSWORD = config.SENDER_PASSWORD 
        SERVER = config.SERVER 
        RECEIVER_EMAIL = config.RECEIVER_EMAIL 
        SUBJECT = config.SUBJECT 

        print(RECEIVER_EMAIL)
        #connection with hive config
        hostname = config.conn_hive['hostname']
        port_dt = config.conn_hive['port_dt']
        auth_dt = config.conn_hive['auth_dt']
        database_dt = config.conn_hive['database_dt']
        kerberos_service_name_dt = config.conn_hive['kerberos_service_name_dt']

        #no of cores
        no_of_cores = config.no_of_cores

        #credentials of writing in hive
        uri_write_in_hive = config.uri_write_in_hive
        schema_name_hive = config.schema_name_hive
        table_name_hive = config.table_name_hive
        fact_table = config.fact_table
        live_atms_table = config.live_atms_table
        branches_table = config.branches_table 
        regions_table = config.regions_table

        #credentials of writing in oracle
        uri_write_in_oracle = config.uri_write_in_oracle
        schema_name_oracle = config.schema_name_oracle
        table_name_oracle = config.table_name_oracle
        oracle_user = config.oracle_usr 
        oracle_pwd = config.oracle_pwd 
        oracle_pwd=base64.b64decode(oracle_pwd).decode('utf-8')
        oracle_host = config.oracle_host
        oracle_port = config.oracle_port
        oracle_service_name = config.oracle_service_name

        #testing dates
        test_date_start = config.test_date_start
        delta_with_training_date = config.delta_with_training_date
        no_of_prediction_days = (config.no_of_prediction_days)

        input_live_atms = config.input_live_atms
        margin_period = config.margin_period
        hyperparameter_table = config.hyperparameter_table
        trainingdatetest_schema_hive = config.trainingdatetest_schema_hive 
        trainingdatetest_table_hive = config.trainingdatetest_table_hive  
        trainingdate_hive = config.trainingdate_hive
        weekday_margin = config.weekday_margin
        weekend_margin = config.weekend_margin
        
        exec_mode = config.exec_mode
        
        os.system('kinit -kt hive.keytab hive/bdakt1node04.abl.com')
        
#         #establish connection with hive for reading
        conn_hive_t24_read = hive.Connection(host=hostname, port=port_dt, auth=auth_dt, database=database_dt,
                                             kerberos_service_name=kerberos_service_name_dt)

        print('Preparing Data')

        #list of live atms
        
        if input_live_atms == 'table':
            testatm = get_live_atms(conn_hive_t24_read,schema_name_hive,live_atms_table)
        else:
            testatm = pd.read_csv(input_live_atms)

        test_atms = list(testatm['atm_id'].astype(int))
        test_atm_ids = extractatmids(test_atms)

        print(len(test_atm_ids))
        
        #test_start_lst = ['1-12-2020']
        
        #for test_date_start in test_start_lst:

        #prepare data
        traindata,testdata,datematrix,forecast_end,forecast_start = prepare_data(conn_hive_t24_read,
                                                                                 test_date_start,delta_with_training_date,
                                                                                 no_of_prediction_days,test_atms,
                                                                                 schema_name_hive,table_name_hive,fact_table)

        #print('margin data')
        #get margin datap
        margin_data = get_margin_data(conn_hive_t24_read,schema_name_hive,table_name_hive,forecast_start,
                                      margin_period,test_atms)

        #print('training dates')
        #get training start date
        trainingdates = get_training_dates(conn_hive_t24_read, trainingdatetest_schema_hive,trainingdate_hive)

        #print('hyper list')
        #read hyperparamters
        hyperparamterslist = read_hyperparamters(conn_hive_t24_read,schema_name_hive,hyperparameter_table)

        # get branches and regions 
        branchesfact = get_branch(conn_hive_t24_read,schema_name_hive,branches_table)
        regionfact = get_region(conn_hive_t24_read,schema_name_hive,regions_table)

        traindata = pd.merge(traindata,branchesfact, on=['branch_id'], how = 'left')
        traindata = pd.merge(traindata,regionfact, on=['region_id'], how = 'left')

        #creating pool 
        pool = mp.Pool(no_of_cores)

        print('Scoring..')
        #atm wise execution
        results = pool.starmap(atm_wise_execution, [(row,traindata,testdata,datematrix,forecast_end,forecast_start,
                                                     hyperparamterslist,margin_data,trainingdates,weekday_margin,weekend_margin) for row in test_atm_ids])
        pool.close()
        print('Pool Closed')

        final_df_write = pd.DataFrame(columns=['forecasted_date','forecasted_amount','marginal_amount','day_of_month','atm_id',
                                               'region_id','region_name','branch_id','branch_name','timestamp_'])

        #concating results in one dataframe
        for k in range(0,len(results)):
            resultsdf = results[k]
            if resultsdf is not None:
                final_df_write = pd.concat([final_df_write,resultsdf])

        if  exec_mode == 'PROD':
            #connecting and writing hive
            print('Writing in hive')
            engine_hive = create_engine(uri_write_in_hive)
            con_hive = engine_hive.connect()                                 
            query_hive = create_insert_query_hive(final_df_write,schema_name_hive,table_name_hive)
            con_hive.execute(query_hive) 
            con_hive.close()

            #connecting and writing in oracle
            print('Writing in oracle')

            schema_name_oracle = config.schema_name_oracle
            oracle_user = config.oracle_usr 
            oracle_pwd = config.oracle_pwd 
            oracle_pwd=base64.b64decode(oracle_pwd).decode('utf-8')
            oracle_host = config.oracle_host
            oracle_port = config.oracle_port
            oracle_service_name = config.oracle_service_name
            tns = """
      (DESCRIPTION =
        (ADDRESS = (PROTOCOL = TCP)(HOST = {})(PORT = {}))
        (CONNECT_DATA =
          (SERVER = DEDICATED)
          (SERVICE_NAME = {})
        )
      )
""".format(oracle_host,oracle_port,oracle_service_name)

            engine_oracle = create_engine('oracle+cx_oracle://%s:%s@%s' % (oracle_user, oracle_pwd, tns))

#             engine_oracle = create_engine('oracle+cx_oracle://%s:%s@%s' % (oracle_user, oracle_pwd, tns))
#             #con_oracle = engine_oracle.connect()

#             #print(final_df_write.dtypes)
            final_df_write['atm_id'] = final_df_write['atm_id'].astype('str')
            final_df_write['forecasted_amount'] = final_df_write['forecasted_amount'].astype('float')
            final_df_write['marginal_amount'] = final_df_write['marginal_amount'].astype('float')
            final_df_write['day_of_month'] = final_df_write['day_of_month'].astype('int')
            final_df_write['branch_id'] = final_df_write['branch_id'].astype('str')
            final_df_write['branch_name'] = final_df_write['branch_name'].astype('str')
            final_df_write['region_id'] = final_df_write['region_id'].astype('str')
            final_df_write['region_name'] = final_df_write['region_name'].astype('str')
            final_df_write['forecasted_date'] = pd.to_datetime(final_df_write['forecasted_date']).dt.date
            final_df_write.to_sql(table_name_oracle,engine_oracle,schema_name_oracle,if_exists = 'append', index = False)    
            #con_oracle.close()
            print(time.time() - start_time)
            message = 'Successful Completion atm_scoring_script.py. Data is available in {} for {} till {}'.format(table_name_oracle,final_df_write.forecasted_date.min(),final_df_write.forecasted_date.max())
            print(SENDER_EMAIL,SENDER_PASSWORD,SERVER,RECEIVER_EMAIL, SUBJECT, message)
            send_message(SENDER_EMAIL,SENDER_PASSWORD,SERVER,RECEIVER_EMAIL, SUBJECT, message)

        print('Successful Completion')
        return final_df_write

    except:
        if  exec_mode == 'PROD':
            error_str = traceback.format_exc()
            send_message(SENDER_EMAIL,SENDER_PASSWORD ,SERVER ,RECEIVER_EMAIL, SUBJECT,error_str)
        logging.info('Exception in main function')
        logging.exception('Exception in main function')
        sys.exit(1)

if __name__ == '__main__': 
    results = main()