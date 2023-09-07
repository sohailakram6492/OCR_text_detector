uri_write_in_hive = 'hive://abldev@bdakt1node04.abl.com:10000/default?auth=KERBEROS&kerberos_service_name=hive'
schema_name_hive = 'aadl'
table_name_hive = 'co_atm_forecasted_results'
hyperparameter_table = 'co_atm_hyperparamter'
live_atms_table = 'co_live_atms'
fact_table= 'cash_opt_atm_fact'
tbl_atm = 'tblatm'
branches_table = 'branches'
regions_table = 'regions'
exec_mode = 'PROD' #either TEST or PROD. TEST will not generate alerts and not write in database.

conn_hive = dict(
    hostname = "bdakt1node04.abl.com",
    port_dt = "10000",
    auth_dt = "KERBEROS",
    database_dt = "t24_dev",
    kerberos_service_name_dt = "hive"
)
no_of_cores = 70
uri_write_in_oracle = 'oracle+cx_oracle://aa_bda:AA_BDA123@10.133.253.179:1526/?service_name=cdbplskt'
oracle_usr = 'aa_bda'
oracle_pwd = 'QUFfQkRBMTIz'
oracle_host = '10.133.253.179'
oracle_port = '1526'
oracle_service_name = 'cdbplskt'
schema_name_oracle = 'AA_BDA'
table_name_oracle = 'ATM_FORECASTED_RESULTS'

trainingdatetest_schema_hive = 'aadl'
trainingdatetest_table_hive = 'co_trainingdataset'
trainingdate_hive = 'co_trainingdate'
trainingdate_selection_weeks = 4
traindate_selection_atms = 'train_atms_list.csv' #set 'table' as default if not using a csv file [must only enter csv]

start_date_replenish = '7-5-2021'
end_date_replenish = '17-5-2021'
replenish_mode = 'AUTO' #either AUTO or 'MANUAL'
replenishment_table = 'ATM_REPLENISHMENT'
monitoring_table = 'co_atm_monitoring_report'
minimum_amount = 100000

test_date_start = '1-1-2014' # must set to default value i.e. 1-1-2014 if wants to run the normal cycle 
                             # else enter date in the format day-month-year
delta_with_training_date = 2
no_of_prediction_days = 7 #value can't be 0
margin_period = 90
input_live_atms = 'table' #set 'table' as default if not using a csv file [must only enter csv]
weekday_margin = 0.24
weekend_margin = 0.24

hyperparamter_test_week_end = '22-03-2021'
hyperparameter_tuning_weeks = 4
train_atms = 'train_atms_list.csv' # set 'table' as default if not using a csv file [must only enter csv]

restore_point_folder = 'training_weeks_data'
restore_point = 'default' # change default to csv e.g week 3.csv

SENDER_EMAIL = 'CDB.Alert@abl.com'
SENDER_PASSWORD = 'some_password'
SERVER = '10.133.43.5:25'
RECEIVER_EMAIL = ['Ahmad.Abrar@abl.com','Mansoor.Khan3@abl.com','Waqas.Ahmed5@abl.com','Sada.Mustafeed@abl.com','ali.raza2@abl.com']
SUBJECT = 'Cash Optimization ATM | ABL BDA'



