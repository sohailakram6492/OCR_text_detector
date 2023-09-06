print("Script Execution Started")
import os, os.path, sys, shutil, re, time, datetime
from datetime import date
from email import send_email, send_email_1
list_1= []
backup_days = []
flag = False

today = datetime.date.today()
folder_name = today.strftime("%d-%b-%Y")
start_Date = datetime.datetime.strptime(folder_name, "%d-%b-%Y")
end_date = start_Date + datetime.timedelta(days=3)
end_date = datetime.datetime.strftime(end_date, "%d-%b-%Y")

start_dat = str(folder_name)
end_dat = str(end_date)
start_ = int(start_dat[:2])
end_ = int(end_dat[:2])

for i in range(-3,2):
    backup_day =  start_Date + datetime.timedelta(days=i)
    backup_days.append(datetime.datetime.strftime(backup_day, "%d-%b-%Y"))

src_path = r"G:\aaa BI Backups\PBI Backups"

files = str(os.listdir(src_path))
# print(files)
re = re.finditer(r"\d{2}-\w{3}-\d{4}", files)
# print(re)

for r in re:
    list_1.append(r.group())
print(list_1)
# print(backup_days)

for i in backup_days:
    for j in list_1:
        if j == i:
            print("Backup folder has been founded")
            print(j)
            flag = True 
            break 

if flag == False:
    send_email_1()
    print("Please check Power Bi backup there is any error folder wasn't created yet")


# print(backup_days)
# print(start_)
# print(end_)
# print(folder_name)
# print(end_date)
# print(backup_days)
# print(list_1)
#     if i == re:
#         # send_email()
#     else:
#         # send_email_1()
