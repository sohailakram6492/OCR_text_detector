import os, os.path, sys, shutil, re, time, datetime
from email import send_email, send_email_1
# from datetime import date, timedelta
# import glob


flag = True

today = datetime.date.today()
folder_name = today.strftime("%d-%b-%Y")
src_path = r"G:\aaa BI Backups\PBI Backups"
path =   src_path + "\\" + folder_name
if os.path.isdir(path):
    print("\n" + folder_name + " has alread been created\n")
else:
    os.mkdir(path)
    print("folder created Successfully\n")

files = list(os.listdir(src_path))
regex  = re.compile('^[0-31]')

for file in files:
    if not regex.match(file):
        source = r"G:\aaa BI Backups\PBI Backups\\" + file
        try:
            shutil.move(source, path)
        except:
            print(file + " is already present in folder\n")
            continue
        print( file + "  is moved to "+ folder_name + "\n")
        flag = False
    else:
        continue

if flag == False:
    print("Files are download and moved successfuly please place it on VSS\n")
    send_email()

if flag == True:
    print("No file found to move from source to destination Please check it\n")







# print(os.listdir( r"G:\aaa BI Backups\PBI Backups"))
#file_name = os.path.join(src_path, file   
# os.chdir('./../PBI Backups')
# print(os.getcwd())
# os.mkdir(path)
# src_path = r"G:\Power BI Backups\PBI Backups\PBL_Report_1_and_2.pbix"
# dst_path = r"\\lhrnas\QA-Share\vss\CDB\BI Apps\Power BI\Backups"
# shutil.copy(src_path, dst_path)
# shutil.rmtree(r"G:\Power BI Backups\PBI Backups\vssver2.scc")
#print(os.path.dirname(sys.executable))