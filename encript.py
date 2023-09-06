import win32com.client as win32
from win32com.client import constants
import cx_Oracle; import pandas as pd; import sys; import datetime
import cx_Oracle; import pandas as pd; import sys; import datetime
import win32com.client as win32;
from win32com.client import constants

conn = cx_Oracle.connect('t24cdb/biteam2010@10.133.253.179:1526/cdbekt')
cursor = conn.cursor()
print("\n\tConnection Established Successfully With Database")
today = datetime.date.today()
file_name = today.strftime("%d-%b-%Y")
password = today.strftime("%Y%m")
global Account_number

# def encrypt_xlsx_file(file_path, password):
#     excel = win32.Dispatch("Excel.Application")
#     excel.Visible = False

#     wb = excel.Workbooks.Open(file_path)
#     wb.Password = password

#     # Save the encrypted workbook with password protection
#     wb.SaveAs(file_path, None, "", password)
#     wb.Close()

#     # Quit Excel
#     excel.Quit()

# # Example usage
# file_path = r"G:\aaa BI Backups\Unibank\12003627 15-May-2023.xlsx"
# today = datetime.date.today()
# file_name = today.strftime("%d-%b-%Y")
# password = today.strftime("%Y%m")
# print(password)


# ____________________________________________________
var_1 = input("\n\nIf you want to varify the Data press 'Y' Otherwise press 'N' ")
if var_1 == 'Y' or var_1 == 'y':
    cursor.execute(f'''select Branch, ACCTNO, NAME from customerdemography_0722@cdb where ACCTNO = '12002319' AND branch = 531''');
    count  = pd.DataFrame(cursor.execute(f''' select COUNT(*) FROM CDB.CUSTOMERFINHIST@CDB WHERE ACCTNO = '12002319' AND branch = 531 '''))
    columns = [desc[0] for desc in cursor.description]
    
    # Fetch the data
    data = {}
    for col in columns:
        data[col] = []

    for row in cursor:
        for i, col in enumerate(columns):
            data[col].append(row[i])
    # print(data)
    # data = pd.DataFrame(data)
    print(f'''Total Record : {count}''')




