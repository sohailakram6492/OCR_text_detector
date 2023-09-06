import cx_Oracle; import pandas as pd; import sys; import datetime
import win32com.client as win32;
# win32com.client.dynamic.Dispatch()
from win32com.client import constants

conn = cx_Oracle.connect('t24cdb/biteam2010@10.133.253.179:1526/cdbekt')
cursor = conn.cursor()
print("\nConnection Established Successfully With Database")
today = datetime.date.today()
file_name = today.strftime("%d-%b-%Y")
password = today.strftime("%Y%m")
var_2 = ''
fg = True
global Branch_no 


def new_account(Account_number):
    
    df_new = pd.DataFrame(cursor.execute(f''' select Branch, oldacctno, newacctno from 
                                t24cdbown.OLD_TO_NEW_ACCTNO_L WHERE NEWACCTNO= {Account_number} '''))
    print("\n----------")
    print(df_new.values)
    print("\n----------")
    if df_new.empty:
        print("\nPlease varify your Account nummber there is no data")
        return 0
    
    global Branch_no 
    Branch_no = int(df_new.iloc[0, 0])
    Account_number = int(df_new.iloc[0, 1])
    new_Account_number = df_new[2]
    
    print(f"\nPlease wait data is in process for Branch: {Branch_no} and Account_Number: {Account_number}")

    cursor.execute(f'''select acctno, branch, postdate, drcrcode, CHQNO, part Particular, LOCEQV AMOUNT
    FROM CDB.CUSTOMERFINHIST@CDB WHERE ACCTNO = {Account_number} AND branch = {Branch_no} ''' )

    columns = [desc[0] for desc in cursor.description]
    
    # Fetch the data
    data = {}
    for col in columns:
        data[col] = []

    for row in cursor:
        for i, col in enumerate(columns):
            data[col].append(row[i])
    # print(data)
    data = pd.DataFrame(data)
    data['POSTDATE'] = data['POSTDATE'].astype(str)
    print("-------------------------")
    print("Account_number", Account_number, "file_name", file_name )
    data.to_excel(f"UniBank Data\{Account_number} ({file_name}).xlsx", index=False)
    return Account_number

def old_account(Account_number,Branch_no):
    print(f"\nPlease wait data is in process for Branch: {Branch_no} and Account_Number: {Account_number}")
    # try:
    cursor.execute(f'''select acctno, branch, postdate, drcrcode, CHQNO, part Particular, LOCEQV AMOUNT
    FROM CDB.CUSTOMERFINHIST@CDB WHERE ACCTNO = {Account_number} AND branch = {Branch_no} '''  )

    print("\n----------")
    # print(df_new.values)
    print("----------")
    # if df_new.empty:
    #     print("\nPlease varify your Account nummber there is no data")
    #     return 0
    # except ValueError:
    #     print("Invalid Branch Code or Account Number, please validate it")

    columns = [desc[0] for desc in cursor.description]
    
    # Fetch the data
    data = {}
    for col in columns:
        data[col] = []

    for row in cursor:
        for i, col in enumerate(columns):
            data[col].append(row[i])
    # print(data)
    
    data = pd.DataFrame(data)
    data['POSTDATE'] = data['POSTDATE'].astype(str)
    data.to_excel(f"UniBank Data\{Account_number} ({file_name}).xlsx", index=False)

def encrypt_xlsx_file(file_path, password):
    excel = win32.Dispatch("Excel.Application")
    # excel =  win32.client.dynamic.Dispatch()
    excel.Visible = False

    wb = excel.Workbooks.Open(file_path)
    wb.Password = password

    excel.DisplayAlerts = False

    # Save the encrypted workbook with password protection
    wb.SaveAs(file_path, None, "", password)
    wb.Close()

    # Quit Excel
    excel.Quit()

# while  input != 'E' or 'e':


def funcation():    
    global Branch_no 
    try:
        if fg == True:
            print("------------------------------------------------------------")
            print("------------------------------------------------------------")
        else: 
            print("\n  *****************************************************************************")
            print("\n  *****************************************************************************")
        Account_number = int(input("\nPlease Enter Account Number to extract Unibank data: "))
        print(len(str(Account_number)))
        if (len(str(Account_number)) <= 14) and (len(str(Account_number)) > 13): 
            print("\nThis is New_Account_number :", Account_number)
            Account_number = new_account(Account_number)
            if Account_number == 0:
                return 0
        elif (len(str(Account_number)) == 8):
            print("\nThis is Old_Account_number :", Account_number)
            def branch_code():
                global Branch_no 
                try:
                    Branch_no = int(input("\nPlease also enter Branch Number  : "))
                    if(len(str(Branch_no))) == 3 :#or len(str(Branch_no)) == 4:
                        try:
                            counts = old_account(Account_number,Branch_no)
                            # print(len(str(counts)))
                            if counts == 0:
                                return 0
                        except ValueError:
                            print("\nInvalid Branch Code please validate it")
                    else:
                        print("\nPlease enter valid Branch Code it must be 3 digits ")
                        branch_code()
                except ValueError:
                    print("\n Please enter valid Branch Code it should not contain alphabets.")
                    branch_code()
                return Branch_no
            Branch_no = branch_code()
            # if counts_2 == 0:
            #     return 0
        else:
            print("\nThis is incorrect Account_number :", Account_number)
            funcation()
        
    except ValueError:
        print("\nThis is incorrect Account_number it should not contain alphabets.")
        funcation()

    file_path = f"G:\\aaa BI Backups\\Unibank\\UniBank Data\\{Account_number} ({file_name}).xlsx"
    print(f"\nData has been downloaded and saved as \"{Account_number} ({file_name}).xlsx\"")
    encrypt_xlsx_file(file_path, password)

    print("\n\t  😄😄😄 \U0001f600\U0001f600\U0001f600","file encripted ","\U0001f600\U0001f600\U0001f600 😄😄😄")

    var_1 = input("\n\nIf you want to varify the Data press 'Y' Otherwise press 'N' ")

    if var_1 == 'Y' or var_1 == 'y':
        cursor.execute(f'''select Branch, ACCTNO, NAME from customerdemography_0722@cdb where ACCTNO = {Account_number} AND branch = {Branch_no}''')
        columns = [desc[0] for desc in cursor.description]
        
        # Fetch the data
        data = {}
        for col in columns:
            data[col] = []

        for row in cursor:
            for i, col in enumerate(columns):
                data[col].append(row[i])
        
        # print("Sohail " , [data])

        count  = pd.DataFrame(cursor.execute(f'''select COUNT(*) FROM CDB.CUSTOMERFINHIST@CDB WHERE ACCTNO={Account_number}AND branch={Branch_no}'''))
        # print(data)
        # data = pd.DataFrame(data)
        # print(count)
        print(f'''\n  Branch Number : {data['BRANCH']}\n  Old Account_Number : {data['ACCTNO']}
        Account_Name : {data['NAME']} \n  Total Number of Record : {count.values}''')
        print("\n")
    # flag = True
    # input = input("Please Press any key to continue or Press E to Exist")
    # fg = False

while var_2 == 'E' or 'e':
    count_fun = funcation()
    fg = False
    if count_fun == 0:
        print("\nPlease try again with correct Account and Branch Number")
        funcation()

cursor.close()
conn.close()
