import cx_Oracle; import pandas as pd; import sys
conn = cx_Oracle.connect('t24cdb/biteam2010@10.133.253.179:1526/cdbekt')
cursor = conn.cursor()

# df_new = pd.DataFrame(cursor.execute(''' select Branch, oldacctno, newacctno from 
#                                          t24cdbown.OLD_TO_NEW_ACCTNO_L WHERE NEWACCTNO= '0010021758350012' '''))

def new_account(Account_number):
    df_new = pd.DataFrame(cursor.execute(f'''select Branch, oldacctno, newacctno from 
                                t24cdbown.OLD_TO_NEW_ACCTNO_L WHERE NEWACCTNO= {Account_number} '''))
    Branch_no = int(df_new[0].values)
    Account_no = int(df_new[1].values)
    new_Account_no = df_new[2]

    print(f"Please wait data is in process for Branch: {Branch_no} and Account_Number: {Account_no}")

    df_old = pd.DataFrame(cursor.execute(f'''select acctno, branch, postdate, drcrcode, CHQNO, part Particular, LOCEQV AMOUNT
        FROM CDB.CUSTOMERFINHIST@CDB WHERE ACCTNO = {Account_no} AND branch = {Branch_no} ''' ))
    columns = [desc[0] for desc in cursor.description]
   
    # print(df_old.columns('ACCTNO','BRANCH','POSTDATE','CHQNO','PARTICULAR','AMOUNT'))

# new_account(10002767890054)
def new_a(Account_number):
    df_new = pd.DataFrame(cursor.execute(f'''select Branch, oldacctno, newacctno from 
                                t24cdbown.OLD_TO_NEW_ACCTNO_L WHERE NEWACCTNO= {Account_number} '''))
    Branch_no = int(df_new[0].values)
    Account_no = int(df_new[1].values)
    new_Account_no = df_new[2]

    print(f"Please wait data is in process for Branch: {Branch_no} and Account_Number: {Account_no}")

    cursor.execute(f'''select acctno, branch, postdate, drcrcode, CHQNO, part Particular, LOCEQV AMOUNT
    FROM CDB.CUSTOMERFINHIST@CDB WHERE ACCTNO = {Account_no} AND branch = {Branch_no} ''' )

    columns = [desc[0] for desc in cursor.description]
    
    # Fetch the data
    data = {}
    for col in columns:
        data[col] = []

    for row in cursor:
        for i, col in enumerate(columns):
            data[col].append(row[i])
    print(data)

new_a(10002767890054)

# Branch_no = df_new[0]
# Account_no = df_new[1]
# new_Account_no = df_new[2]
