import win32com.client as win32
def send_email():
    outlook = win32.Dispatch('outlook.application')
    mail = outlook.CreateItem(0)
    mail.To = 'sohail.akram@abl.com'
    mail.Subject = 'Power BI Backup'
    mail.Body = 'Power Bi Backup has been download successfuly Please Place it on VSS'
    mail.Send()
    print("\tEmail has been sended\n")

def send_email_1():
    outlook = win32.Dispatch('outlook.application')
    mail = outlook.CreateItem(0)
    mail.To = 'sohail.akram@abl.com'
    mail.Subject = 'Power BI Backup'
    mail.Body = "\tPlease check Power Bi Backup it's not downloaded till yet\n"
    mail.Send()
    # print("check Power Bi Backup")


# mail.HTMLBody = '<h4> Regards Muhammad Sohail Akram</h4>'
# Officer MIS Reporting <br> ABL Head Office Lahore <br>
