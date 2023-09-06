# $removepath = 'G:\Power BI Backups\PBI Backups' 

#clean up the folder
# Get-ChildItem -Path $removepath -File -Recurse | Remove-Item -Verbose

#General Variables
$PortalURL = 'https://advance-analytics.abl.com/Reports'

#Variables for source and target folders
$Target_01= 'G:\Power BI Backups\PBI Backups' 
$Source_01= '/Foldername -->UseHTMLEncodedFolderName' 

#download items
Out-RsRestFolderContent -ReportPortalUri $PortalURL -RsFolder '/' -Recurse -Destination $Target_01
