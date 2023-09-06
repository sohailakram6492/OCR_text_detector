#Write-Output "Welcome....Report name and path is "$args[0]
Import-Module "G:\Power BI Backups\PBI QA\ReportingServicesTools.psd1" -Force
Write-Output "Packages Imported"
#-----------------------------Report parameter
#$ReportName = $args[0]
$ReportName = '/BSG/Demographics'  #just change the report name there and this is applicable for only metics or Shahzad UR Rehman 
#----others parameters 
$ReportPortalUri = "https://advance-analytics.abl.com/Reports"
$FTP="\\192.168.252.171\BSG Data Feed\"
$FtpUser="FTP"
$FtpPass="Allied@12345%$#"
$DbUser="PBI_RO"
$DbPass="PbiRo2022#$"
Write-Output "Variable initilazed"
$parameters = Get-RsRestItemDataModelParameters $ReportName -ReportPortalUri $ReportPortalUri
Write-Output "Existing Sources detail..."
$parameters
$parameterdictionary = @{}
foreach ($parameter in $parameters) { $parameterdictionary.Add($parameter.Name, $parameter); }

foreach ($parameter in $parameters) { if($parameter.name -ne 'CDB') {$parameterdictionary[$parameter.name].Value =$FTP+$parameter.value.Split('\')[-1]}}
Set-RsRestItemDataModelParameters -RsItem $ReportName -ReportPortalUri $ReportPortalUri -DataModelParameters $parameters
Write-Output "Updated Sources detail..."
$parameterdictionary.Values
Write-Output "Data sources credentials setting "
$dataSources = Get-RsRestItemDataSource -RsItem $ReportName -ReportPortalUri $ReportPortalUri
$Dslen=$dataSources.length-1
foreach ($P in 0..$Dslen){
if ($dataSources[$P].ConnectionString.Substring(0,2) -eq '\\')
{
$dataSources[$P].DataModelDataSource.AuthType = 'Windows'
$dataSources[$P].DataModelDataSource.Username =$FtpUser
$dataSources[$P].DataModelDataSource.Secret = $FtpPass
$P++
}
elseif($dataSources[$P].ConnectionString.Substring(0,2) -eq '10')
{
$dataSources[$P].DataModelDataSource.AuthType ='UsernamePassword'
$dataSources[$P].DataModelDataSource.Username =$DbUser
$dataSources[$P].DataModelDataSource.Secret = $DbPass
}
}
Set-RsRestItemDataSource -RsItem $ReportName -ReportPortalUri $ReportPortalUri -RsItemType PowerBIReport -DataSources $datasources
Write-Output "Done... developed by Adnan "
Write-Output "*****"

