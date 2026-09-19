param(
    [Parameter(Mandatory=$true)][string]$Msi,
    [Parameter(Mandatory=$true)][string]$Stage,
    [Parameter(Mandatory=$true)][string]$ExpectedVersion
)
$ErrorActionPreference = 'Stop'
$package = Get-Item -LiteralPath $Msi
if ($package.Length -ge 2GB) { throw 'MSI exceeds the Windows Installer 2GB package limit' }
$installer = New-Object -ComObject WindowsInstaller.Installer
$db = $installer.OpenDatabase($package.FullName, 0)
function Read-Rows([string]$sql, [int]$columns) {
    $view = $db.OpenView($sql)
    [void]$view.Execute()
    try {
        while ($record = $view.Fetch()) {
            $values = @(for ($i = 1; $i -le $columns; $i++) { $record.StringData($i) })
            ,$values
        }
    } finally { [void]$view.Close() }
}
$properties = @{}
foreach ($row in (Read-Rows 'SELECT `Property`, `Value` FROM `Property`' 2)) { $properties[$row[0]] = $row[1] }
if ($properties.ProductVersion -ne $ExpectedVersion) { throw 'Wrong ProductVersion' }
if ($properties.UpgradeCode -ne '{ABC1243C-CA79-481B-BF17-22C8A491A6D6}') { throw 'UpgradeCode changed' }
$files = @(Read-Rows 'SELECT `File`, `FileName`, `FileSize`, `Version` FROM `File`' 4)
$staged = @(Get-ChildItem -LiteralPath $Stage -File -Recurse)
if ($files.Count -ne $staged.Count) { throw "File count mismatch: MSI=$($files.Count) stage=$($staged.Count)" }
$bytes = ($files | ForEach-Object { [long]$_[2] } | Measure-Object -Sum).Sum
if ($bytes -ne ($staged | Measure-Object Length -Sum).Sum) { throw 'Payload byte count mismatch' }
$exe = @($files | Where-Object { ($_[1] -split '\|')[-1] -eq 'rvc-app.exe' })
if ($exe.Count -ne 1 -or $exe[0][3] -notlike "$ExpectedVersion*") { throw 'Wrong packaged exe version' }
$cabinets = @(Read-Rows 'SELECT `Cabinet` FROM `Media`' 1)
foreach ($row in $cabinets) {
    $name = $row[0]
    if (!$name -or !$name.StartsWith('#')) { throw 'All cabinets must be embedded in the single MSI' }
}
[pscustomobject]@{
    product_version=$properties.ProductVersion
    upgrade_code=$properties.UpgradeCode
    files=$files.Count
    payload_bytes=$bytes
    embedded_cabinets=$cabinets.Count
    external_cabinets=0
    msi_bytes=$package.Length
    app_file_id=$exe[0][0]
    passed=$true
} | ConvertTo-Json
