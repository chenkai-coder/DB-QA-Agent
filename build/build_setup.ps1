param(
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DistDir = Join-Path $ProjectRoot "dist\DB-QA-Agent"
$SetupDir = Join-Path $ProjectRoot "setup"
$StageDir = Join-Path $SetupDir "stage"
$ArchivePath = Join-Path $StageDir "app.zip"
$OutputExe = Join-Path $SetupDir "DB-QA-Agent-Setup.exe"
$IExpress = Join-Path $env:WINDIR "System32\iexpress.exe"
$SedPath = Join-Path $SetupDir "installer.sed"

if (!(Test-Path $IExpress)) {
    throw "未找到 iexpress.exe，无法构建安装包。"
}

if (!(Test-Path $SetupDir)) {
    New-Item -ItemType Directory -Path $SetupDir | Out-Null
}

if (!(Test-Path $DistDir)) {
    # If the distribution is not there, we shouldn't continue, but let's assume it's already built for speed!
    & (Join-Path $PSScriptRoot "build_app.ps1") -PythonExe $PythonExe
}

if (!(Test-Path $StageDir)) {
    New-Item -ItemType Directory -Path $StageDir | Out-Null
}

Compress-Archive -Path (Join-Path $DistDir "*") -DestinationPath $ArchivePath -Force

$InstallerPs1 = Join-Path $PSScriptRoot "installer_install.ps1"
# Ensure installer_install.ps1 has UTF-8 BOM so PowerShell on Chinese Windows reads it correctly
$content = [System.IO.File]::ReadAllText($InstallerPs1)
[System.IO.File]::WriteAllText((Join-Path $StageDir "installer_install.ps1"), $content, (New-Object System.Text.UTF8Encoding $true))
Copy-Item -Force (Join-Path $PSScriptRoot "installer_install.cmd") $StageDir

$EscapedStageDir = $StageDir.Replace("\", "\\")
$EscapedOutputExe = $OutputExe.Replace("\", "\\")

$SedContent = @"
[Version]
Class=IEXPRESS
SEDVersion=3
[Options]
PackagePurpose=InstallApp
ShowInstallProgramWindow=1
HideExtractAnimation=0
UseLongFileName=1
InsideCompressed=0
CAB_FixedSize=0
CAB_ResvCodeSigning=0
RebootMode=N
InstallPrompt=
DisplayLicense=
FinishMessage=
TargetName=$EscapedOutputExe
FriendlyName=DB-QA-Agent Setup
AppLaunched=installer_install.cmd
PostInstallCmd=<None>
AdminQuietInstCmd=
UserQuietInstCmd=
SourceFiles=SourceFiles
[SourceFiles]
SourceFiles0=$EscapedStageDir
[SourceFiles0]
%FILE0%=installer_install.cmd
%FILE1%=installer_install.ps1
%FILE2%=app.zip
[Strings]
FILE0=installer_install.cmd
FILE1=installer_install.ps1
FILE2=app.zip
"@

Set-Content -Path $SedPath -Value $SedContent -Encoding ASCII

& $IExpress /N $SedPath | Out-Null

if (!(Test-Path $OutputExe)) {
    throw "安装包生成失败：$OutputExe"
}

Write-Output "SETUP_BUILD_OK"
Write-Output $OutputExe
