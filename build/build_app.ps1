param(
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$AssetsDir = Join-Path $ProjectRoot "assets"
$IconPath = Join-Path $AssetsDir "app.ico"
$SpecPath = Join-Path $ProjectRoot "build\DB_QA_Agent.spec"
$DistDir = Join-Path $ProjectRoot "dist\DB-QA-Agent"
$DataDir = Join-Path $ProjectRoot "data"
$SitePackagesDir = (& $PythonExe -c "import sysconfig; print(sysconfig.get_paths()['purelib'])").Trim()
$EnumPackageDir = Join-Path $SitePackagesDir "enum"
$EnumDistInfoDir = Join-Path $SitePackagesDir "enum34-1.1.10.dist-info"
$EnumPackageBackupDir = Join-Path $SitePackagesDir "_enum34_build_backup_enum"
$EnumDistInfoBackupDir = Join-Path $SitePackagesDir "_enum34_build_backup_distinfo"

if (!(Test-Path $AssetsDir)) {
    New-Item -ItemType Directory -Path $AssetsDir | Out-Null
}

if (!(Test-Path $IconPath)) {
    $SafeIconPath = $IconPath.Replace("\", "\\")
    $IconGenScriptPath = Join-Path $env:TEMP "dbqa_icon_gen.py"
    $IconGenScript = @'
from PIL import Image, ImageDraw
img = Image.new('RGBA', (256, 256), (33, 37, 43, 255))
draw = ImageDraw.Draw(img)
draw.rounded_rectangle((24, 24, 232, 232), radius=36, fill=(38, 121, 255, 255))
draw.rounded_rectangle((52, 52, 204, 204), radius=28, fill=(245, 247, 250, 255))
draw.rectangle((76, 76, 180, 180), fill=(38, 121, 255, 255))
draw.ellipse((134, 134, 214, 214), fill=(28, 196, 120, 255))
img.save(r"__ICON_PATH__", format="ICO", sizes=[(256, 256), (128, 128), (64, 64), (32, 32), (16, 16)])
'@
    $IconGenScript = $IconGenScript.Replace("__ICON_PATH__", $SafeIconPath)
    Set-Content -Path $IconGenScriptPath -Value $IconGenScript -Encoding UTF8
    & $PythonExe $IconGenScriptPath
    if ($LASTEXITCODE -ne 0) {
        throw "图标生成失败，请检查 Pillow 是否可用。"
    }
    if (Test-Path $IconGenScriptPath) {
        Remove-Item -Force $IconGenScriptPath
    }
}

Push-Location $ProjectRoot
try {
    if ((Test-Path $EnumPackageDir) -and !(Test-Path $EnumPackageBackupDir)) {
        Rename-Item -Path $EnumPackageDir -NewName (Split-Path $EnumPackageBackupDir -Leaf)
    }
    if ((Test-Path $EnumDistInfoDir) -and !(Test-Path $EnumDistInfoBackupDir)) {
        Rename-Item -Path $EnumDistInfoDir -NewName (Split-Path $EnumDistInfoBackupDir -Leaf)
    }

    $TclLib = (& $PythonExe -c "import tkinter; print(tkinter.Tcl().eval('info library'))")
    $TkLib = $TclLib.Replace("tcl8.6", "tk8.6")
    $env:TCL_LIBRARY=$TclLib
    $env:TK_LIBRARY=$TkLib

    & $PythonExe -m PyInstaller --noconfirm --clean $SpecPath
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller 打包失败，请先修复上述报错后重试。"
    }

    if (Test-Path $DataDir) {
        $TargetDataDir = Join-Path $DistDir "data"
        if (Test-Path $TargetDataDir) {
            Remove-Item -Recurse -Force $TargetDataDir
        }
        Copy-Item -Recurse -Force $DataDir $TargetDataDir
    }

    Write-Output "APP_BUILD_OK"
    Write-Output $DistDir
}
finally {
    if ((Test-Path $EnumPackageBackupDir) -and !(Test-Path $EnumPackageDir)) {
        Rename-Item -Path $EnumPackageBackupDir -NewName (Split-Path $EnumPackageDir -Leaf)
    }
    if ((Test-Path $EnumDistInfoBackupDir) -and !(Test-Path $EnumDistInfoDir)) {
        Rename-Item -Path $EnumDistInfoBackupDir -NewName (Split-Path $EnumDistInfoDir -Leaf)
    }
    Pop-Location
}
