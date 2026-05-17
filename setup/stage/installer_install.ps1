param(
    [string]$SourceDir
)

try {
    if (-not $SourceDir) {
        $SourceDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    }
    
    # Force remove any stray double quotes that might have bleed in from cmd
    $SourceDir = $SourceDir.Trim('"')

    $AppName = "DB-QA-Agent"

    Add-Type -AssemblyName System.Windows.Forms
    Add-Type -AssemblyName PresentationFramework

    $FolderBrowser = New-Object System.Windows.Forms.FolderBrowserDialog
    $FolderBrowser.Description = "Please select the installation directory for ${AppName}:"
    $FolderBrowser.SelectedPath = $env:LOCALAPPDATA

    $Result = $FolderBrowser.ShowDialog()
    if ($Result -ne [System.Windows.Forms.DialogResult]::OK) {
        exit 0
    }

    $SelectedPath = $FolderBrowser.SelectedPath
    $InstallRoot = Join-Path $SelectedPath $AppName

    $AppSource = Join-Path $SourceDir "app"
    $AppArchive = Join-Path $SourceDir "app.zip"
    $AppExe = Join-Path $InstallRoot "DB-QA-Agent.exe"
    $DesktopShortcut = Join-Path ([Environment]::GetFolderPath("Desktop")) "$AppName.lnk"
    $StartMenuDir = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\$AppName"
    $StartMenuShortcut = Join-Path $StartMenuDir "$AppName.lnk"

    if (!(Test-Path $AppSource) -and !(Test-Path $AppArchive)) {
        throw "Application files not found: $AppSource or $AppArchive"
    }

    if (Test-Path $InstallRoot) {
        Remove-Item -Recurse -Force $InstallRoot
    }

    New-Item -ItemType Directory -Path $InstallRoot | Out-Null
    if (Test-Path $AppArchive) {
        Add-Type -AssemblyName System.IO.Compression.FileSystem
        [System.IO.Compression.ZipFile]::ExtractToDirectory($AppArchive, $InstallRoot)
    }
    else {
        Copy-Item -Recurse -Force (Join-Path $AppSource "*") $InstallRoot
    }

    if (!(Test-Path $StartMenuDir)) {
        New-Item -ItemType Directory -Path $StartMenuDir | Out-Null
    }

    $WshShell = New-Object -ComObject WScript.Shell

    $Desktop = $WshShell.CreateShortcut($DesktopShortcut)
    $Desktop.TargetPath = $AppExe
    $Desktop.WorkingDirectory = $InstallRoot
    $Desktop.IconLocation = "$AppExe,0"
    $Desktop.Save()

    $StartMenu = $WshShell.CreateShortcut($StartMenuShortcut)
    $StartMenu.TargetPath = $AppExe
    $StartMenu.WorkingDirectory = $InstallRoot
    $StartMenu.IconLocation = "$AppExe,0"
    $StartMenu.Save()

    [System.Windows.MessageBox]::Show(
        "DB-QA-Agent has been installed successfully!`nInstallation directory: $InstallRoot`nA desktop shortcut has been created.",
        "Installation Complete",
        [System.Windows.MessageBoxButton]::OK,
        [System.Windows.MessageBoxImage]::Information
    ) | Out-Null
}
catch {
    Add-Type -AssemblyName PresentationFramework
    [System.Windows.MessageBox]::Show(
        "An error occurred during installation:`n$($_.Exception.Message)`n$($_.InvocationInfo.PositionMessage)",
        "Installation Failed",
        [System.Windows.MessageBoxButton]::OK,
        [System.Windows.MessageBoxImage]::Error
    ) | Out-Null
    exit 1
}
