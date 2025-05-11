# Simple script to wake computer and press Enter to unlock
# Designed for computers without password lock screens

function Write-Log {
    param([string]$Message)
    
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logEntry = "[$timestamp] $Message"
    
    # Create logs directory if it doesn't exist
    $logDir = Join-Path -Path $PSScriptRoot -ChildPath "logs\scheduler"
    if (-not (Test-Path -Path $logDir)) {
        New-Item -Path $logDir -ItemType Directory -Force | Out-Null
    }
    
    # Create log file with today's date
    $logFile = Join-Path -Path $logDir -ChildPath "unlock_$(Get-Date -Format 'yyyyMMdd').log"
    
    # Write to log file and console
    Add-Content -Path $logFile -Value $logEntry
    Write-Host $logEntry
}

Write-Log "Starting computer wake and unlock process"

# Wake monitor if in power saving mode
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;

public class DisplayHelper {
    [DllImport("user32.dll")]
    public static extern int SendMessage(int hWnd, int hMsg, int wParam, int lParam);
    
    public static void WakeDisplay() {
        SendMessage(-1, 0x0112, 0xF170, 2);
    }
}
"@

[DisplayHelper]::WakeDisplay()
Write-Log "Wake display command sent"

# Move mouse to ensure system is responsive
$signature = @'
[DllImport("user32.dll")]
public static extern bool SetCursorPos(int X, int Y);
'@
Add-Type -MemberDefinition $signature -Name Win32Functions -Namespace Win32
[Win32.Win32Functions]::SetCursorPos(500, 500)
Start-Sleep -Milliseconds 100
[Win32.Win32Functions]::SetCursorPos(600, 600)
Write-Log "Mouse moved"

# Check if locked by looking for logon UI process
$isLocked = Get-Process logonui -ErrorAction SilentlyContinue
if ($isLocked) {
    Write-Log "Computer is locked. Sending Enter key to unlock..."
    
    # Wait a moment for the system to be fully responsive
    Start-Sleep -Seconds 2
    
    # Send Enter key to unlock
    $wsh = New-Object -ComObject WScript.Shell
    $wsh.SendKeys("{ENTER}")
    
    # Wait to confirm unlock
    Start-Sleep -Seconds 3
    $stillLocked = Get-Process logonui -ErrorAction SilentlyContinue
    
    if ($stillLocked) {
        Write-Log "First unlock attempt didn't work, trying again..."
        # Try again with a slightly different approach
        $wsh.SendKeys(" ")  # Send space key
        Start-Sleep -Milliseconds 500
        $wsh.SendKeys("{ENTER}")
        
        # Give it another moment
        Start-Sleep -Seconds 3
        $finalCheck = Get-Process logonui -ErrorAction SilentlyContinue
        
        if ($finalCheck) {
            Write-Log "Computer still locked after second attempt."
            exit 1
        }
    }
    
    Write-Log "Computer successfully unlocked"
} else {
    Write-Log "Computer is already unlocked"
}

Write-Log "Wake and unlock process completed successfully"
exit 0