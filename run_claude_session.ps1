# Script to wake computer, press Enter to unlock, and run Claude automation
# For computers without password lock screens

param (
    [Parameter(Mandatory=$true)]
    [string]$SessionId,
    
    [Parameter(Mandatory=$false)]
    [string]$ProjectRoot = $PSScriptRoot
)

function Write-Log {
    param(
        [string]$Message,
        [string]$Level = "INFO"
    )
    
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logEntry = "[$timestamp] [$Level] $Message"
    
    # Create logs directory if it doesn't exist
    $logDir = Join-Path -Path $ProjectRoot -ChildPath "logs\scheduler"
    if (-not (Test-Path -Path $logDir)) {
        New-Item -Path $logDir -ItemType Directory -Force | Out-Null
    }
    
    # Create log file with today's date
    $logFile = Join-Path -Path $logDir -ChildPath "automation_$(Get-Date -Format 'yyyyMMdd').log"
    
    # Write to log file and console
    Add-Content -Path $logFile -Value $logEntry
    Write-Host $logEntry
}

function Wake-And-Unlock {
    try {
        Write-Log "Waking display..."
        
        # Wake display
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
        
        # Move mouse to ensure screen is active
        $signature = @'
        [DllImport("user32.dll")]
        public static extern bool SetCursorPos(int X, int Y);
'@
        Add-Type -MemberDefinition $signature -Name Win32Functions -Namespace Win32
        [Win32.Win32Functions]::SetCursorPos(500, 500)
        Start-Sleep -Milliseconds 100
        [Win32.Win32Functions]::SetCursorPos(600, 600)
        
        # Check if locked
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
                Write-Log "First unlock attempt didn't work, trying again..." -Level "WARNING"
                # Try a different approach
                $wsh.SendKeys(" ")  # Send space key
                Start-Sleep -Milliseconds 500
                $wsh.SendKeys("{ENTER}")
                
                # Give it another moment
                Start-Sleep -Seconds 3
                $finalCheck = Get-Process logonui -ErrorAction SilentlyContinue
                
                if ($finalCheck) {
                    Write-Log "Computer still locked after second attempt." -Level "ERROR"
                    return $false
                }
            }
            
            Write-Log "Computer successfully unlocked"
        } else {
            Write-Log "Computer is already unlocked"
        }
        
        return $true
    }
    catch {
        Write-Log "Error during wake and unlock: $_" -Level "ERROR"
        return $false
    }
}

function Run-ClaudeAutomation {
    param (
        [string]$SessionId
    )
    
    try {
        Write-Log "Starting Claude automation for session: $SessionId"
        
        # Get the Python executable path
        $pythonPath = (Get-Command python -ErrorAction SilentlyContinue).Source
        
        if (-not $pythonPath) {
            Write-Log "Python executable not found in PATH" -Level "ERROR"
            return $false
        }
        
        # Change to project directory
        Set-Location -Path $ProjectRoot
        
        # Run the automation
        Write-Log "Running: python -m src.main --session $SessionId --debug"
        
        $process = Start-Process -FilePath $pythonPath -ArgumentList "-m", "src.main", "--session", $SessionId, "--debug" -NoNewWindow -PassThru -Wait
        
        if ($process.ExitCode -eq 0) {
            Write-Log "Claude automation completed successfully"
            return $true
        }
        else {
            Write-Log "Claude automation failed with exit code: $($process.ExitCode)" -Level "ERROR"
            return $false
        }
    }
    catch {
        Write-Log "Error running Claude automation: $_" -Level "ERROR"
        return $false
    }
}

# Start logging
Write-Log "===== Starting Automated Claude Session: $SessionId ====="

# Step 1: Wake the computer and unlock screen
$unlockSuccess = Wake-And-Unlock
if (-not $unlockSuccess) {
    Write-Log "Failed to wake/unlock computer, aborting" -Level "ERROR"
    exit 1
}

# Step 2: Wait for desktop to stabilize
Write-Log "Waiting for desktop to fully load..."
Start-Sleep -Seconds 10

# Step 3: Run the Claude automation
$automationSuccess = Run-ClaudeAutomation -SessionId $SessionId
if (-not $automationSuccess) {
    Write-Log "Claude automation failed" -Level "ERROR"
    exit 1
}

# Done
Write-Log "===== Automated Claude Session Completed Successfully ====="
exit 0