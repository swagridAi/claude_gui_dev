# PowerShell script to set up scheduled tasks for Claude automation
# For computers without password lock screens
# IMPORTANT: Must be run as administrator

# Get script path
$scriptPath = Join-Path -Path $PSScriptRoot -ChildPath "run_claude_session.ps1"
if (-not (Test-Path -Path $scriptPath)) {
    Write-Host "Error: run_claude_session.ps1 not found at $scriptPath" -ForegroundColor Red
    exit 1
}

# Function to create a scheduled task
function Create-ClaudeTask {
    param (
        [string]$TaskName,
        [string]$SessionId,
        [string]$Time,
        [string[]]$DaysOfWeek = @("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")
    )
    
    # Create the action to run the script
    $argument = "-ExecutionPolicy Bypass -File `"$scriptPath`" -SessionId `"$SessionId`" -ProjectRoot `"$PSScriptRoot`""
    $action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $argument -WorkingDirectory $PSScriptRoot
    
    # Create a trigger for each day of the week
    $triggers = @()
    foreach ($day in $DaysOfWeek) {
        # Parse the time
        $triggerTime = [DateTime]::Parse($Time)
        $trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek $day -At $triggerTime
        $triggers += $trigger
    }
    
    # Create principal to run with highest privileges (needed for screen unlock)
    $principal = New-ScheduledTaskPrincipal -LogonType S4U -RunLevel Highest -UserId "SYSTEM"
    
    # Set additional settings - wake computer to run
    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -WakeToRun
    
    # Create the task
    try {
        # Delete existing task if it exists
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
        
        # Register with multiple triggers
        $task = Register-ScheduledTask -TaskName $TaskName -Action $action -Settings $settings -Principal $principal -Force
        
        # Add each trigger separately (workaround for multiple triggers)
        foreach ($trigger in $triggers) {
            $task.Triggers.Add($trigger)
        }
        
        # Save the updated task
        $taskFolder = $task.TaskPath
        $taskName = $task.TaskName
        $task | Set-ScheduledTask
        
        Write-Host "Scheduled task '$TaskName' created successfully." -ForegroundColor Green
        Write-Host "The task will run at $Time on: $($DaysOfWeek -join ', ')" -ForegroundColor Cyan
        return $true
    }
    catch {
        Write-Host "Error creating task '$TaskName': $_" -ForegroundColor Red
        return $false
    }
}

# Create the tasks for each session
Write-Host "Setting up Claude Automation Scheduled Tasks..." -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# Morning session - Research Questions (session1)
$taskName1 = "Claude_Research_Questions"
$time1 = "09:00"
$created1 = Create-ClaudeTask -TaskName $taskName1 -SessionId "session1" -Time $time1

# Afternoon session - Code Tasks (session2)
$taskName2 = "Claude_Code_Tasks"
$time2 = "14:00"
$created2 = Create-ClaudeTask -TaskName $taskName2 -SessionId "session2" -Time $time2

Write-Host "==========================================" -ForegroundColor Cyan

# Show summary
if ($created1 -and $created2) {
    Write-Host "All tasks created successfully!" -ForegroundColor Green
} else {
    Write-Host "Some tasks could not be created. Please check the errors above." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "IMPORTANT: These tasks have been configured to:"
Write-Host "1. Wake your computer if it's asleep"
Write-Host "2. Press Enter to unlock your screen"
Write-Host "3. Run the specified Claude automation session"
Write-Host ""
Write-Host "The tasks will run on weekdays (Monday through Friday) at the specified times."
Write-Host "To run on different days, edit the task in Task Scheduler."
Write-Host ""