# Orchestrator watchdog.
# The bot once silently died and nobody noticed for 3 days with an open
# position. This script checks the orchestrator log's last-write time; if it's
# stale, it starts the orchestrator. The bot's OS-level singleton lock makes
# this safe: if the bot is actually alive, the new instance exits immediately.
#
# Register (run once from a normal PowerShell):
#   schtasks /Create /TN "futbot-watchdog" /TR "powershell -NoProfile -ExecutionPolicy Bypass -File F:\trade_claude\scripts\watchdog.ps1" /SC MINUTE /MO 15 /F
# Remove:
#   schtasks /Delete /TN "futbot-watchdog" /F

$LogPath   = "F:\trade_claude\data\logs\orchestrator.log"
$StaleMin  = 45   # orchestrator logs at least every 5-min monitor cycle; 45 min = clearly dead
$Python    = "python"
$WorkDir   = "F:\trade_claude"

$stale = $true
if (Test-Path $LogPath) {
    $age = (Get-Date) - (Get-Item $LogPath).LastWriteTime
    $stale = $age.TotalMinutes -gt $StaleMin
}

if ($stale) {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path "F:\trade_claude\data\logs\watchdog.log" -Value "$stamp watchdog: log stale, starting orchestrator" -Encoding utf8
    Start-Process -FilePath $Python -ArgumentList "-m", "futbot.orchestrator.main" -WorkingDirectory $WorkDir -WindowStyle Minimized
}
