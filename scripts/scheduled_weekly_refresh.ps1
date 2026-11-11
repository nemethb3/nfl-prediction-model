# Real, unattended Tuesday-morning weekly refresh - runs the exact same
# refresh_weekly.py pipeline (which itself already generates Weekly Picks
# + Credit Picks + the results tracker, as of the 2026-11-05 fix) a human
# has been running manually every week, then commits and pushes the real
# regenerated data - ONLY if the pipeline and both real test suites pass.
#
# Real, deliberate safety choices:
# 1. Never pushes on any failure - a broken/partial pipeline run must
#    never reach origin/main unattended. On failure this script exits
#    non-zero (Task Scheduler's own History tab shows this) and leaves a
#    full log for a human to read before anything is pushed.
# 2. git add is scoped to the exact real paths refresh_weekly.py/
#    generate_juke_picks.py/track_juke_results.py are expected to touch
#    (frontend/src/data/, data/forecast_log/, output/) - never a blanket
#    `git add -A`, so a stray uncommitted manual edit sitting in the
#    working tree from an interactive session is never swept into an
#    unattended commit.
# 3. Runs both the Python and frontend test suites before ever touching
#    git - the same real checks a human has been running by hand before
#    every push this session.
# 4. Each step checks $LASTEXITCODE itself, immediately after its own
#    native command, before any other cmdlet (even Pop-Location) can run -
#    $LASTEXITCODE is real, global PowerShell state that the next cmdlet
#    can silently overwrite, so it's never allowed to cross a function/
#    scriptblock boundary unchecked.

# Real bug fixed here (2026-11-13, found running this script for real):
# "Stop" here, combined with *>> redirecting a native exe's stderr
# stream, makes PowerShell wrap EVERY stderr line as a terminating
# NativeCommandError - including completely benign progress output (Jest
# writes "PASS ..." to stderr), aborting the run on a real pass. Every
# step below already checks $LASTEXITCODE explicitly and throws its own
# real error - "Continue" lets that real check be the only thing that
# decides pass/fail, not an incidental stderr line.
$ErrorActionPreference = "Continue"

$ProjectRoot = "D:\Projects\NFL_Model"
$Python = "D:\venvs\nfl-model\Scripts\python.exe"
$LogDir = Join-Path $ProjectRoot "data\scheduled_run_logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogFile = Join-Path $LogDir "tuesday_refresh_$Timestamp.log"

function Write-Log {
    param([string]$Message)
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $Message"
    Add-Content -Path $LogFile -Value $line
    Write-Host $line
}

Set-Location $ProjectRoot
Write-Log "=== Scheduled Tuesday weekly refresh starting ==="

try {
    Write-Log "--- refresh_weekly.py ---"
    & $Python (Join-Path $ProjectRoot "src\refresh_weekly.py") *>> $LogFile
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) { throw "refresh_weekly.py failed (exit code $exitCode) - see $LogFile" }
    Write-Log "refresh_weekly.py: OK"

    Write-Log "--- Python test suite ---"
    & $Python -m pytest (Join-Path $ProjectRoot "tests\test_juke_picks.py") -q *>> $LogFile
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) { throw "Python test suite failed (exit code $exitCode) - see $LogFile" }
    Write-Log "Python test suite: OK"

    Write-Log "--- Frontend test suite ---"
    Push-Location (Join-Path $ProjectRoot "frontend")
    $env:CI = "true"
    & npx react-scripts test --watchAll=false *>> $LogFile
    $exitCode = $LASTEXITCODE
    Pop-Location
    if ($exitCode -ne 0) { throw "Frontend test suite failed (exit code $exitCode) - see $LogFile" }
    Write-Log "Frontend test suite: OK"

    Write-Log "--- Staging real data changes ---"
    git -C $ProjectRoot add frontend/src/data/ data/forecast_log/ output/ *>> $LogFile
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) { throw "git add failed (exit code $exitCode) - see $LogFile" }

    $changed = git -C $ProjectRoot status --porcelain -- frontend/src/data/ data/forecast_log/ output/
    if (-not $changed) {
        Write-Log "No real changes to commit this run - nothing to push."
    } else {
        $commitMsg = "Automated weekly refresh - $(Get-Date -Format 'yyyy-MM-dd')`n`n" +
            "Scheduled Tuesday-morning run: refresh_weekly.py (Weekly Games/Fantasy/" +
            "Season Projections/Accuracy Tracker/Weekly Summary/Betting Analysis/" +
            "Weekly Picks/Credit Picks), real test suites both green before push.`n`n" +
            "Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
        git -C $ProjectRoot commit -m $commitMsg *>> $LogFile
        $exitCode = $LASTEXITCODE
        if ($exitCode -ne 0) { throw "git commit failed (exit code $exitCode) - see $LogFile" }

        Write-Log "--- Pushing to origin/main ---"
        git -C $ProjectRoot push origin main *>> $LogFile
        $exitCode = $LASTEXITCODE
        if ($exitCode -ne 0) { throw "git push failed (exit code $exitCode) - see $LogFile" }
        Write-Log "Push succeeded."
    }

    Write-Log "=== Scheduled refresh completed successfully ==="
    exit 0
} catch {
    Write-Log "FAILED: $_"
    Write-Log "=== Scheduled refresh ABORTED - nothing pushed ==="
    exit 1
}
