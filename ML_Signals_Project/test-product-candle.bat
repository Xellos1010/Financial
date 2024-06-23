@echo off
setlocal enabledelayedexpansion

:: Function to convert a date to a UNIX timestamp
call :dateToUnix "1 month ago" startTime
call :dateToUnix "now" endTime

:: Display the start and end times
echo Start time (UNIX timestamp): !startTime!
echo End time (UNIX timestamp): !endTime!

:: Pause to review the timestamps
pause

:: Make the API request
curl -X GET "https://api.coinbase.com/api/v3/brokerage/market/products/BTC-USD/candles?start=!startTime!&end=!endTime!&granularity=ONE_DAY" -H "Accept: application/json" -H "Content-Type: application/json"

endlocal
pause
exit /b

:dateToUnix
:: Convert a date to a UNIX timestamp
:: %1 - Date (e.g., "1 month ago" or "now")
:: %2 - Output variable name
setlocal
set "dateStr=%~1"
for /f "tokens=*" %%I in ('powershell -command "[int][double]::Parse((Get-Date -Date '%dateStr%' -UFormat %%s))"') do set "timestamp=%%I"
endlocal & set "%~2=%timestamp%"
exit /b
