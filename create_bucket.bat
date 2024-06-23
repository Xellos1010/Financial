@echo off
setlocal

:: Set AWS credentials and region
set BUCKET_NAME=coinbase-data-ml-signals
aws s3 mb s3://%BUCKET_NAME% --profile SystemDeveloper-Xellos

:end
endlocal

exit /b 0
