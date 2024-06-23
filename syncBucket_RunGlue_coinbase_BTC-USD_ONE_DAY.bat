@echo off
setlocal

:: Set AWS credentials and region
set AWS_PROFILE=SystemDeveloper-Xellos

:: Set the S3 bucket name and the local directory to sync
set BUCKET_NAME=coinbase-data-ml-signals
set LOCAL_DIR=.\ML_Signals_Project\data\coinbase

:: Create the S3 bucket
aws s3 mb s3://%BUCKET_NAME% --profile %AWS_PROFILE%

:: Sync the local directory to the S3 bucket
aws s3 sync %LOCAL_DIR% s3://%BUCKET_NAME% --profile %AWS_PROFILE%

:: Start the Glue crawler to update the data catalog
aws glue start-crawler --name BTC_USD_ONE_DAY_Crawler --profile %AWS_PROFILE%

endlocal
pause
