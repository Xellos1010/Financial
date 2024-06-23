@echo off
setlocal

set AWS_PROFILE=SystemDeveloper-Xellos

:: Set the Glue database name and crawler name
set DATABASE_NAME=coinbase_data
set CRAWLER_NAME=BTC_USD_ONE_DAY_Crawler
set BUCKET_NAME=coinbase-data-ml-signals
set S3_PATH=candles/BTC-USD/ONE_DAY

:: Create a Glue database
aws glue create-database --database-input "{\"Name\":\"%DATABASE_NAME%\"}" --profile %AWS_PROFILE%

:: Create a Glue crawler
aws glue create-crawler ^
    --name %CRAWLER_NAME% ^
    --role GlueServiceRole ^
    --database-name %DATABASE_NAME% ^
    --targets "{\"S3Targets\":[{\"Path\":\"s3://%BUCKET_NAME%/%S3_PATH%/\"}]}" --profile %AWS_PROFILE%

:: Start the Glue crawler
aws glue start-crawler --name %CRAWLER_NAME% --profile %AWS_PROFILE%

endlocal
pause