@echo off
setlocal enabledelayedexpansion

:: Path to your products CSV file
set "productsCsvPath=.\data\coinbase\products\products.csv"

:: Granularities
set granularities=ONE_MINUTE FIVE_MINUTE FIFTEEN_MINUTE THIRTY_MINUTE ONE_HOUR TWO_HOUR SIX_HOUR ONE_DAY

:: Check if the products CSV file exists
if not exist "%productsCsvPath%" (
    echo Products CSV file not found at %productsCsvPath%
    exit /b 1
)

:: Iterate over each line in the CSV file (skipping the header)
for /f "skip=1 tokens=1 delims=," %%A in (%productsCsvPath%) do (
    set "product_id=%%A"
    :: Create directories for each granularity
    for %%G in (%granularities%) do (
        set "directory=.\data\coinbase\candles\!product_id!\%%G"
        if not exist "!directory!" (
            mkdir "!directory!"
            echo Created directory: !directory!
        ) else (
            echo Directory already exists: !directory!
        )
        set "directory=.\parameter-optimization\coinbase\!product_id!\%%G\RSI"
        if not exist "!directory!" (
            mkdir "!directory!"
            echo Created directory: !directory!
        ) else (
            echo Directory already exists: !directory!
        )
    )
)


echo All directories created successfully.
endlocal
pause
