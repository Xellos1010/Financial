@echo off
setlocal enabledelayedexpansion

:: Set the path to the candles directory
set "candlesDir=.\data\coinbase\candles"

:: Iterate over all CSV files in the candles directory
for %%f in (%candlesDir%\*.csv) do (
    :: Extract the filename without extension
    set "filename=%%~nf"

    :: Extract the product_id and granularity from the filename
    for /f "tokens=1,2,3 delims=_" %%a in ("!filename!") do (
        set "product_id=%%a"
        set "granularity=%%b_%%c"
    )
    @REM echo !filename!
    @REM echo !product_id!
    @REM echo !granularity!

    :: Set the destination directory
    set "destDir=%candlesDir%\!product_id!\!granularity!"

    :: Create the destination directory if it doesn't exist
    if not exist "!destDir!" (
        mkdir "!destDir!"
    )

    :: Move the file to the destination directory
    move "%%f" "!destDir!\"
    echo Moved %%f to !destDir!
)

echo All files have been moved successfully.
endlocal
pause
