@echo off
setlocal

:: Create the main project directory
mkdir ML_Signals_Project
cd ML_Signals_Project

:: Create subdirectories
mkdir data
mkdir data\raw
mkdir data\processed
mkdir data\coinbase
mkdir data\coinbase\products
mkdir data\coinbase\candles
mkdir models
mkdir notebooks
mkdir scripts
mkdir scripts\coinbase
mkdir scripts\coinbase\listProducts
mkdir scripts\coinbase\getProductCandles
mkdir logs
mkdir config

echo Project directories created successfully.
endlocal
pause
