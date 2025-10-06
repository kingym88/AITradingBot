@echo off
echo Enhanced AI Trading System Startup
echo ==================================

REM Check if virtual environment exists
if not exist "trading_env" (
    echo Creating virtual environment...
    python -m venv trading_env
    call trading_env\Scripts\activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt
) else (
    echo Activating virtual environment...
    call trading_env\Scripts\activate
)

REM Check if .env file exists
if not exist ".env" (
    echo Configuration file not found!
    echo Please copy .env.template to .env and add your API key
    echo Example: copy .env.template .env
    pause
    exit /b 1
)

REM Run the system
echo Starting Enhanced AI Trading System...
python enhanced_trading_system.py

echo.
echo System closed. Press any key to exit.
pause > nul
