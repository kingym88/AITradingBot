#!/bin/bash
echo "Enhanced AI Trading System Startup"
echo "=================================="

# Check if virtual environment exists
if [ ! -d "trading_env" ]; then
    echo "Creating virtual environment..."
    python3 -m venv trading_env
    source trading_env/bin/activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt
else
    echo "Activating virtual environment..."
    source trading_env/bin/activate
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "Configuration file not found!"
    echo "Please copy .env.template to .env and add your API key"
    echo "Example: cp .env.template .env"
    exit 1
fi

# Run the system
echo "Starting Enhanced AI Trading System..."
python enhanced_trading_system.py

echo "System closed."
