# Enhanced AI Trading System - Beginner Setup Guide

## 🚀 Quick Start Guide

This enhanced AI trading system uses Perplexity AI to make intelligent stock trading decisions on micro-cap stocks (market cap < $300M). The system has been completely rewritten to be production-ready with proper error handling, database storage, and risk management.

### 📋 Prerequisites

1. **Python 3.11 or higher** installed on your computer
2. **Perplexity API Key** (get from https://www.perplexity.ai/)
3. **Basic command line knowledge** (don't worry, we'll guide you!)

### 🛠️ Installation Steps

#### Step 1: Extract and Navigate to the Project

```bash
# Extract the downloaded zip file
# Navigate to the extracted folder
cd enhanced-ai-trading-system

# On Windows, use Command Prompt or PowerShell
# On Mac/Linux, use Terminal
```

#### Step 2: Set Up Python Environment

```bash
# Create a virtual environment (recommended)
python -m venv trading_env

# Activate the environment
# On Windows:
trading_env\Scripts\activate
# On Mac/Linux:
source trading_env/bin/activate

# Upgrade pip
python -m pip install --upgrade pip
```

#### Step 3: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# If you encounter any errors, try installing packages individually:
# pip install pandas numpy yfinance requests python-dotenv sqlalchemy
```

#### Step 4: Configuration Setup

1. **Copy the environment template:**
   ```bash
   # On Windows:
   copy .env.template .env
   # On Mac/Linux:
   cp .env.template .env
   ```

2. **Edit the .env file:**
   Open the `.env` file in any text editor (Notepad, VS Code, etc.) and fill in your details:

   ```
   PERPLEXITY_API_KEY=your_actual_api_key_here
   INITIAL_CAPITAL=100.0
   # ... (other settings can remain as default)
   ```

#### Step 5: Get Your Perplexity API Key

1. Go to https://www.perplexity.ai/
2. Sign up for an account
3. Navigate to API settings
4. Generate an API key
5. Copy the key and paste it in your `.env` file

#### Step 6: Test the Installation

```bash
# Run the integration test
python test_system.py

# You should see:
# ✅ All imports successful
# ✅ Configuration loaded
# ✅ Data manager initialized  
# ✅ Database manager initialized
# 🎉 Integration test passed!
```

#### Step 7: Run the System

```bash
# Start the enhanced trading system
python enhanced_trading_system.py
```

### 🎮 Using the System

When you run the system, you'll see an interactive menu:

```
Enhanced AI Trading System - Interactive Mode
==================================================
1. Run Daily Update
2. Show Portfolio Summary  
3. Show Performance Report
4. Manual Trade
5. Exit
```

#### Menu Options Explained:

1. **Run Daily Update**: 
   - Updates all stock prices
   - Gets AI recommendations from Perplexity
   - Executes recommended trades (simulated)
   - Performs weekly research (on Fridays)

2. **Show Portfolio Summary**:
   - Displays current portfolio value
   - Shows all positions and their P&L
   - Cash available for trading

3. **Show Performance Report**:
   - Performance metrics (returns, Sharpe ratio, etc.)
   - Trade history
   - Win/loss statistics

4. **Manual Trade**: 
   - Currently disabled in demo mode
   - In production, would allow manual override

5. **Exit**: 
   - Safely closes the system

### 📊 Understanding the Output

#### Portfolio Summary Example:
```
Portfolio Summary:
Total Value: $105.50
Cash: $45.00
Invested: $60.50
Total Return: 5.50%
Positions: 2

Current Positions:
  ABIO: 100 shares @ $2.25 (P&L: 12.50%)
  ACRX: 50 shares @ $3.10 (P&L: -2.30%)
```

#### AI Recommendations:
The system will log AI recommendations like:
```
[INFO] Processing recommendation: BUY ADMA
[INFO] BUY executed: 75 shares of ADMA at $4.20
```

### ⚙️ Configuration Options

You can modify the `.env` file to adjust system behavior:

- `INITIAL_CAPITAL`: Starting amount (default: $100)
- `MAX_POSITION_SIZE`: Maximum % per stock (default: 30%)
- `STOP_LOSS_PERCENTAGE`: Auto-sell trigger (default: 15% loss)
- `TAKE_PROFIT_PERCENTAGE`: Profit target (default: 25% gain)
- `MAX_POSITIONS`: Maximum number of stocks (default: 10)

### 📁 File Structure

```
enhanced-ai-trading-system/
├── enhanced_trading_system.py  # Main application
├── config.py                   # Configuration management
├── data_manager.py            # Stock data handling
├── perplexity_client.py       # AI API integration
├── database.py                # Data storage
├── test_system.py             # Testing suite
├── requirements.txt           # Dependencies
├── .env.template             # Configuration template
├── .env                      # Your configuration (create this)
├── README.md                 # This guide
├── logs/                     # System logs
├── data/                     # Data files
├── reports/                  # Performance reports
└── trading_system.db         # SQLite database (created automatically)
```

### 🔧 Troubleshooting

#### Common Issues:

1. **"Module not found" errors**:
   ```bash
   # Make sure virtual environment is activated
   pip install -r requirements.txt
   ```

2. **API key errors**:
   - Double-check your Perplexity API key in `.env`
   - Ensure no extra spaces around the key

3. **Database errors**:
   - The system creates `trading_system.db` automatically
   - Delete it and restart if corrupted

4. **Network/data errors**:
   - Check internet connection
   - Yahoo Finance sometimes has rate limits

#### Getting Help:

1. **Check logs**: Look in `logs/trading_system.log` for detailed error messages
2. **Run tests**: Use `python test_system.py` to diagnose issues
3. **Reset database**: Delete `trading_system.db` to start fresh

### 🚨 Important Warnings

#### This is a Demonstration System:
- **SIMULATED TRADING ONLY**: No real money is traded
- **EDUCATIONAL PURPOSE**: For learning about AI trading systems
- **NOT FINANCIAL ADVICE**: Do not use for actual investment decisions

#### Risk Considerations:
- Micro-cap stocks are highly volatile
- AI recommendations are not guaranteed to be profitable
- Always do your own research before real trading
- Never invest more than you can afford to lose

### 🆙 Advanced Usage

#### Running Automated Daily Updates:
You can set up the system to run automatically using cron (Linux/Mac) or Task Scheduler (Windows):

```bash
# Example cron job (runs at 9:45 AM daily)
45 9 * * 1-5 /path/to/your/trading_env/bin/python /path/to/enhanced_trading_system.py
```

#### Custom Development:
The system is modular and extensible:
- Add new data sources in `data_manager.py`
- Modify AI prompts in `perplexity_client.py`
- Add custom indicators in the analysis modules

### 📈 Performance Tracking

The system automatically tracks:
- Portfolio value over time
- Individual position performance
- Trade history with reasoning
- Risk metrics (Sharpe ratio, max drawdown)
- AI recommendation accuracy

All data is stored in the SQLite database and can be exported for further analysis.

### 🎯 Next Steps

Once you have the system running:

1. **Monitor Performance**: Run daily updates and track results
2. **Analyze Patterns**: Review AI reasoning and market conditions
3. **Adjust Parameters**: Fine-tune risk settings based on results
4. **Learn About Markets**: Use this as a tool to understand micro-cap investing

### 📞 Support

This enhanced system includes:
- ✅ Production-ready error handling
- ✅ Comprehensive logging
- ✅ Database persistence
- ✅ Risk management
- ✅ Performance analytics
- ✅ Modular architecture
- ✅ Full documentation

The system has been completely rewritten to address the issues in the original:
- ✅ Fixed stock price accuracy with multiple data validation
- ✅ Replaced OpenAI API with Perplexity API for better financial data
- ✅ Added robust error handling and logging
- ✅ Implemented proper database storage
- ✅ Enhanced risk management features
- ✅ Production-ready configuration management

Happy trading! 🚀📈
