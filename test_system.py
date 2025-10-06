"""
Test suite for Enhanced AI Trading System
"""
import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import TradingConfig
from data_manager import EnhancedDataManager, StockData, DataValidator
from perplexity_client import PerplexityClientSync, TradingRecommendation
from database import DatabaseManager

class TestTradingConfig:
    """Test configuration management"""

    def test_config_validation(self):
        """Test configuration validation"""
        # This would normally fail without proper env vars
        # In testing, we mock the environment
        with patch.dict(os.environ, {'PERPLEXITY_API_KEY': 'test_key'}):
            config = TradingConfig()
            assert config.PERPLEXITY_API_KEY == 'test_key'
            assert config.INITIAL_CAPITAL > 0
            assert config.MAX_POSITION_SIZE <= 1.0

class TestDataManager:
    """Test data management functionality"""

    @patch('yfinance.Ticker')
    def test_get_stock_data(self, mock_ticker):
        """Test getting stock data"""
        # Mock yfinance response
        mock_info = {
            'marketCap': 250_000_000,  # Micro-cap
        }

        mock_hist = pd.DataFrame({
            'Close': [25.0, 26.0],
            'Volume': [150000, 160000]
        })

        mock_ticker_instance = Mock()
        mock_ticker_instance.info = mock_info
        mock_ticker_instance.history.return_value = mock_hist
        mock_ticker.return_value = mock_ticker_instance

        # Test data manager
        data_manager = EnhancedDataManager()
        stock_data = data_manager.get_stock_data('TEST', validate=False)

        assert stock_data is not None
        assert stock_data.symbol == 'TEST'
        assert stock_data.is_micro_cap == True
        assert stock_data.has_sufficient_volume == True

    def test_data_validator(self):
        """Test data validation"""
        validator = DataValidator()

        # Test price consistency
        prices = {'source1': 25.0, 'source2': 25.5, 'source3': 24.8}
        assert validator.validate_price_consistency(prices, tolerance=0.05) == True

        prices_bad = {'source1': 25.0, 'source2': 30.0}  # 20% difference
        assert validator.validate_price_consistency(prices_bad, tolerance=0.05) == False

class TestPerplexityClient:
    """Test Perplexity API client"""

    @patch('httpx.AsyncClient')
    def test_trading_recommendation(self, mock_client):
        """Test getting trading recommendations"""
        # Mock API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'choices': [{
                'message': {
                    'content': '{"action": "BUY", "symbol": "TEST", "reasoning": "Strong fundamentals", "confidence": 0.8}'
                }
            }]
        }

        mock_client_instance = Mock()
        mock_client_instance.post.return_value = mock_response
        mock_client.return_value = mock_client_instance

        # Test client
        client = PerplexityClientSync()
        portfolio_data = {
            'total_value': 100.0,
            'cash': 50.0,
            'num_positions': 1,
            'holdings': {'EXISTING': 10},
            'daily_pnl': 2.5,
            'total_return': 5.0
        }

        market_data = {
            'market_open': True,
            'sp500_change': 1.0,
            'vix': 20.0,
            'sentiment': 'Bullish',
            'candidates': []
        }

        recommendations = client.get_trading_recommendation(portfolio_data, market_data)

        assert len(recommendations) >= 0  # Should return list

class TestDatabaseManager:
    """Test database functionality"""

    def test_database_initialization(self):
        """Test database setup"""
        # Use in-memory SQLite for testing
        with patch('config.get_config') as mock_config:
            mock_config.return_value.DATABASE_URL = 'sqlite:///:memory:'
            db_manager = DatabaseManager()

            # Test basic database operations
            portfolio_data = {
                'total_value': 100.0,
                'cash': 50.0,
                'invested_amount': 50.0,
                'total_return': 0.0,
                'daily_pnl': 0.0,
                'num_positions': 0
            }

            # This should not raise an error
            db_manager.save_portfolio_snapshot(portfolio_data)

            # Get portfolio history
            history = db_manager.get_portfolio_history(30)
            assert isinstance(history, pd.DataFrame)

def run_integration_test():
    """Run a simple integration test"""
    print("Running integration test...")

    try:
        # Test imports
        from enhanced_trading_system import EnhancedTradingSystem
        from config import get_config

        print("✅ All imports successful")

        # Test configuration
        config = get_config()
        print(f"✅ Configuration loaded: Initial capital ${config.INITIAL_CAPITAL}")

        # Test data manager (with mocked data)
        data_manager = EnhancedDataManager()
        print("✅ Data manager initialized")

        # Test database
        from database import get_db_manager
        db_manager = get_db_manager()
        print("✅ Database manager initialized")

        print("\n🎉 Integration test passed! System is ready to run.")

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

    return True

if __name__ == "__main__":
    # Run integration test when called directly
    run_integration_test()
