"""
Enhanced Data Management System for AI Trading
Handles multiple data sources, validation, and accuracy checking
"""
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import requests
from dataclasses import dataclass
from loguru import logger
import time
from config import get_config

config = get_config()

@dataclass
class StockData:
    """Data class for stock information"""
    symbol: str
    price: float
    volume: int
    market_cap: Optional[int]
    previous_close: float
    change: float
    change_percent: float
    timestamp: datetime

    @property
    def is_micro_cap(self) -> bool:
        """Check if stock qualifies as micro-cap"""
        return self.market_cap and self.market_cap <= config.MAX_MARKET_CAP

    @property
    def has_sufficient_volume(self) -> bool:
        """Check if stock has sufficient trading volume"""
        return self.volume >= config.MIN_VOLUME

class DataValidator:
    """Validates data accuracy across multiple sources"""

    @staticmethod
    def validate_price_consistency(prices: Dict[str, float], tolerance: float = 0.05) -> bool:
        """
        Validate price consistency across multiple sources
        Returns True if prices are within tolerance percentage
        """
        if len(prices) < 2:
            return True

        price_values = list(prices.values())
        avg_price = np.mean(price_values)

        for source, price in prices.items():
            deviation = abs(price - avg_price) / avg_price
            if deviation > tolerance:
                logger.warning(f"Price deviation for {source}: {deviation:.2%}")
                return False

        return True

    @staticmethod
    def validate_market_hours() -> bool:
        """Check if market is open"""
        now = datetime.now()
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)

        # Check if it's a weekday and within market hours
        is_weekday = now.weekday() < 5  # 0-4 are Mon-Fri
        is_market_hours = market_open <= now <= market_close

        return is_weekday and is_market_hours

class EnhancedDataManager:
    """Enhanced data manager with multiple sources and validation"""

    def __init__(self):
        self.validator = DataValidator()
        self.cache = {}
        self.cache_expiry = timedelta(minutes=5)

    def get_stock_data(self, symbol: str, validate: bool = True) -> Optional[StockData]:
        """
        Get comprehensive stock data with validation

        Args:
            symbol: Stock symbol
            validate: Whether to validate across multiple sources

        Returns:
            StockData object or None if data unavailable
        """
        try:
            # Check cache first
            cache_key = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M')}"
            if cache_key in self.cache:
                cache_time, data = self.cache[cache_key]
                if datetime.now() - cache_time < self.cache_expiry:
                    return data

            # Get primary data from Yahoo Finance
            primary_data = self._get_yahoo_data(symbol)
            if not primary_data:
                logger.error(f"Failed to get primary data for {symbol}")
                return None

            # Validate with additional sources if requested
            if validate:
                validation_prices = {'yahoo': primary_data['price']}

                # Add backup sources for validation
                backup_price = self._get_backup_price(symbol)
                if backup_price:
                    validation_prices['backup'] = backup_price

                if not self.validator.validate_price_consistency(validation_prices):
                    logger.warning(f"Price validation failed for {symbol}")
                    # Continue with primary data but log the issue

            # Create StockData object
            stock_data = StockData(
                symbol=symbol,
                price=primary_data['price'],
                volume=primary_data['volume'],
                market_cap=primary_data['market_cap'],
                previous_close=primary_data['previous_close'],
                change=primary_data['change'],
                change_percent=primary_data['change_percent'],
                timestamp=datetime.now()
            )

            # Cache the result
            self.cache[cache_key] = (datetime.now(), stock_data)

            return stock_data

        except Exception as e:
            logger.error(f"Error getting stock data for {symbol}: {e}")
            return None

    def _get_yahoo_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="2d")

            if hist.empty:
                return None

            current_price = hist['Close'].iloc[-1]
            previous_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
            volume = hist['Volume'].iloc[-1]

            change = current_price - previous_close
            change_percent = (change / previous_close) * 100 if previous_close > 0 else 0

            return {
                'price': float(current_price),
                'volume': int(volume),
                'market_cap': info.get('marketCap'),
                'previous_close': float(previous_close),
                'change': float(change),
                'change_percent': float(change_percent)
            }

        except Exception as e:
            logger.error(f"Error getting Yahoo Finance data for {symbol}: {e}")
            return None

    def _get_backup_price(self, symbol: str) -> Optional[float]:
        """Get price from backup source for validation"""
        try:
            # This is a placeholder for backup data source
            # In production, you would integrate with Alpha Vantage, Finnhub, etc.
            # For now, we'll add some noise to simulate a backup source
            primary_price = self.cache.get(f"{symbol}_primary_price")
            if primary_price:
                # Simulate backup source with small random variation
                noise = np.random.normal(0, 0.01)  # 1% standard deviation
                return primary_price * (1 + noise)
            return None

        except Exception as e:
            logger.error(f"Error getting backup price for {symbol}: {e}")
            return None

    def get_multiple_stocks(self, symbols: List[str]) -> Dict[str, Optional[StockData]]:
        """Get data for multiple stocks efficiently"""
        results = {}
        for symbol in symbols:
            results[symbol] = self.get_stock_data(symbol)
            time.sleep(0.1)  # Rate limiting
        return results

    def get_market_data(self, symbols: List[str]) -> pd.DataFrame:
        """Get market data as DataFrame for analysis"""
        data = []
        for symbol in symbols:
            stock_data = self.get_stock_data(symbol)
            if stock_data:
                data.append({
                    'symbol': stock_data.symbol,
                    'price': stock_data.price,
                    'volume': stock_data.volume,
                    'market_cap': stock_data.market_cap,
                    'change_percent': stock_data.change_percent,
                    'timestamp': stock_data.timestamp
                })

        return pd.DataFrame(data)

    def get_historical_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """Get historical data for backtesting and analysis"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            return hist
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            return pd.DataFrame()

    def screen_micro_caps(self, min_volume: int = None) -> List[str]:
        """Screen for micro-cap stocks meeting criteria"""
        # This is a placeholder for micro-cap screening
        # In production, you would implement proper screening logic
        # using financial databases or APIs

        sample_microcaps = [
            'AAON', 'ABCB', 'ABEO', 'ABEQ', 'ABIO', 'ABUS', 'ACAD', 'ACET',
            'ACIU', 'ACRX', 'ADAP', 'ADCT', 'ADMA', 'ADMS', 'ADPT', 'ADRO'
        ]

        # Filter by volume and market cap
        filtered_symbols = []
        for symbol in sample_microcaps:
            stock_data = self.get_stock_data(symbol, validate=False)
            if stock_data and stock_data.is_micro_cap and stock_data.has_sufficient_volume:
                filtered_symbols.append(symbol)

        return filtered_symbols
