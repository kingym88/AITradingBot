"""
Enhanced Data Management System for AI Trading
Handles multiple data sources, validation, and accuracy checking
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from loguru import logger
import time
import requests
import csv
from pathlib import Path
from config import get_config

config = get_config()

# Static fallback list of vetted micro-cap symbols (expand as needed)
PRODUCTION_MICRO_CAPS = [
    "HOFT", "RGCO", "EBMT", "OVLY", "TSBK", "NWFL", "PINE", "SAMG", "HRZN", "EARN",
    "PFSW", "MRTN", "TOWN", "HWBK", "PFBC", "CCBG", "NECB", "WASH", "PBHC", "HAFC"
]

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

    ETF_HOLDINGS_CACHE = Path("data/etf_holdings_microcap.csv")
    # iShares Micro-Cap ETF holdings CSV URL
    ETF_HOLDINGS_URL = "https://www.ishares.com/us/products/239710/ishares-micro-cap-etf/1467271812596.ajax?fileType=cv&fileName=IWC_holdings.csv"

    def __init__(self):
        self.validator = DataValidator()
        self.cache = {}
        self.cache_expiry = timedelta(minutes=5)

    def get_stock_data(self, symbol: str, validate: bool = True) -> Optional[StockData]:
        """
        Get comprehensive stock data with validation
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

    def get_historical_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """Get historical data for backtesting and analysis"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            return hist
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            return pd.DataFrame()

    def fetch_etf_holdings(self) -> List[str]:
        """
        Fetch and cache micro-cap ETF holdings dynamically from iShares ETF CSV.
        Returns a list of ticker symbols.
        """
        try:
            logger.info(f"Downloading ETF holdings from {self.ETF_HOLDINGS_URL} ...")
            response = requests.get(self.ETF_HOLDINGS_URL, timeout=15)
            response.raise_for_status()

            self.ETF_HOLDINGS_CACHE.parent.mkdir(parents=True, exist_ok=True)
            self.ETF_HOLDINGS_CACHE.write_text(response.text)
            logger.info("ETF holdings downloaded and cached locally.")

            tickers = self.parse_etf_csv(response.text)
            logger.info(f"Parsed {len(tickers)} tickers from ETF holdings.")
            return tickers

        except Exception as e:
            logger.error(f"Error downloading ETF holdings: {e}")
            # Fallback to cached file if available
            if self.ETF_HOLDINGS_CACHE.exists():
                logger.info("Using cached ETF holdings due to download failure.")
                return self.load_cached_etf_holdings()
            else:
                logger.warning("No cached ETF holdings available.")
                return []

    def parse_etf_csv(self, csv_text: str) -> List[str]:
        """Parse tickers from ETF holdings CSV text."""
        tickers = []
        try:
            reader = csv.DictReader(csv_text.splitlines())
            for row in reader:
                # Try multiple column names that iShares might use
                ticker = (row.get("Holding Ticker") or 
                         row.get("Ticker") or 
                         row.get("Symbol") or 
                         row.get("Holdings Ticker"))
                if ticker and ticker.strip() and len(ticker.strip()) <= 5:
                    tickers.append(ticker.strip().upper())
        except Exception as e:
            logger.error(f"Error parsing ETF CSV: {e}")
        return tickers

    def load_cached_etf_holdings(self) -> List[str]:
        """Load ETF holdings from local cache file."""
        try:
            with open(self.ETF_HOLDINGS_CACHE, 'r') as f:
                csv_text = f.read()
            return self.parse_etf_csv(csv_text)
        except Exception as e:
            logger.error(f"Error loading cached ETF holdings: {e}")
            return []

    def screen_micro_caps(self, min_volume: int = None) -> List[str]:
        """
        Screen micro-caps dynamically by ETF holdings.
        Falls back to static list if no holdings found.
        """
        symbols = self.fetch_etf_holdings()
        if not symbols:
            logger.info("Using fallback static micro-cap list")
            symbols = PRODUCTION_MICRO_CAPS

        filtered = []
        for symbol in symbols:
            try:
                stock_data = self.get_stock_data(symbol, validate=False)
                if stock_data and stock_data.is_micro_cap and stock_data.has_sufficient_volume:
                    filtered.append(symbol)
            except:
                continue

        return filtered

    def get_multiple_stocks(self, symbols: List[str]) -> Dict[str, Optional[StockData]]:
        """Get data for multiple stocks efficiently"""
        results = {}
        for symbol in symbols:
            results[symbol] = self.get_stock_data(symbol)
            time.sleep(0.1)  # Rate limiting
        return results
