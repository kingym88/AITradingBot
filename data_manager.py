"""
Updated Data Manager for GitHub Stock Sources Integration
UPDATED: Uses GitHub JSON files for universe, still uses yfinance for price data
"""
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import time
from dataclasses import dataclass
from loguru import logger

from config import get_config
config = get_config()

@dataclass
class StockData:
    symbol: str
    price: float
    volume: int
    market_cap: Optional[float] = None
    change_percent: Optional[float] = None
    pe_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    sector: Optional[str] = None
    industry: Optional[str] = None

class MarketHoursValidator:
    """Validates market hours and trading sessions"""

    def __init__(self):
        self.market_open_time = "09:30"
        self.market_close_time = "16:00"

    def validate_market_hours(self) -> bool:
        """Check if market is currently open"""
        try:
            now = datetime.now()
            if now.weekday() > 4:  # Saturday, Sunday
                return False
            current_time = now.strftime("%H:%M")
            return self.market_open_time <= current_time <= self.market_close_time
        except Exception as e:
            logger.error(f"Error validating market hours: {e}")
            return False

class GitHubStockDataManager:
    """Manages stock symbol data from GitHub JSON sources"""

    def __init__(self):
        import requests
        self.github_urls = {
            'NASDAQ': 'https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/nasdaq/nasdaq_full_tickers.json',
            'NYSE': 'https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/nyse/nyse_full_tickers.json',
            'AMEX': 'https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/amex/amex_full_tickers.json'
        }
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'AI-Trading-Bot/1.0'
        })

    def fetch_exchange_stocks(self, exchange: str) -> List[Dict]:
        try:
            if exchange not in self.github_urls:
                raise ValueError(f"Unknown exchange: {exchange}")
            url = self.github_urls[exchange]
            logger.info(f"Fetching {exchange} stocks from GitHub: {url}")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            stocks = response.json()
            logger.info(f"Successfully fetched {len(stocks)} {exchange} stocks")
            return stocks
        except Exception as e:
            logger.error(f"Error fetching {exchange} stocks: {e}")
            return []

    def fetch_all_stocks(self) -> List[Dict]:
        all_stocks = []
        for exchange in self.github_urls.keys():
            exchange_stocks = self.fetch_exchange_stocks(exchange)
            for stock in exchange_stocks:
                stock['exchange'] = exchange
            all_stocks.extend(exchange_stocks)
        logger.info(f"Total stocks fetched from all exchanges: {len(all_stocks)}")
        return all_stocks

    def filter_stocks_for_analysis(self, stocks: List[Dict], max_market_cap: int = None) -> List[Dict]:
        """Filter by MC/volume/US/securities type..."""
        if max_market_cap is None:
            max_market_cap = config.MAX_MARKET_CAP
        filtered_stocks = []
        for stock in stocks:
            try:
                if not stock.get('symbol') or not stock.get('marketCap'):
                    continue
                # Parse market cap
                market_cap_str = str(stock.get('marketCap', '0')).replace(',', '')
                try:
                    market_cap = float(market_cap_str)
                except (ValueError, TypeError):
                    continue
                if market_cap <= 0 or (max_market_cap and market_cap > max_market_cap):
                    continue
                # Parse volume
                try:
                    volume = int(str(stock.get('volume', '0')).replace(',', ''))
                except (ValueError, TypeError):
                    volume = 0
                if volume < config.MIN_VOLUME:
                    continue
                # Exclude certain securities
                name = stock.get('name', '').lower()
                if any(term in name for term in ['warrant', 'rights', 'unit', 'preferred']):
                    continue
                industry = stock.get('industry', '').lower()
                if 'blank check' in industry: continue
                country = stock.get('country', '')
                if country and country != 'United States': continue

                filtered_stocks.append(stock)

            except Exception as e:
                logger.debug(f"Error processing stock {stock.get('symbol', 'UNKNOWN')}: {e}")
                continue
        logger.info(f"Filtered to {len(filtered_stocks)} qualifying stocks")
        return filtered_stocks

    def get_stock_symbols(self, limit: int = None) -> List[str]:
        try:
            all_stocks = self.fetch_all_stocks()
            filtered_stocks = self.filter_stocks_for_analysis(all_stocks)
            # Sort by market cap, then volume
            filtered_stocks.sort(key=lambda x: (
                float(str(x.get('marketCap', '0')).replace(',', '') or 0),
                int(str(x.get('volume', '0')).replace(',', '') or 0)
            ), reverse=True)
            symbols = [stock['symbol'] for stock in filtered_stocks]
            if limit: symbols = symbols[:limit]
            logger.info(f"Returning {len(symbols)} stock symbols for analysis")
            return symbols
        except Exception as e:
            logger.error(f"Error getting stock symbols: {e}")
            return []

    def get_stocks_by_sector(self, sector: str) -> List[Dict]:
        try:
            all_stocks = self.fetch_all_stocks()
            filtered_stocks = self.filter_stocks_for_analysis(all_stocks)
            return [s for s in filtered_stocks if s.get('sector', '').lower() == sector.lower()]
        except Exception as e:
            logger.error(f"Error getting stocks by sector {sector}: {e}")
            return []

    def get_stocks_by_industry(self, industry: str) -> List[Dict]:
        try:
            all_stocks = self.fetch_all_stocks()
            filtered_stocks = self.filter_stocks_for_analysis(all_stocks)
            return [s for s in filtered_stocks if industry.lower() in s.get('industry', '').lower()]
        except Exception as e:
            logger.error(f"Error getting stocks by industry {industry}: {e}")
            return []

class EnhancedDataManager:
    """
    Enhanced data manager with GitHub universe and yfinance prices
    """
    def __init__(self):
        self.config = config
        self.validator = MarketHoursValidator()
        self.github_manager = GitHubStockDataManager()
        self._session_cache = {}
        self._last_cache_clear = datetime.now()

    def get_stock_data(self, symbol: str, validate: bool = True) -> Optional[StockData]:
        try:
            if validate and not self.validator.validate_market_hours():
                logger.debug(f"Market closed - getting last available data for {symbol}")
            if (datetime.now() - self._last_cache_clear).seconds > 3600:
                self._session_cache.clear()
                self._last_cache_clear = datetime.now()
            ticker = yf.Ticker(symbol)
            info = ticker.info
            if not info: return None
            hist = ticker.history(period="2d")
            if hist.empty: return None
            latest = hist.iloc[-1]
            previous_close = hist.iloc[-2]['Close'] if len(hist) > 1 else latest['Close']
            change_percent = ((latest['Close'] - previous_close) / previous_close * 100) if previous_close > 0 else 0
            return StockData(
                symbol=symbol,
                price=float(latest['Close']),
                volume=int(latest['Volume']),
                market_cap=info.get('marketCap'),
                change_percent=change_percent,
                pe_ratio=info.get('forwardPE') or info.get('trailingPE'),
                dividend_yield=info.get('dividendYield'),
                sector=info.get('sector'),
                industry=info.get('industry')
            )
        except Exception as e:
            logger.error(f"Error getting stock data for {symbol}: {e}")
            return None

    def get_multiple_stock_data(self, symbols: List[str]) -> Dict[str, StockData]:
        results = {}
        batch_size = 10
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            for symbol in batch:
                try:
                    stock_data = self.get_stock_data(symbol, validate=False)
                    if stock_data:
                        results[symbol] = stock_data
                    time.sleep(0.1)
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
                    continue
        logger.info(f"Retrieved data for {len(results)} out of {len(symbols)} symbols")
        return results

    def screen_micro_caps(self, max_market_cap: int = None) -> List[str]:
        try:
            if max_market_cap is None:
                max_market_cap = config.MAX_MARKET_CAP
            logger.info("📡 Screening stocks from GitHub sources (NASDAQ, NYSE, AMEX)...")
            symbols = self.github_manager.get_stock_symbols(limit=100)
            if not symbols:
                logger.warning("No symbols retrieved from GitHub sources")
                return []
            logger.info(f"✅ Found {len(symbols)} qualifying symbols from GitHub sources")
            return symbols
        except Exception as e:
            logger.error(f"Error screening micro caps from GitHub: {e}")
            return []

    def get_sector_stocks(self, sector: str, limit: int = 20) -> List[str]:
        try:
            sector_stocks = self.github_manager.get_stocks_by_sector(sector)
            symbols = [stock['symbol'] for stock in sector_stocks[:limit]]
            logger.info(f"Found {len(symbols)} stocks in {sector} sector")
            return symbols
        except Exception as e:
            logger.error(f"Error getting sector stocks: {e}")
            return []

    def get_industry_stocks(self, industry: str, limit: int = 20) -> List[str]:
        try:
            industry_stocks = self.github_manager.get_stocks_by_industry(industry)
            symbols = [stock['symbol'] for stock in industry_stocks[:limit]]
            logger.info(f"Found {len(symbols)} stocks in {industry} industry")
            return symbols
        except Exception as e:
            logger.error(f"Error getting industry stocks: {e}")
            return []

    def validate_symbol_exists(self, symbol: str) -> bool:
        try:
            stock_data = self.get_stock_data(symbol, validate=False)
            return stock_data is not None
        except Exception as e:
            logger.error(f"Error validating symbol {symbol}: {e}")
            return False
        
    def get_historical_data(self, symbol: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
        """
        Return historical price data for a symbol using yfinance.
        Default: 6 months daily.
        """
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period, interval=interval)
            if hist is not None and not hist.empty:
                return hist
            else:
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()    

def get_enhanced_data_manager() -> EnhancedDataManager:
    return EnhancedDataManager()

def validate_stock_symbol(symbol: str) -> bool:
    manager = EnhancedDataManager()
    return manager.validate_symbol_exists(symbol)

def get_current_price(symbol: str) -> Optional[float]:
    manager = EnhancedDataManager()
    stock_data = manager.get_stock_data(symbol, validate=False)
    return stock_data.price if stock_data else None
