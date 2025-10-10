"""
Updated Enhanced AI Trading System with GitHub Stock Symbol Sources
UPDATED: Now uses AMEX, NASDAQ & NYSE JSON files from GitHub instead of iShares ETF data
"""
import sys
import json
import requests
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from loguru import logger

from config import get_config
config = get_config()

# Configure logging
logger.remove()
logger.add(sys.stderr, level=config.LOG_LEVEL, 
           format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}")
logger.add(config.LOG_FILE, rotation="10 MB", retention="1 month", level=config.LOG_LEVEL,
           format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}")

from data_manager import EnhancedDataManager, StockData
from perplexity_client import PerplexityClientSync, TradingRecommendation
from database import get_db_manager, Position, Trade, AIRecommendation
from ml_engine import MLRecommendationEngine
from weekend_analyzer import WeekendAnalyzer
from performance_analyzer import PerformanceAnalyzer

class GitHubStockDataManager:
    """Manages stock symbol data from GitHub JSON sources"""
    
    def __init__(self):
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
        """Fetch stock data from specific exchange JSON file"""
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
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching {exchange} stocks: {e}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing {exchange} JSON: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error fetching {exchange} stocks: {e}")
            return []

    def fetch_all_stocks(self) -> List[Dict]:
        """Fetch stocks from all exchanges (NASDAQ, NYSE, AMEX)"""
        all_stocks = []
        
        for exchange in self.github_urls.keys():
            exchange_stocks = self.fetch_exchange_stocks(exchange)
            # Add exchange info to each stock
            for stock in exchange_stocks:
                stock['exchange'] = exchange
            all_stocks.extend(exchange_stocks)
        
        logger.info(f"Total stocks fetched from all exchanges: {len(all_stocks)}")
        return all_stocks

    def filter_stocks_for_analysis(self, stocks: List[Dict], max_market_cap: int = None) -> List[Dict]:
        """Filter stocks based on trading criteria"""
        try:
            if max_market_cap is None:
                max_market_cap = config.MAX_MARKET_CAP
            
            filtered_stocks = []
            
            for stock in stocks:
                try:
                    # Skip stocks without proper data
                    if not stock.get('symbol') or not stock.get('marketCap'):
                        continue
                    
                    # Parse market cap (remove commas and convert to float)
                    market_cap_str = str(stock.get('marketCap', '0')).replace(',', '')
                    try:
                        market_cap = float(market_cap_str)
                    except (ValueError, TypeError):
                        continue
                    
                    # Skip stocks with zero or invalid market cap
                    if market_cap <= 0:
                        continue
                    
                    # Filter by market cap if specified
                    if max_market_cap and market_cap > max_market_cap:
                        continue
                    
                    # Parse volume
                    try:
                        volume = int(str(stock.get('volume', '0')).replace(',', ''))
                    except (ValueError, TypeError):
                        volume = 0
                    
                    # Filter by minimum volume
                    if volume < config.MIN_VOLUME:
                        continue
                    
                    # Skip certain types of securities
                    name = stock.get('name', '').lower()
                    if any(term in name for term in ['warrant', 'rights', 'unit', 'preferred']):
                        continue
                    
                    # Skip blank check companies and SPACs
                    industry = stock.get('industry', '').lower()
                    if 'blank check' in industry:
                        continue
                    
                    # Only include US stocks by default
                    country = stock.get('country', '')
                    if country and country != 'United States':
                        continue
                    
                    filtered_stocks.append(stock)
                    
                except Exception as e:
                    logger.debug(f"Error processing stock {stock.get('symbol', 'UNKNOWN')}: {e}")
                    continue
            
            logger.info(f"Filtered to {len(filtered_stocks)} qualifying stocks")
            return filtered_stocks
            
        except Exception as e:
            logger.error(f"Error filtering stocks: {e}")
            return []

    def get_stock_symbols(self, limit: int = None) -> List[str]:
        """Get list of stock symbols for analysis"""
        try:
            all_stocks = self.fetch_all_stocks()
            filtered_stocks = self.filter_stocks_for_analysis(all_stocks)
            
            # Sort by market cap (largest first) then by volume
            filtered_stocks.sort(key=lambda x: (
                float(str(x.get('marketCap', '0')).replace(',', '') or 0),
                int(str(x.get('volume', '0')).replace(',', '') or 0)
            ), reverse=True)
            
            symbols = [stock['symbol'] for stock in filtered_stocks]
            
            if limit:
                symbols = symbols[:limit]
            
            logger.info(f"Returning {len(symbols)} stock symbols for analysis")
            return symbols
            
        except Exception as e:
            logger.error(f"Error getting stock symbols: {e}")
            return []

    def get_stocks_by_sector(self, sector: str) -> List[Dict]:
        """Get stocks filtered by sector"""
        try:
            all_stocks = self.fetch_all_stocks()
            filtered_stocks = self.filter_stocks_for_analysis(all_stocks)
            
            sector_stocks = [
                stock for stock in filtered_stocks 
                if stock.get('sector', '').lower() == sector.lower()
            ]
            
            logger.info(f"Found {len(sector_stocks)} stocks in {sector} sector")
            return sector_stocks
            
        except Exception as e:
            logger.error(f"Error getting stocks by sector {sector}: {e}")
            return []

    def get_stocks_by_industry(self, industry: str) -> List[Dict]:
        """Get stocks filtered by industry"""
        try:
            all_stocks = self.fetch_all_stocks()
            filtered_stocks = self.filter_stocks_for_analysis(all_stocks)
            
            industry_stocks = [
                stock for stock in filtered_stocks 
                if industry.lower() in stock.get('industry', '').lower()
            ]
            
            logger.info(f"Found {len(industry_stocks)} stocks in {industry} industry")
            return industry_stocks
            
        except Exception as e:
            logger.error(f"Error getting stocks by industry {industry}: {e}")
            return []

class RiskManager:
    """Enhanced risk management system"""

    def __init__(self):
        self.config = config

    def validate_trade(self, action: str, symbol: str, quantity: int, 
                       price: float, portfolio_value: float, 
                       current_positions: List[Position]) -> Tuple[bool, str]:
        try:
            trade_value = quantity * price

            # Check position size limits
            position_size_percent = trade_value / portfolio_value if portfolio_value > 0 else 0
            if position_size_percent > self.config.MAX_POSITION_SIZE:
                return False, f"Position size {position_size_percent:.1%} exceeds maximum {self.config.MAX_POSITION_SIZE:.1%}"

            # Check maximum positions limit
            if action == 'BUY' and len(current_positions) >= self.config.MAX_POSITIONS:
                return False, f"Maximum positions limit ({self.config.MAX_POSITIONS}) reached"

            return True, "Trade validation passed"

        except Exception as e:
            logger.error(f"Error validating trade: {e}")
            return False, f"Validation error: {e}"

class PortfolioManager:
    """Enhanced portfolio management system with proper trade tracking"""

    def __init__(self):
        self.db_manager = get_db_manager()
        self.data_manager = EnhancedDataManager()
        self.risk_manager = RiskManager()
        self.initial_capital = config.INITIAL_CAPITAL

    def calculate_portfolio_from_trades(self) -> Dict[str, Any]:
        """Calculate current portfolio value from all historical trades"""
        try:
            # Get all trades to calculate current positions
            all_trades = self.get_all_trades()
            
            # Calculate positions from trades
            positions = self._calculate_positions_from_trades(all_trades)
            
            # Update positions with current market prices
            self._update_positions_with_current_prices(positions)
            
            # Calculate cash remaining
            total_trade_amount = sum(
                -trade['total_amount'] if trade['action'] == 'BUY' 
                else trade['total_amount'] 
                for trade in all_trades
            )
            cash = self.initial_capital + total_trade_amount
            
            # Calculate total portfolio value
            total_market_value = sum(pos.market_value for pos in positions.values())
            total_value = total_market_value + cash
            total_return = ((total_value - self.initial_capital) / self.initial_capital) * 100
            
            # Calculate daily P&L
            daily_pnl = sum(p.unrealized_pnl_percent for p in positions.values()) / len(positions) if positions else 0.0
            
            portfolio_summary = {
                'total_value': total_value,
                'cash': cash,
                'invested_amount': total_market_value,
                'total_return': total_return,
                'daily_pnl': daily_pnl,
                'num_positions': len(positions),
                'holdings': {symbol: pos.quantity for symbol, pos in positions.items()}
            }
            
            # Update positions in database
            self._update_positions_in_db(positions)
            
            # Save portfolio snapshot
            self.db_manager.save_portfolio_snapshot(portfolio_summary)
            
            return portfolio_summary
            
        except Exception as e:
            logger.error(f"Error calculating portfolio from trades: {e}")
            return {
                'total_value': self.initial_capital,
                'cash': self.initial_capital,
                'invested_amount': 0.0,
                'total_return': 0.0,
                'daily_pnl': 0.0,
                'num_positions': 0,
                'holdings': {}
            }

    def _calculate_positions_from_trades(self, trades: List[Dict]) -> Dict[str, Position]:
        """Calculate current positions from trade history"""
        positions = {}
        
        for trade in trades:
            symbol = trade['symbol']
            action = trade['action']
            quantity = trade['quantity']
            price = trade['price']
            
            if symbol not in positions:
                # Create new position
                positions[symbol] = Position(
                    symbol=symbol,
                    quantity=0,
                    entry_price=0.0,
                    current_price=price,
                    market_value=0.0,
                    unrealized_pnl=0.0,
                    unrealized_pnl_percent=0.0,
                    entry_date=trade['timestamp'],
                    last_updated=datetime.now(),
                    is_active=True
                )
            
            pos = positions[symbol]
            
            if action == 'BUY':
                # Add to position with weighted average price
                total_cost = (pos.quantity * pos.entry_price) + (quantity * price)
                pos.quantity += quantity
                pos.entry_price = total_cost / pos.quantity if pos.quantity > 0 else price
            elif action == 'SELL':
                # Reduce position
                pos.quantity -= quantity
                if pos.quantity <= 0:
                    # Position closed
                    pos.quantity = 0
                    pos.is_active = False
        
        # Filter out closed positions
        return {symbol: pos for symbol, pos in positions.items() if pos.quantity > 0}

    def _update_positions_with_current_prices(self, positions: Dict[str, Position]):
        """Update positions with current market prices"""
        for symbol, pos in positions.items():
            try:
                stock_data = self.data_manager.get_stock_data(symbol, validate=False)
                if stock_data:
                    pos.current_price = stock_data.price
                    pos.market_value = pos.quantity * pos.current_price
                    pos.unrealized_pnl = pos.market_value - (pos.quantity * pos.entry_price)
                    pos.unrealized_pnl_percent = (pos.unrealized_pnl / (pos.quantity * pos.entry_price)) * 100 if pos.entry_price > 0 else 0
                    pos.last_updated = datetime.now()
            except Exception as e:
                logger.error(f"Error updating price for {symbol}: {e}")
                pos.market_value = pos.quantity * pos.entry_price
                pos.unrealized_pnl = 0.0
                pos.unrealized_pnl_percent = 0.0

    def _update_positions_in_db(self, positions: Dict[str, Position]):
        """Update or create positions in database"""
        try:
            with self.db_manager.get_session() as session:
                # Clear existing positions
                session.query(Position).update({'is_active': False})
                
                # Add current positions
                for pos in positions.values():
                    session.add(pos)
                
                session.commit()
        except Exception as e:
            logger.error(f"Error updating positions in database: {e}")

    def get_all_trades(self) -> List[Dict]:
        """Get all trades from database"""
        try:
            with self.db_manager.get_session() as session:
                trades = session.query(Trade).order_by(Trade.timestamp.asc()).all()
                return [
                    {
                        'symbol': t.symbol,
                        'action': t.action,
                        'quantity': t.quantity,
                        'price': t.price,
                        'total_amount': t.total_amount,
                        'timestamp': t.timestamp,
                        'fees': t.fees or 0.0
                    }
                    for t in trades
                ]
        except Exception as e:
            logger.error(f"Error getting all trades: {e}")
            return []

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get current portfolio summary calculated from actual trades"""
        return self.calculate_portfolio_from_trades()

    def execute_trade(self, symbol: str, action: str, quantity: int, price: float, 
                     reasoning: str = "", confidence: float = None) -> Tuple[bool, str]:
        """Execute a trade and update portfolio accordingly"""
        try:
            # Get current portfolio value
            current_portfolio = self.get_portfolio_summary()
            portfolio_value_before = current_portfolio['total_value']
            
            # Validate trade
            current_positions = self.db_manager.get_active_positions()
            is_valid, validation_msg = self.risk_manager.validate_trade(
                action, symbol, quantity, price, portfolio_value_before, current_positions
            )
            
            if not is_valid:
                return False, validation_msg
            
            # Calculate trade amount
            total_amount = quantity * price
            
            # Check if we have enough cash for BUY orders
            if action == 'BUY' and total_amount > current_portfolio['cash']:
                return False, f"Insufficient cash. Need ${total_amount:.2f}, have ${current_portfolio['cash']:.2f}"
            
            # Log the trade
            trade_data = {
                'symbol': symbol,
                'action': action,
                'quantity': quantity,
                'price': price,
                'total_amount': total_amount,
                'fees': 0.0,
                'reasoning': reasoning,
                'confidence': confidence,
                'timestamp': datetime.now(),
                'portfolio_value_before': portfolio_value_before,
                'portfolio_value_after': None  # Will be updated after portfolio recalculation
            }
            
            self.db_manager.save_trade(trade_data)
            
            # Recalculate portfolio
            updated_portfolio = self.calculate_portfolio_from_trades()
            
            # Update the trade record with new portfolio value
            self._update_trade_portfolio_value_after(trade_data['timestamp'], updated_portfolio['total_value'])
            
            logger.info(f"Trade executed: {action} {quantity} {symbol} @ ${price:.2f}")
            logger.info(f"Portfolio value: ${portfolio_value_before:.2f} -> ${updated_portfolio['total_value']:.2f}")
            
            return True, f"Trade executed successfully. Portfolio value: ${updated_portfolio['total_value']:.2f}"
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return False, f"Trade execution failed: {e}"

    def _update_trade_portfolio_value_after(self, timestamp: datetime, portfolio_value_after: float):
        """Update the portfolio_value_after field for a specific trade"""
        try:
            with self.db_manager.get_session() as session:
                session.query(Trade).filter_by(timestamp=timestamp).update({
                    'portfolio_value_after': portfolio_value_after
                })
                session.commit()
        except Exception as e:
            logger.error(f"Error updating trade portfolio value: {e}")

class EnhancedMLTradingSystem:
    """Main enhanced ML trading system with GitHub stock data sources"""

    def __init__(self):
        self.config = config
        self.portfolio_manager = PortfolioManager()
        self.data_manager = EnhancedDataManager()
        self.github_stock_manager = GitHubStockDataManager()  # NEW: GitHub stock data manager
        self.ai_client = PerplexityClientSync()
        self.ml_engine = MLRecommendationEngine()
        self.risk_manager = RiskManager()
        self.db_manager = get_db_manager()
        self.weekend_analyzer = WeekendAnalyzer()

        logger.info("Enhanced ML Trading System with GitHub Stock Sources initialized")

    def run_daily_update(self) -> None:
        try:
            logger.info("🚀 Starting daily update with GitHub stock analysis...")

            # Get updated portfolio data from actual trades
            portfolio_data = self.portfolio_manager.get_portfolio_summary()
            
            # NEW: Get market data from GitHub sources instead of iShares
            market_data = self._get_market_data_from_github()

            logger.info("🧠 Training ML models from recent trades...")
            self.ml_engine.learn_from_trades()

            logger.info("📊 Generating profit-optimized ML recommendations...")
            ml_recommendations = self.ml_engine.generate_recommendations()

            logger.info("🔍 Getting AI analysis...")
            ai_recommendations = self.ai_client.get_trading_recommendation(
                portfolio_data, market_data
            )

            all_recommendations = self._combine_recommendations(ml_recommendations, ai_recommendations)
            self._display_recommendations(all_recommendations, portfolio_data)

            logger.info(f"📊 Portfolio Value: ${portfolio_data['total_value']:.2f}")
            logger.info(f"📈 Total Return: {portfolio_data['total_return']:.2f}%")
            logger.info(f"📋 Positions: {portfolio_data['num_positions']}")

        except Exception as e:
            logger.error(f"Error in daily update: {e}")

    def _get_market_data_from_github(self) -> Dict[str, Any]:
        """NEW: Get market data from GitHub stock sources instead of iShares"""
        try:
            logger.info("📡 Fetching fresh stock data from GitHub sources...")
            
            # Get stock symbols from GitHub (fresh data each time)
            stock_symbols = self.github_stock_manager.get_stock_symbols(limit=50)
            
            if not stock_symbols:
                logger.warning("No stock symbols retrieved from GitHub sources")
                return {'market_open': False, 'sentiment': 'Neutral', 'candidates': []}
            
            market_data = {
                'market_open': self.data_manager.validator.validate_market_hours(),
                'sentiment': 'Neutral',
                'candidates': [],
                'total_symbols_available': len(stock_symbols)
            }

            # Get current price data for top symbols using yfinance
            logger.info(f"🔍 Getting current prices for top {min(10, len(stock_symbols))} symbols...")
            for symbol in stock_symbols[:10]:
                try:
                    stock_data = self.data_manager.get_stock_data(symbol, validate=False)
                    if stock_data:
                        market_data['candidates'].append({
                            'symbol': stock_data.symbol,
                            'price': stock_data.price,
                            'volume': stock_data.volume,
                            'change_percent': stock_data.change_percent
                        })
                except Exception as e:
                    logger.debug(f"Error getting price data for {symbol}: {e}")
                    continue

            logger.info(f"✅ Retrieved price data for {len(market_data['candidates'])} stocks")
            return market_data

        except Exception as e:
            logger.error(f"Error getting market data from GitHub: {e}")
            return {'market_open': False, 'sentiment': 'Neutral', 'candidates': []}

    def run_weekend_analysis(self) -> None:
        try:
            logger.info("🔍 Starting weekend deep analysis...")

            analysis_results = self.weekend_analyzer.run_weekend_analysis()

            if 'error' not in analysis_results:
                self._display_weekend_analysis(analysis_results)
            else:
                logger.error(f"Weekend analysis failed: {analysis_results['error']}")

        except Exception as e:
            logger.error(f"Error in weekend analysis: {e}")

    def _combine_recommendations(self, ml_recs: List[Dict], ai_recs: List[TradingRecommendation]) -> List[Dict]:
        try:
            combined_recs = []

            for rec in ml_recs:
                rec['source'] = 'ML'
                combined_recs.append(rec)

            for rec in ai_recs:
                ai_rec = {
                    'symbol': rec.symbol,
                    'action': rec.action,
                    'confidence': rec.confidence,
                    'reasoning': rec.reasoning,
                    'price_target': rec.price_target,
                    'expected_return': None,
                    'risk_score': 0.5,
                    'source': 'AI'
                }
                combined_recs.append(ai_rec)

            combined_recs.sort(key=lambda x: x.get('confidence', 0), reverse=True)

            seen_symbols = set()
            unique_recs = []
            for rec in combined_recs:
                if rec['symbol'] not in seen_symbols:
                    seen_symbols.add(rec['symbol'])
                    unique_recs.append(rec)

            return unique_recs[:config.MAX_DAILY_RECOMMENDATIONS]

        except Exception as e:
            logger.error(f"Error combining recommendations: {e}")
            return []

    def _display_recommendations(self, recommendations: List[Dict], portfolio_data: Dict) -> None:
        try:
            print("\n" + "="*80)
            print("🚀 ENHANCED ML AI TRADING SYSTEM - GITHUB STOCK ANALYSIS")
            print("="*80)

            if not recommendations:
                print("❌ No recommendations generated today.")
                print("This could be due to:")
                print("  • GitHub stock data download issues")
                print("  • No qualifying stocks found")
                print("  • Market data unavailable")
                print("\n💡 Try running the system again or check your internet connection.")
                return

            print(f"📊 Portfolio Value: ${portfolio_data['total_value']:.2f} | Return: {portfolio_data['total_return']:.2f}%")
            print(f"💰 Cash Available: ${portfolio_data['cash']:.2f} | Positions: {portfolio_data['num_positions']}")
            
            if portfolio_data['holdings']:
                print("📋 Current Holdings:")
                for symbol, quantity in portfolio_data['holdings'].items():
                    print(f"   {symbol}: {quantity} shares")
            
            print()
            print("🎯 RECOMMENDATIONS (Based on NASDAQ, NYSE & AMEX Analysis):")
            print("-" * 80)

            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. {rec['symbol']} - {rec['action']}")
                print(f"   Source: {rec['source']} | Confidence: {rec['confidence']:.1%}")

                if rec.get('price_target'):
                    print(f"   Price Target: ${rec['price_target']:.2f}")

                if rec.get('expected_return'):
                    print(f"   Expected Return: {rec['expected_return']:.1%}")

                print(f"   Reasoning: {rec['reasoning'][:100]}...")

                try:
                    stock_data = self.data_manager.get_stock_data(rec['symbol'], validate=False)
                    if stock_data:
                        print(f"   Current Price: ${stock_data.price:.2f} | Volume: {stock_data.volume:,}")
                except:
                    pass

                print()

            for rec in recommendations:
                self._save_recommendation(rec)

            print("="*80)
            print("💡 Stock recommendations now based on:")
            print("  📊 Fresh data from NASDAQ, NYSE & AMEX exchanges")
            print("  🔄 Updated daily from GitHub sources")
            print("  📈 Technical analysis and volume patterns")
            print("  💰 Market cap and liquidity filtering")
            print("\n📝 Use Trade Logger to record your actual trades for ML learning!")
            print("📅 Run Weekend Analysis for comprehensive portfolio review.")

        except Exception as e:
            logger.error(f"Error displaying recommendations: {e}")

    def _save_recommendation(self, recommendation: Dict) -> None:
        try:
            rec_data = {
                'symbol': recommendation['symbol'],
                'action': recommendation['action'],
                'reasoning': recommendation['reasoning'],
                'confidence': recommendation['confidence'],
                'price_target': recommendation.get('price_target'),
                'stop_loss': None,
                'position_size': None,
                'time_horizon': None,
                'risk_level': recommendation.get('risk_score')
            }
            self.db_manager.save_ai_recommendation(rec_data)

        except Exception as e:
            logger.error(f"Error saving recommendation: {e}")

    def _display_weekend_analysis(self, analysis: Dict) -> None:
        try:
            print("\n" + "="*60)
            print("📊 WEEKEND DEEP ANALYSIS RESULTS")
            print("="*60)

            if 'portfolio_review' in analysis:
                portfolio = analysis['portfolio_review']
                if 'performance_summary' in portfolio:
                    perf = portfolio['performance_summary']
                    print(f"\n📈 PORTFOLIO PERFORMANCE:")
                    print(f"   Total Value: ${perf.get('total_market_value', 0):.2f}")
                    print(f"   Unrealized P&L: {perf.get('total_unrealized_pnl_percent', 0):.2f}%")

            print("\n📄 Detailed analysis saved to reports directory")

        except Exception as e:
            logger.error(f"Error displaying weekend analysis: {e}")

    def show_github_stock_stats(self) -> None:
        """NEW: Show statistics about GitHub stock data sources"""
        try:
            print("\n" + "="*60)
            print("📊 GITHUB STOCK DATA SOURCES STATUS")
            print("="*60)
            
            for exchange in ['NASDAQ', 'NYSE', 'AMEX']:
                try:
                    stocks = self.github_stock_manager.fetch_exchange_stocks(exchange)
                    print(f"{exchange}: {len(stocks)} stocks available")
                    
                    if stocks:
                        # Show some stats
                        sectors = {}
                        countries = {}
                        for stock in stocks[:100]:  # Sample first 100
                            sector = stock.get('sector', 'Unknown')
                            country = stock.get('country', 'Unknown')
                            sectors[sector] = sectors.get(sector, 0) + 1
                            countries[country] = countries.get(country, 0) + 1
                        
                        print(f"  Top sectors: {', '.join(list(sectors.keys())[:3])}")
                        print(f"  Countries: {', '.join(list(countries.keys())[:3])}")
                except Exception as e:
                    print(f"{exchange}: Error fetching data - {e}")
            
            # Test filtering
            try:
                filtered_symbols = self.github_stock_manager.get_stock_symbols(limit=10)
                print(f"\nFiltered symbols for analysis: {len(filtered_symbols)}")
                print(f"Sample symbols: {', '.join(filtered_symbols[:5])}")
            except Exception as e:
                print(f"Error testing filtering: {e}")
            
        except Exception as e:
            logger.error(f"Error showing GitHub stock stats: {e}")

    def run_interactive_mode(self) -> None:
        logger.info("🚀 Starting Enhanced ML Trading System with GitHub Sources...")

        while True:
            try:
                print("\n" + "="*70)
                print("🚀 ENHANCED ML AI TRADING SYSTEM - GITHUB STOCK SOURCES")
                print("="*70)
                print("1. 📈 Daily Update & Stock Recommendations")
                print("2. 📊 Portfolio Summary")
                print("3. 📝 Trade Logger")
                print("4. 🔍 Weekend Deep Analysis")
                print("5. 📉 Performance Report")
                print("6. 🤖 ML Model Status")
                print("7. 📡 GitHub Stock Data Status")
                print("8. ⚙️ System Settings")
                print("9. 🚪 Exit")

                choice = input("\nSelect option (1-9): ").strip()

                if choice == '1':
                    self.run_daily_update()

                elif choice == '2':
                    portfolio = self.portfolio_manager.get_portfolio_summary()
                    self._display_portfolio_summary(portfolio)

                elif choice == '3':
                    from trade_logger import run_trade_logging_interface
                    run_trade_logging_interface()

                elif choice == '4':
                    if config.WEEKEND_ANALYSIS_ENABLED:
                        self.run_weekend_analysis()
                    else:
                        print("Weekend analysis is disabled in configuration")

                elif choice == '5':
                    analyzer = PerformanceAnalyzer()
                    analyzer.print_summary_report(30)

                elif choice == '6':
                    self.show_ml_model_status()

                elif choice == '7':
                    self.show_github_stock_stats()

                elif choice == '8':
                    self._show_system_settings()

                elif choice == '9':
                    logger.info("👋 Exiting Enhanced ML Trading System...")
                    break

                else:
                    print("❌ Invalid option. Please select 1-9.")

            except KeyboardInterrupt:
                logger.info("👋 Interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error in interactive mode: {e}")
                print(f"❌ Error: {e}")

    def _display_portfolio_summary(self, portfolio: Dict) -> None:
        try:
            print("\n" + "="*50)
            print("📊 PORTFOLIO SUMMARY")
            print("="*50)
            print(f"💰 Total Value: ${portfolio['total_value']:.2f}")
            print(f"💵 Cash Available: ${portfolio['cash']:.2f}")
            print(f"📈 Invested Amount: ${portfolio['invested_amount']:.2f}")
            print(f"🎯 Total Return: {portfolio['total_return']:.2f}%")
            print(f"📋 Active Positions: {portfolio['num_positions']}")
            
            if portfolio['holdings']:
                print("\n📋 CURRENT HOLDINGS:")
                for symbol, quantity in portfolio['holdings'].items():
                    try:
                        stock_data = self.data_manager.get_stock_data(symbol, validate=False)
                        if stock_data:
                            value = quantity * stock_data.price
                            print(f"   {symbol}: {quantity} shares @ ${stock_data.price:.2f} = ${value:.2f}")
                        else:
                            print(f"   {symbol}: {quantity} shares (price unavailable)")
                    except:
                        print(f"   {symbol}: {quantity} shares (price unavailable)")

        except Exception as e:
            logger.error(f"Error displaying portfolio summary: {e}")

    def show_ml_model_status(self) -> None:
        try:
            print("\n" + "="*50)
            print("🤖 ML MODEL STATUS")
            print("="*50)

            model_info = self.ml_engine.get_model_info()

            print(f"Profitability Analyzer: {'✅ Active' if model_info.get('profitability_analyzer_available') else '❌ Not available'}")
            print(f"Technical Analysis: ✅ Active")
            print(f"Volume Analysis: ✅ Active") 
            print(f"Value Assessment: ✅ Active")
            print(f"GitHub Stock Sources: ✅ Active")

            trade_history = self.db_manager.get_trade_history(days=90)
            print(f"\nTraining Data: {len(trade_history)} trades (min {config.MIN_TRAINING_SAMPLES} for ML)")

            if len(trade_history) < config.MIN_TRAINING_SAMPLES:
                print("⚠️ Need more trade data for advanced ML training")
                print("💡 Use Trade Logger to record your actual trades")

        except Exception as e:
            logger.error(f"Error showing ML model status: {e}")

    def _show_system_settings(self) -> None:
        try:
            print("\n" + "="*50)
            print("⚙️ SYSTEM SETTINGS")
            print("="*50)
            print(f"Initial Capital: ${config.INITIAL_CAPITAL}")
            print(f"Max Position Size: {config.MAX_POSITION_SIZE:.1%}")
            print(f"Stop Loss: {config.STOP_LOSS_PERCENTAGE:.1%}")
            print(f"Take Profit: {config.TAKE_PROFIT_PERCENTAGE:.1%}")
            print(f"Max Positions: {config.MAX_POSITIONS}")
            print(f"Daily Recommendations: {config.MAX_DAILY_RECOMMENDATIONS}")
            print(f"Weekend Analysis: {'Enabled' if config.WEEKEND_ANALYSIS_ENABLED else 'Disabled'}")
            print(f"Stock Sources: GitHub (NASDAQ, NYSE, AMEX)")
            print(f"Price Data: yfinance API")

        except Exception as e:
            logger.error(f"Error showing system settings: {e}")

def main():
    try:
        print("🚀 Initializing Enhanced ML Trading System with GitHub Stock Sources...")

        trading_system = EnhancedMLTradingSystem()

        if (datetime.now().weekday() in [5, 6] and config.WEEKEND_ANALYSIS_ENABLED):
            print("📅 Weekend detected - running deep analysis...")
            trading_system.run_weekend_analysis()

        trading_system.run_interactive_mode()

    except Exception as e:
        logger.error(f"Critical error in main: {e}")
        print(f"❌ Critical error: {e}")
        print("Please check your configuration and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()
