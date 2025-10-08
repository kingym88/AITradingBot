# Updated Enhanced AI Trading System with Portfolio Value Tracking

"""
Enhanced AI Trading System with Machine Learning and Portfolio Value Tracking
Main application integrating ML recommendations, trade logging, and weekend analysis
UPDATED: Now properly tracks portfolio values after trades and calculates from actual trade history
"""
import sys
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

# Updated TradeLogger class
class TradeLogger:
    def __init__(self):
        self.db_manager = get_db_manager()
        self.data_manager = EnhancedDataManager()
        self.portfolio_manager = PortfolioManager()

    def log_trade_interactive(self) -> bool:
        """Interactive trade logging with portfolio update"""
        try:
            print("\n" + "="*50)
            print("📝 TRADE LOGGING SYSTEM")
            print("="*50)

            symbol = input("Stock Symbol (e.g., AAPL): ").strip().upper()
            if not symbol:
                print("❌ Symbol is required")
                return False

            stock_data = self.data_manager.get_stock_data(symbol, validate=False)
            if stock_data:
                print(f"✅ Current price for {symbol}: ${stock_data.price:.2f}")

            while True:
                action = input("Action (BUY/SELL): ").strip().upper()
                if action in ['BUY', 'SELL']:
                    break
                print("❌ Please enter BUY or SELL")

            while True:
                try:
                    quantity = int(input("Quantity (number of shares): ").strip())
                    if quantity > 0:
                        break
                    else:
                        print("❌ Quantity must be positive")
                except ValueError:
                    print("❌ Please enter a valid number")

            while True:
                try:
                    price_input = input(f"Price per share (current: ${stock_data.price:.2f} if available): ").strip()
                    if not price_input and stock_data:
                        price = stock_data.price
                        print(f"Using current market price: ${price:.2f}")
                        break
                    else:
                        price = float(price_input)
                        if price > 0:
                            break
                        else:
                            print("❌ Price must be positive")
                except ValueError:
                    print("❌ Please enter a valid price")

            reasoning = input("Reasoning (optional): ").strip()
            if not reasoning:
                reasoning = f"Manual {action.lower()} entry"

            confidence_input = input("Confidence (0-1, optional): ").strip()
            confidence = None
            if confidence_input:
                try:
                    confidence = float(confidence_input)
                    confidence = max(0, min(1, confidence))
                except ValueError:
                    confidence = None

            # Show current portfolio
            current_portfolio = self.portfolio_manager.get_portfolio_summary()
            print(f"\n💰 Current Portfolio Value: ${current_portfolio['total_value']:.2f}")
            print(f"💵 Available Cash: ${current_portfolio['cash']:.2f}")

            print("\n" + "-"*30)
            print("📋 TRADE SUMMARY:")
            print("-"*30)
            print(f"Symbol: {symbol}")
            print(f"Action: {action}")
            print(f"Quantity: {quantity:,}")
            print(f"Price: ${price:.2f}")
            print(f"Total: ${quantity * price:.2f}")
            print(f"Reasoning: {reasoning}")
            if confidence:
                print(f"Confidence: {confidence:.2f}")

            confirm = input("\nConfirm this trade? (y/n): ").strip().lower()
            if confirm in ['y', 'yes']:
                success, message = self.portfolio_manager.execute_trade(
                    symbol, action, quantity, price, reasoning, confidence
                )
                if success:
                    print(f"✅ {message}")
                    
                    # Show updated portfolio
                    updated_portfolio = self.portfolio_manager.get_portfolio_summary()
                    print(f"\n📊 Updated Portfolio Value: ${updated_portfolio['total_value']:.2f}")
                    print(f"💵 Cash Remaining: ${updated_portfolio['cash']:.2f}")
                    print(f"📈 Total Return: {updated_portfolio['total_return']:.2f}%")
                    
                    return True
                else:
                    print(f"❌ {message}")
                    return False
            else:
                print("❌ Trade cancelled")
                return False

        except KeyboardInterrupt:
            print("\n❌ Trade logging cancelled")
            return False
        except Exception as e:
            logger.error(f"Error in interactive trade logging: {e}")
            print(f"❌ Error: {e}")
            return False

class EnhancedMLTradingSystem:
    """Main enhanced ML trading system with proper portfolio tracking"""

    def __init__(self):
        self.config = config
        self.portfolio_manager = PortfolioManager()
        self.data_manager = EnhancedDataManager()
        self.ai_client = PerplexityClientSync()
        self.ml_engine = MLRecommendationEngine()
        self.risk_manager = RiskManager()
        self.db_manager = get_db_manager()
        self.trade_logger = TradeLogger()
        self.weekend_analyzer = WeekendAnalyzer()

        logger.info("Enhanced ML Trading System with Portfolio Tracking initialized")

    def run_daily_update(self) -> None:
        try:
            logger.info("🚀 Starting daily update with profitable stock analysis...")

            # Get updated portfolio data from actual trades
            portfolio_data = self.portfolio_manager.get_portfolio_summary()
            market_data = self._get_market_data()

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

    def _get_market_data(self) -> Dict[str, Any]:
        try:
            candidates = self.data_manager.screen_micro_caps()

            market_data = {
                'market_open': self.data_manager.validator.validate_market_hours(),
                'sentiment': 'Neutral',
                'candidates': []
            }

            for symbol in candidates[:10]:
                stock_data = self.data_manager.get_stock_data(symbol, validate=False)
                if stock_data:
                    market_data['candidates'].append({
                        'symbol': stock_data.symbol,
                        'price': stock_data.price,
                        'volume': stock_data.volume,
                        'change_percent': stock_data.change_percent
                    })

            return market_data

        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return {'market_open': False, 'sentiment': 'Neutral', 'candidates': []}

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
            print("🚀 ENHANCED ML AI TRADING SYSTEM - PROFITABLE STOCK RECOMMENDATIONS")
            print("="*80)

            if not recommendations:
                print("❌ No recommendations generated today.")
                print("This could be due to:")
                print("  • ETF holdings download issues")
                print("  • No qualifying micro-cap stocks found")
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
            print("🎯 PROFIT-OPTIMIZED RECOMMENDATIONS (Based on Current Market Analysis):")
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
            print("💡 These are AI-generated recommendations based on:")
            print("  📊 Technical analysis (RSI, momentum, support/resistance)")
            print("  📈 Volume and price action analysis")
            print("  💰 Value opportunity assessment")
            print("  🎯 Risk-adjusted profit potential")
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

    def run_interactive_mode(self) -> None:
        logger.info("🚀 Starting Enhanced ML Trading System...")

        while True:
            try:
                print("\n" + "="*70)
                print("🚀 ENHANCED ML AI TRADING SYSTEM - PROFIT OPTIMIZED")
                print("="*70)
                print("1. 📈 Daily Update & Profitable Stock Recommendations")
                print("2. 📊 Portfolio Summary")
                print("3. 📝 Trade Logger (Essential for ML Learning)")
                print("4. 🔍 Weekend Deep Analysis")
                print("5. 📉 Performance Report")
                print("6. 🤖 ML Model Status")
                print("7. ⚙️ System Settings")
                print("8. 🚪 Exit")

                choice = input("\nSelect option (1-8): ").strip()

                if choice == '1':
                    self.run_daily_update()

                elif choice == '2':
                    portfolio = self.portfolio_manager.get_portfolio_summary()
                    self._display_portfolio_summary(portfolio)

                elif choice == '3':
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
                    self._show_system_settings()

                elif choice == '8':
                    logger.info("👋 Exiting Enhanced ML Trading System...")
                    break

                else:
                    print("❌ Invalid option. Please select 1-8.")

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

        except Exception as e:
            logger.error(f"Error showing system settings: {e}")

def run_trade_logging_interface():
    """Updated trade logging interface"""
    trade_logger = TradeLogger()
    portfolio_manager = PortfolioManager()

    while True:
        try:
            print("\n" + "="*50)
            print("📝 TRADE LOGGING MENU")
            print("="*50)
            print("1. Log Single Trade")
            print("2. Show Recent Trades") 
            print("3. Trade Summary")
            print("4. Recalculate Portfolio from Trades")
            print("5. Back to Main Menu")

            choice = input("\nSelect option (1-5): ").strip()

            if choice == '1':
                trade_logger.log_trade_interactive()
            elif choice == '2':
                show_recent_trades(trade_logger.db_manager)
            elif choice == '3':
                show_trade_summary(trade_logger.db_manager)
            elif choice == '4':
                print("🔄 Recalculating portfolio from all trades...")
                portfolio = portfolio_manager.calculate_portfolio_from_trades()
                print(f"✅ Portfolio recalculated: ${portfolio['total_value']:.2f}")
            elif choice == '5':
                break
            else:
                print("❌ Invalid option. Please select 1-5.")

        except KeyboardInterrupt:
            print("\n❌ Cancelled")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def show_recent_trades(db_manager, days: int = 30):
    """Show recent trades"""
    try:
        trades_df = db_manager.get_trade_history(days=days)
        if trades_df.empty:
            print("📋 No trades found in the last 30 days")
            return
        
        print(f"\n📋 RECENT TRADES (Last {days} days):")
        print("-" * 60)
        
        for _, trade in trades_df.iterrows():
            print(f"{trade['timestamp'].strftime('%Y-%m-%d')} | "
                  f"{trade['action']} {trade['quantity']} {trade['symbol']} @ "
                  f"${trade['price']:.2f} = ${trade['total_amount']:.2f}")
            if trade['reasoning']:
                print(f"   Reason: {trade['reasoning'][:50]}...")
        
    except Exception as e:
        print(f"❌ Error showing recent trades: {e}")

def show_trade_summary(db_manager):
    """Show trade summary statistics"""
    try:
        trades_df = db_manager.get_trade_history(days=365)  # Last year
        if trades_df.empty:
            print("📋 No trades found")
            return
        
        print("\n📊 TRADE SUMMARY:")
        print("-" * 40)
        print(f"Total Trades: {len(trades_df)}")
        print(f"Buy Orders: {len(trades_df[trades_df['action'] == 'BUY'])}")
        print(f"Sell Orders: {len(trades_df[trades_df['action'] == 'SELL'])}")
        print(f"Total Volume: ${trades_df['total_amount'].sum():.2f}")
        print(f"Average Trade Size: ${trades_df['total_amount'].mean():.2f}")
        
        # Most traded symbols
        symbol_counts = trades_df['symbol'].value_counts()
        print(f"\nMost Traded Symbols:")
        for symbol, count in symbol_counts.head(5).items():
            print(f"  {symbol}: {count} trades")
        
    except Exception as e:
        print(f"❌ Error showing trade summary: {e}")

def main():
    try:
        print("🚀 Initializing Enhanced ML Trading System with Portfolio Tracking...")

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
