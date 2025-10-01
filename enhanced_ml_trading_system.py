"""
Enhanced AI Trading System with Machine Learning and Profitable Stock Analysis
Main application integrating ML recommendations, trade logging, and weekend analysis
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
          format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
logger.add(config.LOG_FILE, rotation="10 MB", retention="1 month", level=config.LOG_LEVEL,
          format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}")

from data_manager import EnhancedDataManager, StockData
from perplexity_client import PerplexityClientSync, TradingRecommendation
from database import get_db_manager, Position, Trade, AIRecommendation
from ml_engine import MLRecommendationEngine
from trade_logger import TradeLogger, run_trade_logging_interface
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
            position_size_percent = trade_value / portfolio_value
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
    """Enhanced portfolio management system"""

    def __init__(self):
        self.db_manager = get_db_manager()
        self.data_manager = EnhancedDataManager()
        self.risk_manager = RiskManager()
        self.initial_capital = config.INITIAL_CAPITAL

    def get_portfolio_summary(self) -> Dict[str, Any]:
        try:
            positions = self.db_manager.get_active_positions()

            total_market_value = sum(p.market_value for p in positions)
            cash = self.initial_capital - sum(p.entry_price * p.quantity for p in positions)
            total_value = total_market_value + cash
            total_return = ((total_value - self.initial_capital) / self.initial_capital) * 100

            daily_pnl = sum(p.unrealized_pnl_percent for p in positions) / len(positions) if positions else 0.0

            portfolio_summary = {
                'total_value': total_value,
                'cash': cash,
                'invested_amount': total_market_value,
                'total_return': total_return,
                'daily_pnl': daily_pnl,
                'num_positions': len(positions),
                'holdings': {p.symbol: p.quantity for p in positions}
            }

            self.db_manager.save_portfolio_snapshot(portfolio_summary)

            return portfolio_summary

        except Exception as e:
            logger.error(f"Error getting portfolio summary: {e}")
            return {
                'total_value': self.initial_capital,
                'cash': self.initial_capital,
                'invested_amount': 0.0,
                'total_return': 0.0,
                'daily_pnl': 0.0,
                'num_positions': 0,
                'holdings': {}
            }

class EnhancedMLTradingSystem:
    """Main enhanced ML trading system"""

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

        logger.info("Enhanced ML Trading System with Profitable Stock Analysis initialized")

    def run_daily_update(self) -> None:
        try:
            logger.info("🚀 Starting daily update with profitable stock analysis...")

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
            print("   📊 Technical analysis (RSI, momentum, support/resistance)")
            print("   📈 Volume and price action analysis")
            print("   💰 Value opportunity assessment")
            print("   🎯 Risk-adjusted profit potential")
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
                print("7. ⚙️  System Settings")
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
                print("⚠️  Need more trade data for advanced ML training")
                print("💡 Use Trade Logger to record your actual trades")

        except Exception as e:
            logger.error(f"Error showing ML model status: {e}")

    def _show_system_settings(self) -> None:
        try:
            print("\n" + "="*50)
            print("⚙️  SYSTEM SETTINGS")
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

def main():
    try:
        print("🚀 Initializing Enhanced ML Trading System with Profitable Stock Analysis...")

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
