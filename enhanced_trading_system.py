"""
Enhanced AI Trading System - Main Application
Integrates all components into a robust, production-ready trading system
"""
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from loguru import logger
import sys
from pathlib import Path

# Configure logging
from config import get_config
config = get_config()

# Configure loguru logger
logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    level=config.LOG_LEVEL,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)
logger.add(
    config.LOG_FILE,
    rotation="10 MB",
    retention="1 month",
    level=config.LOG_LEVEL,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
)

# Import our modules
from data_manager import EnhancedDataManager, StockData
from perplexity_client import PerplexityClientSync, TradingRecommendation
from database import get_db_manager, Position, Trade, AIRecommendation

class RiskManager:
    """Enhanced risk management system"""

    def __init__(self):
        self.config = config

    def validate_trade(self, action: str, symbol: str, quantity: int, 
                      price: float, portfolio_value: float, 
                      current_positions: List[Position]) -> Tuple[bool, str]:
        """
        Validate if a trade should be executed

        Returns:
            (is_valid, reason)
        """
        try:
            trade_value = quantity * price

            # Check position size limits
            position_size_percent = trade_value / portfolio_value
            if position_size_percent > self.config.MAX_POSITION_SIZE:
                return False, f"Position size {position_size_percent:.1%} exceeds maximum {self.config.MAX_POSITION_SIZE:.1%}"

            # Check maximum positions limit
            if action == 'BUY' and len(current_positions) >= self.config.MAX_POSITIONS:
                return False, f"Maximum positions limit ({self.config.MAX_POSITIONS}) reached"

            # Check daily loss limit
            daily_loss = self._calculate_daily_loss(current_positions)
            if daily_loss > self.config.MAX_DAILY_LOSS:
                return False, f"Daily loss limit ({self.config.MAX_DAILY_LOSS:.1%}) exceeded"

            # Validate symbol exists in current positions for SELL orders
            if action == 'SELL':
                position = next((p for p in current_positions if p.symbol == symbol), None)
                if not position:
                    return False, f"No position found for {symbol} to sell"
                if position.quantity < quantity:
                    return False, f"Insufficient quantity to sell (have {position.quantity}, want to sell {quantity})"

            return True, "Trade validation passed"

        except Exception as e:
            logger.error(f"Error validating trade: {e}")
            return False, f"Validation error: {e}"

    def _calculate_daily_loss(self, positions: List[Position]) -> float:
        """Calculate current daily loss percentage"""
        total_daily_pnl = sum(p.unrealized_pnl for p in positions)
        total_value = sum(p.market_value for p in positions)

        if total_value == 0:
            return 0.0

        return abs(min(0, total_daily_pnl)) / total_value

    def calculate_stop_loss(self, entry_price: float, action: str = 'BUY') -> float:
        """Calculate stop loss price"""
        if action == 'BUY':
            return entry_price * (1 - self.config.STOP_LOSS_PERCENTAGE)
        else:  # SELL/SHORT
            return entry_price * (1 + self.config.STOP_LOSS_PERCENTAGE)

    def calculate_take_profit(self, entry_price: float, action: str = 'BUY') -> float:
        """Calculate take profit price"""
        if action == 'BUY':
            return entry_price * (1 + self.config.TAKE_PROFIT_PERCENTAGE)
        else:  # SELL/SHORT
            return entry_price * (1 - self.config.TAKE_PROFIT_PERCENTAGE)

class PortfolioManager:
    """Enhanced portfolio management system"""

    def __init__(self):
        self.db_manager = get_db_manager()
        self.data_manager = EnhancedDataManager()
        self.risk_manager = RiskManager()
        self.initial_capital = config.INITIAL_CAPITAL

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary"""
        try:
            positions = self.db_manager.get_active_positions()

            # Calculate portfolio metrics
            total_market_value = sum(p.market_value for p in positions)
            total_unrealized_pnl = sum(p.unrealized_pnl for p in positions)
            cash = self.initial_capital - sum(p.entry_price * p.quantity for p in positions)
            total_value = total_market_value + cash
            total_return = ((total_value - self.initial_capital) / self.initial_capital) * 100

            # Calculate daily P&L (simplified - would need historical data for accurate calculation)
            daily_pnl = sum(p.unrealized_pnl_percent for p in positions) / len(positions) if positions else 0.0

            portfolio_summary = {
                'total_value': total_value,
                'cash': cash,
                'invested_amount': total_market_value,
                'total_return': total_return,
                'daily_pnl': daily_pnl,
                'num_positions': len(positions),
                'positions': [{
                    'symbol': p.symbol,
                    'quantity': p.quantity,
                    'entry_price': p.entry_price,
                    'current_price': p.current_price,
                    'market_value': p.market_value,
                    'unrealized_pnl': p.unrealized_pnl,
                    'unrealized_pnl_percent': p.unrealized_pnl_percent,
                    'stop_loss': p.stop_loss,
                    'take_profit': p.take_profit
                } for p in positions],
                'holdings': {p.symbol: p.quantity for p in positions}
            }

            # Save portfolio snapshot
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
                'positions': [],
                'holdings': {}
            }

    def update_positions(self) -> None:
        """Update all position values with current market data"""
        try:
            positions = self.db_manager.get_active_positions()

            for position in positions:
                # Get current market data
                stock_data = self.data_manager.get_stock_data(position.symbol)
                if stock_data:
                    # Update position data
                    position.current_price = stock_data.price
                    position.market_value = position.quantity * stock_data.price
                    position.unrealized_pnl = position.market_value - (position.quantity * position.entry_price)
                    position.unrealized_pnl_percent = (position.unrealized_pnl / (position.quantity * position.entry_price)) * 100

                    # Check stop loss and take profit
                    if position.stop_loss and stock_data.price <= position.stop_loss:
                        logger.warning(f"Stop loss triggered for {position.symbol} at ${stock_data.price}")
                        # In production, this would trigger an automatic sell order

                    if position.take_profit and stock_data.price >= position.take_profit:
                        logger.info(f"Take profit target reached for {position.symbol} at ${stock_data.price}")

                    # Save updated position
                    position_data = {
                        'symbol': position.symbol,
                        'quantity': position.quantity,
                        'entry_price': position.entry_price,
                        'current_price': position.current_price,
                        'market_value': position.market_value,
                        'unrealized_pnl': position.unrealized_pnl,
                        'unrealized_pnl_percent': position.unrealized_pnl_percent,
                        'stop_loss': position.stop_loss,
                        'take_profit': position.take_profit
                    }
                    self.db_manager.save_position(position_data)

        except Exception as e:
            logger.error(f"Error updating positions: {e}")

class EnhancedTradingSystem:
    """Main enhanced trading system"""

    def __init__(self):
        self.config = config
        self.portfolio_manager = PortfolioManager()
        self.data_manager = EnhancedDataManager()
        self.ai_client = PerplexityClientSync()
        self.risk_manager = RiskManager()
        self.db_manager = get_db_manager()

        logger.info("Enhanced Trading System initialized")

    def run_daily_update(self) -> None:
        """Run daily portfolio update and get AI recommendations"""
        try:
            logger.info("Starting daily update...")

            # Update position values
            self.portfolio_manager.update_positions()

            # Get portfolio summary
            portfolio_data = self.portfolio_manager.get_portfolio_summary()

            # Get market data
            market_data = self._get_market_data()

            # Check if it's a research day (weekly deep research)
            research_context = None
            if datetime.now().weekday() in config.DEEP_RESEARCH_DAYS:
                research_context = self._perform_deep_research()

            # Get AI recommendations
            recommendations = self.ai_client.get_trading_recommendation(
                portfolio_data, market_data, research_context
            )

            # Process recommendations
            for rec in recommendations:
                self._process_recommendation(rec, portfolio_data)

            # Log portfolio status
            logger.info(f"Portfolio Value: ${portfolio_data['total_value']:.2f}")
            logger.info(f"Total Return: {portfolio_data['total_return']:.2f}%")
            logger.info(f"Positions: {portfolio_data['num_positions']}")

        except Exception as e:
            logger.error(f"Error in daily update: {e}")

    def _get_market_data(self) -> Dict[str, Any]:
        """Get comprehensive market data"""
        try:
            # Get micro-cap candidates
            candidates = self.data_manager.screen_micro_caps()

            # Get benchmark data (simplified)
            market_data = {
                'market_open': self.data_manager.validator.validate_market_hours(),
                'sp500_change': 0.0,  # Would fetch from actual source
                'vix': 'N/A',  # Would fetch from actual source
                'sentiment': 'Neutral',  # Would calculate from news sentiment
                'candidates': []
            }

            # Get data for top candidates
            for symbol in candidates[:10]:  # Limit to top 10
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
            return {
                'market_open': False,
                'sp500_change': 0.0,
                'vix': 'N/A',
                'sentiment': 'Neutral',
                'candidates': []
            }

    def _perform_deep_research(self) -> str:
        """Perform weekly deep research"""
        try:
            logger.info("Performing weekly deep research...")

            # Get current positions
            positions = self.db_manager.get_active_positions()
            current_symbols = [p.symbol for p in positions]

            # Get potential new symbols
            candidates = self.data_manager.screen_micro_caps()
            research_symbols = current_symbols + candidates[:20]

            # Get research from AI
            research = self.ai_client.get_market_research(research_symbols)

            logger.info("Deep research completed")
            return research

        except Exception as e:
            logger.error(f"Error performing deep research: {e}")
            return ""

    def _process_recommendation(self, recommendation: TradingRecommendation, 
                              portfolio_data: Dict[str, Any]) -> None:
        """Process and potentially execute AI recommendation"""
        try:
            logger.info(f"Processing recommendation: {recommendation.action} {recommendation.symbol}")

            # Save recommendation to database
            rec_data = {
                'symbol': recommendation.symbol,
                'action': recommendation.action,
                'reasoning': recommendation.reasoning,
                'confidence': recommendation.confidence,
                'price_target': recommendation.price_target,
                'stop_loss': recommendation.stop_loss,
                'position_size': recommendation.position_size,
                'time_horizon': recommendation.time_horizon,
                'risk_level': recommendation.risk_level
            }
            self.db_manager.save_ai_recommendation(rec_data)

            # Get current market data for the symbol
            stock_data = self.data_manager.get_stock_data(recommendation.symbol)
            if not stock_data:
                logger.warning(f"Could not get market data for {recommendation.symbol}")
                return

            # Validate the recommendation meets our criteria
            if not stock_data.is_micro_cap:
                logger.warning(f"{recommendation.symbol} is not a micro-cap stock")
                return

            if not stock_data.has_sufficient_volume:
                logger.warning(f"{recommendation.symbol} has insufficient volume")
                return

            # Calculate trade parameters
            if recommendation.action == 'BUY':
                position_size = recommendation.position_size or 0.1  # Default 10%
                trade_value = portfolio_data['total_value'] * position_size
                quantity = int(trade_value / stock_data.price)

                if quantity == 0:
                    logger.warning(f"Calculated quantity is 0 for {recommendation.symbol}")
                    return

                # Validate trade with risk manager
                positions = self.db_manager.get_active_positions()
                is_valid, reason = self.risk_manager.validate_trade(
                    'BUY', recommendation.symbol, quantity, stock_data.price,
                    portfolio_data['total_value'], positions
                )

                if not is_valid:
                    logger.warning(f"Trade validation failed: {reason}")
                    return

                # Execute buy order (simulated)
                self._execute_trade('BUY', recommendation.symbol, quantity, 
                                  stock_data.price, recommendation, portfolio_data)

            elif recommendation.action == 'SELL':
                # Find existing position
                positions = self.db_manager.get_active_positions()
                position = next((p for p in positions if p.symbol == recommendation.symbol), None)

                if not position:
                    logger.warning(f"No position found to sell for {recommendation.symbol}")
                    return

                quantity = position.quantity  # Sell entire position

                # Execute sell order (simulated)
                self._execute_trade('SELL', recommendation.symbol, quantity,
                                  stock_data.price, recommendation, portfolio_data)

        except Exception as e:
            logger.error(f"Error processing recommendation: {e}")

    def _execute_trade(self, action: str, symbol: str, quantity: int, price: float,
                      recommendation: TradingRecommendation, 
                      portfolio_data: Dict[str, Any]) -> None:
        """Execute trade (simulated - in production would connect to broker)"""
        try:
            total_amount = quantity * price

            # Save trade to database
            trade_data = {
                'symbol': symbol,
                'action': action,
                'quantity': quantity,
                'price': price,
                'total_amount': total_amount,
                'fees': 0.0,  # Assuming no fees for simplicity
                'reasoning': recommendation.reasoning,
                'confidence': recommendation.confidence,
                'portfolio_value_before': portfolio_data['total_value']
            }

            if action == 'BUY':
                # Create new position
                stop_loss = self.risk_manager.calculate_stop_loss(price, 'BUY')
                take_profit = self.risk_manager.calculate_take_profit(price, 'BUY')

                position_data = {
                    'symbol': symbol,
                    'quantity': quantity,
                    'entry_price': price,
                    'current_price': price,
                    'market_value': total_amount,
                    'unrealized_pnl': 0.0,
                    'unrealized_pnl_percent': 0.0,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit
                }

                self.db_manager.save_position(position_data)
                logger.info(f"BUY executed: {quantity} shares of {symbol} at ${price:.2f}")

            elif action == 'SELL':
                # Close position
                self.db_manager.close_position(symbol)
                logger.info(f"SELL executed: {quantity} shares of {symbol} at ${price:.2f}")

            # Calculate portfolio value after trade
            updated_portfolio = self.portfolio_manager.get_portfolio_summary()
            trade_data['portfolio_value_after'] = updated_portfolio['total_value']

            # Save trade
            self.db_manager.save_trade(trade_data)

            logger.info(f"Trade executed successfully: {action} {quantity} {symbol} at ${price:.2f}")

        except Exception as e:
            logger.error(f"Error executing trade: {e}")

    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        try:
            # Get performance metrics
            metrics = self.db_manager.calculate_performance_metrics()

            # Get portfolio history
            portfolio_history = self.db_manager.get_portfolio_history(30)

            # Get trade history
            trade_history = self.db_manager.get_trade_history(30)

            report = {
                'timestamp': datetime.now(),
                'portfolio_summary': self.portfolio_manager.get_portfolio_summary(),
                'performance_metrics': metrics,
                'portfolio_history': portfolio_history.to_dict('records') if not portfolio_history.empty else [],
                'recent_trades': trade_history.to_dict('records') if not trade_history.empty else []
            }

            return report

        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return {}

    def run_interactive_mode(self) -> None:
        """Run system in interactive mode for testing"""
        logger.info("Starting Interactive Mode...")

        while True:
            try:
                print("\n" + "="*50)
                print("Enhanced AI Trading System - Interactive Mode")
                print("="*50)
                print("1. Run Daily Update")
                print("2. Show Portfolio Summary")
                print("3. Show Performance Report")
                print("4. Manual Trade")
                print("5. Exit")

                choice = input("\nSelect option (1-5): ").strip()

                if choice == '1':
                    self.run_daily_update()
                    print("Daily update completed!")

                elif choice == '2':
                    portfolio = self.portfolio_manager.get_portfolio_summary()
                    print(f"\nPortfolio Summary:")
                    print(f"Total Value: ${portfolio['total_value']:.2f}")
                    print(f"Cash: ${portfolio['cash']:.2f}")
                    print(f"Invested: ${portfolio['invested_amount']:.2f}")
                    print(f"Total Return: {portfolio['total_return']:.2f}%")
                    print(f"Positions: {portfolio['num_positions']}")

                    if portfolio['positions']:
                        print("\nCurrent Positions:")
                        for pos in portfolio['positions']:
                            print(f"  {pos['symbol']}: {pos['quantity']} shares @ ${pos['current_price']:.2f} "
                                f"(P&L: {pos['unrealized_pnl_percent']:.2f}%)")

                elif choice == '3':
                    report = self.generate_performance_report()
                    print("\nPerformance Report:")
                    metrics = report.get('performance_metrics', {})
                    for key, value in metrics.items():
                        if isinstance(value, float):
                            print(f"  {key.replace('_', ' ').title()}: {value:.2f}%")
                        else:
                            print(f"  {key.replace('_', ' ').title()}: {value}")

                elif choice == '4':
                    print("Manual trading not implemented in demo mode")

                elif choice == '5':
                    logger.info("Exiting interactive mode...")
                    break

                else:
                    print("Invalid option. Please select 1-5.")

            except KeyboardInterrupt:
                logger.info("Interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error in interactive mode: {e}")

def main():
    """Main entry point"""
    try:
        # Initialize the trading system
        trading_system = EnhancedTradingSystem()

        # Run in interactive mode for demonstration
        trading_system.run_interactive_mode()

    except Exception as e:
        logger.error(f"Critical error in main: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
