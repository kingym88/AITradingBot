"""
Trade Logging System for ML Learning
"""
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass
from loguru import logger

from database import get_db_manager, Trade
from data_manager import EnhancedDataManager
from config import get_config

config = get_config()

@dataclass
class TradeEntry:
    symbol: str
    action: str  # BUY, SELL
    quantity: int
    price: float
    date: datetime
    reasoning: Optional[str] = None
    confidence: Optional[float] = None
    source: str = "manual"

class TradeLogger:
    def __init__(self):
        self.db_manager = get_db_manager()
        self.data_manager = EnhancedDataManager()

    def log_trade_interactive(self) -> bool:
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

            while True:
                date_input = input("Date (YYYY-MM-DD) or press Enter for today: ").strip()
                if not date_input:
                    trade_date = datetime.now()
                    break
                else:
                    try:
                        trade_date = datetime.strptime(date_input, "%Y-%m-%d")
                        break
                    except ValueError:
                        print("❌ Please use format YYYY-MM-DD")

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

            trade_entry = TradeEntry(
                symbol=symbol,
                action=action,
                quantity=quantity,
                price=price,
                date=trade_date,
                reasoning=reasoning,
                confidence=confidence,
                source="manual"
            )

            print("\n" + "-"*30)
            print("📋 TRADE SUMMARY:")
            print("-"*30)
            print(f"Symbol: {trade_entry.symbol}")
            print(f"Action: {trade_entry.action}")
            print(f"Quantity: {trade_entry.quantity:,}")
            print(f"Price: ${trade_entry.price:.2f}")
            print(f"Total: ${trade_entry.quantity * trade_entry.price:.2f}")
            print(f"Date: {trade_entry.date.strftime('%Y-%m-%d')}")
            print(f"Reasoning: {trade_entry.reasoning}")
            if trade_entry.confidence:
                print(f"Confidence: {trade_entry.confidence:.2f}")

            confirm = input("\nConfirm this trade? (y/n): ").strip().lower()
            if confirm in ['y', 'yes']:
                success = self.log_trade(trade_entry)
                if success:
                    print("✅ Trade logged successfully!")
                    return True
                else:
                    print("❌ Failed to log trade")
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

    def log_trade(self, trade_entry: TradeEntry) -> bool:
        try:
            total_amount = trade_entry.quantity * trade_entry.price

            trade_data = {
                'symbol': trade_entry.symbol,
                'action': trade_entry.action,
                'quantity': trade_entry.quantity,
                'price': trade_entry.price,
                'total_amount': total_amount,
                'fees': 0.0,
                'reasoning': trade_entry.reasoning,
                'confidence': trade_entry.confidence,
                'timestamp': trade_entry.date,
                'portfolio_value_before': None,
                'portfolio_value_after': None
            }

            self.db_manager.save_trade(trade_data)

            logger.info(f"Trade logged: {trade_entry.action} {trade_entry.quantity} {trade_entry.symbol} @ ${trade_entry.price:.2f}")
            return True

        except Exception as e:
            logger.error(f"Error logging trade: {e}")
            return False

def run_trade_logging_interface():
    trade_logger = TradeLogger()

    while True:
        try:
            print("\n" + "="*50)
            print("📝 TRADE LOGGING MENU")
            print("="*50)
            print("1. Log Single Trade")
            print("2. Show Recent Trades") 
            print("3. Trade Summary")
            print("4. Back to Main Menu")

            choice = input("\nSelect option (1-4): ").strip()

            if choice == '1':
                trade_logger.log_trade_interactive()
            elif choice == '2':
                print("Recent trades feature coming soon...")
            elif choice == '3':
                print("Trade summary feature coming soon...")
            elif choice == '4':
                break
            else:
                print("❌ Invalid option. Please select 1-4.")

        except KeyboardInterrupt:
            print("\n❌ Cancelled")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    run_trade_logging_interface()
