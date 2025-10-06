"""
Performance Analysis Tool for Enhanced AI Trading System
Provides detailed analytics and visualizations
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

from database import get_db_manager
from config import get_config

config = get_config()

class PerformanceAnalyzer:
    """Advanced performance analysis for the trading system"""

    def __init__(self):
        self.db_manager = get_db_manager()

    def generate_comprehensive_report(self, days: int = 90) -> Dict[str, Any]:
        """Generate comprehensive performance report"""

        print("Generating comprehensive performance report...")

        # Get data
        portfolio_history = self.db_manager.get_portfolio_history(days)
        trade_history = self.db_manager.get_trade_history(days)

        if portfolio_history.empty:
            print("No portfolio data available for analysis")
            return {}

        # Calculate metrics
        returns = self._calculate_returns(portfolio_history)
        risk_metrics = self._calculate_risk_metrics(portfolio_history)
        trade_metrics = self._calculate_trade_metrics(trade_history)

        report = {
            'analysis_period': f"{days} days",
            'report_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'returns': returns,
            'risk_metrics': risk_metrics,
            'trade_metrics': trade_metrics,
            'portfolio_data': portfolio_history.to_dict('records'),
            'trade_data': trade_history.to_dict('records')
        }

        return report

    def _calculate_returns(self, portfolio_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate return metrics"""
        if portfolio_df.empty or len(portfolio_df) < 2:
            return {}

        portfolio_df = portfolio_df.sort_values('timestamp')
        initial_value = portfolio_df['total_value'].iloc[0]
        final_value = portfolio_df['total_value'].iloc[-1]

        total_return = ((final_value - initial_value) / initial_value) * 100

        # Daily returns
        portfolio_df['daily_return'] = portfolio_df['total_value'].pct_change()
        daily_returns = portfolio_df['daily_return'].dropna()

        if len(daily_returns) == 0:
            return {'total_return': total_return}

        avg_daily_return = daily_returns.mean() * 100
        annualized_return = ((1 + daily_returns.mean()) ** 252 - 1) * 100

        return {
            'total_return': round(total_return, 2),
            'average_daily_return': round(avg_daily_return, 3),
            'annualized_return': round(annualized_return, 2),
            'best_day': round(daily_returns.max() * 100, 2),
            'worst_day': round(daily_returns.min() * 100, 2)
        }

    def _calculate_risk_metrics(self, portfolio_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate risk-adjusted metrics"""
        if portfolio_df.empty or len(portfolio_df) < 2:
            return {}

        portfolio_df = portfolio_df.sort_values('timestamp')
        portfolio_df['daily_return'] = portfolio_df['total_value'].pct_change()
        daily_returns = portfolio_df['daily_return'].dropna()

        if len(daily_returns) == 0:
            return {}

        # Volatility (annualized)
        volatility = daily_returns.std() * (252 ** 0.5) * 100

        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * (252 ** 0.5) if daily_returns.std() > 0 else 0

        # Maximum drawdown
        rolling_max = portfolio_df['total_value'].expanding().max()
        drawdown = (portfolio_df['total_value'] - rolling_max) / rolling_max * 100
        max_drawdown = drawdown.min()

        # Sortino ratio (downside deviation)
        negative_returns = daily_returns[daily_returns < 0]
        downside_deviation = negative_returns.std() * (252 ** 0.5)
        sortino_ratio = (daily_returns.mean() * 252) / downside_deviation if downside_deviation > 0 else 0

        return {
            'volatility': round(volatility, 2),
            'sharpe_ratio': round(sharpe_ratio, 3),
            'max_drawdown': round(max_drawdown, 2),
            'sortino_ratio': round(sortino_ratio, 3),
            'var_95': round(np.percentile(daily_returns, 5) * 100, 2)  # Value at Risk 95%
        }

    def _calculate_trade_metrics(self, trade_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate trading-specific metrics"""
        if trade_df.empty:
            return {'total_trades': 0}

        total_trades = len(trade_df)
        buy_trades = len(trade_df[trade_df['action'] == 'BUY'])
        sell_trades = len(trade_df[trade_df['action'] == 'SELL'])

        # Calculate win rate (simplified - pairs buy/sell orders)
        symbols = trade_df['symbol'].unique()
        winning_trades = 0
        total_completed_trades = 0

        for symbol in symbols:
            symbol_trades = trade_df[trade_df['symbol'] == symbol].sort_values('timestamp')

            position = 0
            entry_price = 0

            for _, trade in symbol_trades.iterrows():
                if trade['action'] == 'BUY':
                    if position == 0:
                        entry_price = trade['price']
                    position += trade['quantity']
                elif trade['action'] == 'SELL' and position > 0:
                    exit_price = trade['price']
                    if exit_price > entry_price:
                        winning_trades += 1
                    total_completed_trades += 1
                    position = max(0, position - trade['quantity'])

        win_rate = (winning_trades / total_completed_trades * 100) if total_completed_trades > 0 else 0

        # Average trade confidence
        avg_confidence = trade_df['confidence'].mean() if 'confidence' in trade_df.columns else 0

        return {
            'total_trades': total_trades,
            'buy_trades': buy_trades,
            'sell_trades': sell_trades,
            'completed_trades': total_completed_trades,
            'win_rate': round(win_rate, 1),
            'average_confidence': round(avg_confidence, 2) if avg_confidence else 'N/A',
            'unique_symbols': len(symbols),
            'most_traded_symbol': trade_df['symbol'].value_counts().index[0] if not trade_df.empty else 'N/A'
        }

    def create_performance_visualization(self, days: int = 30) -> str:
        """Create performance visualization charts"""
        try:
            portfolio_history = self.db_manager.get_portfolio_history(days)

            if portfolio_history.empty:
                return "No data available for visualization"

            # Set up the plotting style
            plt.style.use('seaborn-v0_8')
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Enhanced AI Trading System - Performance Dashboard ({days} days)', fontsize=16, fontweight='bold')

            # Plot 1: Portfolio Value Over Time
            portfolio_history['timestamp'] = pd.to_datetime(portfolio_history['timestamp'])
            ax1.plot(portfolio_history['timestamp'], portfolio_history['total_value'], 
                    linewidth=2, color='blue', marker='o', markersize=3)
            ax1.axhline(y=config.INITIAL_CAPITAL, color='red', linestyle='--', alpha=0.7, label=f'Initial Capital (${config.INITIAL_CAPITAL})')
            ax1.set_title('Portfolio Value Over Time', fontweight='bold')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Portfolio Value ($)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Plot 2: Daily Returns Distribution
            if len(portfolio_history) > 1:
                daily_returns = portfolio_history['total_value'].pct_change().dropna() * 100
                ax2.hist(daily_returns, bins=20, alpha=0.7, color='green', edgecolor='black')
                ax2.axvline(daily_returns.mean(), color='red', linestyle='--', label=f'Mean: {daily_returns.mean():.2f}%')
                ax2.set_title('Daily Returns Distribution', fontweight='bold')
                ax2.set_xlabel('Daily Return (%)')
                ax2.set_ylabel('Frequency')
                ax2.legend()
                ax2.grid(True, alpha=0.3)

            # Plot 3: Cash vs Invested Amount
            ax3.plot(portfolio_history['timestamp'], portfolio_history['cash'], 
                    label='Cash', linewidth=2, color='orange')
            ax3.plot(portfolio_history['timestamp'], portfolio_history['invested_amount'], 
                    label='Invested', linewidth=2, color='purple')
            ax3.set_title('Cash vs Invested Amount', fontweight='bold')
            ax3.set_xlabel('Date')
            ax3.set_ylabel('Amount ($)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # Plot 4: Number of Positions Over Time
            ax4.plot(portfolio_history['timestamp'], portfolio_history['num_positions'], 
                    linewidth=2, color='brown', marker='s', markersize=4)
            ax4.set_title('Number of Positions Over Time', fontweight='bold')
            ax4.set_xlabel('Date')
            ax4.set_ylabel('Number of Positions')
            ax4.grid(True, alpha=0.3)

            # Adjust layout and save
            plt.tight_layout()

            # Create reports directory if it doesn't exist
            config.REPORTS_DIR.mkdir(exist_ok=True)

            filename = f"performance_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            filepath = config.REPORTS_DIR / filename

            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()

            return f"Performance dashboard saved as: {filepath}"

        except Exception as e:
            return f"Error creating visualization: {e}"

    def print_summary_report(self, days: int = 30):
        """Print a formatted summary report"""
        report = self.generate_comprehensive_report(days)

        if not report:
            print("No data available for analysis")
            return

        print("\n" + "="*60)
        print("🚀 ENHANCED AI TRADING SYSTEM - PERFORMANCE REPORT")
        print("="*60)
        print(f"📅 Analysis Period: {report['analysis_period']}")
        print(f"🕐 Generated: {report['report_date']}")
        print()

        # Returns section
        print("📈 RETURNS ANALYSIS")
        print("-" * 30)
        returns = report.get('returns', {})
        for key, value in returns.items():
            formatted_key = key.replace('_', ' ').title()
            if isinstance(value, (int, float)):
                print(f"{formatted_key:.<25} {value:>8.2f}%")
            else:
                print(f"{formatted_key:.<25} {value:>10}")

        # Risk metrics section
        print("\n⚠️  RISK METRICS")
        print("-" * 30)
        risk = report.get('risk_metrics', {})
        for key, value in risk.items():
            formatted_key = key.replace('_', ' ').title()
            if 'ratio' in key.lower():
                print(f"{formatted_key:.<25} {value:>10.3f}")
            else:
                print(f"{formatted_key:.<25} {value:>8.2f}%")

        # Trading metrics section
        print("\n📊 TRADING METRICS")
        print("-" * 30)
        trades = report.get('trade_metrics', {})
        for key, value in trades.items():
            formatted_key = key.replace('_', ' ').title()
            if isinstance(value, (int, float)) and 'rate' in key.lower():
                print(f"{formatted_key:.<25} {value:>8.1f}%")
            elif isinstance(value, (int, float)):
                print(f"{formatted_key:.<25} {value:>10}")
            else:
                print(f"{formatted_key:.<25} {value:>10}")

        print("\n" + "="*60)
        print("📊 Dashboard: Run create_performance_visualization() to generate charts")
        print("="*60)

def main():
    """Main function for standalone execution"""
    analyzer = PerformanceAnalyzer()

    print("Enhanced AI Trading System - Performance Analyzer")
    print("=" * 50)

    try:
        days = input("Enter analysis period in days (default 30): ").strip()
        days = int(days) if days else 30
    except ValueError:
        days = 30
        print("Using default 30 days")

    # Generate and print report
    analyzer.print_summary_report(days)

    # Create visualization
    create_viz = input("\nCreate performance dashboard? (y/n): ").strip().lower()
    if create_viz in ['y', 'yes']:
        result = analyzer.create_performance_visualization(days)
        print(f"\n{result}")

if __name__ == "__main__":
    main()
