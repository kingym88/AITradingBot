"""
Database management system for AI Trading System
Provides persistent storage for portfolio, trades, and analysis
"""
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime
from typing import List, Optional, Dict, Any
import pandas as pd
from loguru import logger
from config import get_config

config = get_config()

Base = declarative_base()

class Portfolio(Base):
    __tablename__ = 'portfolio'

    id = Column(Integer, primary_key=True)
    total_value = Column(Float, nullable=False)
    cash = Column(Float, nullable=False)
    invested_amount = Column(Float, nullable=False)
    total_return = Column(Float, nullable=False)
    daily_pnl = Column(Float, nullable=False)
    num_positions = Column(Integer, nullable=False)
    timestamp = Column(DateTime, default=datetime.now)

class Position(Base):
    __tablename__ = 'positions'

    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), nullable=False)
    quantity = Column(Integer, nullable=False)
    entry_price = Column(Float, nullable=False)
    current_price = Column(Float, nullable=False)
    market_value = Column(Float, nullable=False)
    unrealized_pnl = Column(Float, nullable=False)
    unrealized_pnl_percent = Column(Float, nullable=False)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    entry_date = Column(DateTime, default=datetime.now)
    last_updated = Column(DateTime, default=datetime.now)
    is_active = Column(Boolean, default=True)

class Trade(Base):
    __tablename__ = 'trades'

    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), nullable=False)
    action = Column(String(10), nullable=False)  # BUY, SELL
    quantity = Column(Integer, nullable=False)
    price = Column(Float, nullable=False)
    total_amount = Column(Float, nullable=False)
    fees = Column(Float, default=0.0)
    reasoning = Column(Text)
    confidence = Column(Float)
    timestamp = Column(DateTime, default=datetime.now)
    portfolio_value_before = Column(Float)
    portfolio_value_after = Column(Float)

class AIRecommendation(Base):
    __tablename__ = 'ai_recommendations'

    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), nullable=False)
    action = Column(String(10), nullable=False)
    reasoning = Column(Text, nullable=False)
    confidence = Column(Float, nullable=False)
    price_target = Column(Float)
    stop_loss = Column(Float)
    position_size = Column(Float)
    time_horizon = Column(String(20))
    risk_level = Column(String(20))
    executed = Column(Boolean, default=False)
    timestamp = Column(DateTime, default=datetime.now)

class MarketData(Base):
    __tablename__ = 'market_data'

    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), nullable=False)
    price = Column(Float, nullable=False)
    volume = Column(Integer, nullable=False)
    market_cap = Column(Integer)
    change = Column(Float, nullable=False)
    change_percent = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.now)

class PerformanceMetrics(Base):
    __tablename__ = 'performance_metrics'

    id = Column(Integer, primary_key=True)
    portfolio_value = Column(Float, nullable=False)
    benchmark_value = Column(Float, nullable=False)  # S&P 500 comparison
    alpha = Column(Float)
    beta = Column(Float)
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float)
    win_rate = Column(Float)
    avg_win = Column(Float)
    avg_loss = Column(Float)
    total_trades = Column(Integer, nullable=False)
    timestamp = Column(DateTime, default=datetime.now)

class DatabaseManager:
    """Database management class"""

    def __init__(self):
        self.engine = create_engine(config.DATABASE_URL, echo=False)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)

    def get_session(self) -> Session:
        """Get database session"""
        return self.SessionLocal()

    def save_portfolio_snapshot(self, portfolio_data: Dict[str, Any]) -> None:
        """Save current portfolio state"""
        try:
            with self.get_session() as session:
                portfolio = Portfolio(
                    total_value=portfolio_data['total_value'],
                    cash=portfolio_data['cash'],
                    invested_amount=portfolio_data['invested_amount'],
                    total_return=portfolio_data['total_return'],
                    daily_pnl=portfolio_data['daily_pnl'],
                    num_positions=portfolio_data['num_positions']
                )
                session.add(portfolio)
                session.commit()
                logger.info("Portfolio snapshot saved")
        except Exception as e:
            logger.error(f"Error saving portfolio snapshot: {e}")

    def save_position(self, position_data: Dict[str, Any]) -> None:
        """Save or update position"""
        try:
            with self.get_session() as session:
                # Check if position already exists
                existing = session.query(Position).filter_by(
                    symbol=position_data['symbol'], is_active=True
                ).first()

                if existing:
                    # Update existing position
                    for key, value in position_data.items():
                        if hasattr(existing, key):
                            setattr(existing, key, value)
                    existing.last_updated = datetime.now()
                else:
                    # Create new position
                    position = Position(**position_data)
                    session.add(position)

                session.commit()
        except Exception as e:
            logger.error(f"Error saving position: {e}")

    def save_trade(self, trade_data: Dict[str, Any]) -> None:
        """Save trade execution"""
        try:
            with self.get_session() as session:
                trade = Trade(**trade_data)
                session.add(trade)
                session.commit()
                logger.info(f"Trade saved: {trade_data['action']} {trade_data['quantity']} {trade_data['symbol']}")
        except Exception as e:
            logger.error(f"Error saving trade: {e}")

    def save_ai_recommendation(self, recommendation_data: Dict[str, Any]) -> None:
        """Save AI recommendation"""
        try:
            with self.get_session() as session:
                rec = AIRecommendation(**recommendation_data)
                session.add(rec)
                session.commit()
        except Exception as e:
            logger.error(f"Error saving AI recommendation: {e}")

    def save_market_data(self, market_data: Dict[str, Any]) -> None:
        """Save market data point"""
        try:
            with self.get_session() as session:
                data = MarketData(**market_data)
                session.add(data)
                session.commit()
        except Exception as e:
            logger.error(f"Error saving market data: {e}")

    def get_active_positions(self) -> List[Position]:
        """Get all active positions"""
        try:
            with self.get_session() as session:
                return session.query(Position).filter_by(is_active=True).all()
        except Exception as e:
            logger.error(f"Error getting active positions: {e}")
            return []

    def get_portfolio_history(self, days: int = 30) -> pd.DataFrame:
        """Get portfolio history as DataFrame"""
        try:
            with self.get_session() as session:
                cutoff_date = datetime.now() - pd.Timedelta(days=days)
                query = session.query(Portfolio).filter(
                    Portfolio.timestamp >= cutoff_date
                ).order_by(Portfolio.timestamp)

                data = [{
                    'timestamp': p.timestamp,
                    'total_value': p.total_value,
                    'cash': p.cash,
                    'invested_amount': p.invested_amount,
                    'total_return': p.total_return,
                    'daily_pnl': p.daily_pnl,
                    'num_positions': p.num_positions
                } for p in query.all()]

                return pd.DataFrame(data)
        except Exception as e:
            logger.error(f"Error getting portfolio history: {e}")
            return pd.DataFrame()

    def get_trade_history(self, days: int = 30) -> pd.DataFrame:
        """Get trade history as DataFrame"""
        try:
            with self.get_session() as session:
                cutoff_date = datetime.now() - pd.Timedelta(days=days)
                query = session.query(Trade).filter(
                    Trade.timestamp >= cutoff_date
                ).order_by(Trade.timestamp.desc())

                data = [{
                    'timestamp': t.timestamp,
                    'symbol': t.symbol,
                    'action': t.action,
                    'quantity': t.quantity,
                    'price': t.price,
                    'total_amount': t.total_amount,
                    'reasoning': t.reasoning,
                    'confidence': t.confidence
                } for t in query.all()]

                return pd.DataFrame(data)
        except Exception as e:
            logger.error(f"Error getting trade history: {e}")
            return pd.DataFrame()

    def calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate and save performance metrics"""
        try:
            portfolio_df = self.get_portfolio_history(90)  # 3 months
            if portfolio_df.empty:
                return {}

            # Calculate basic metrics
            portfolio_df['returns'] = portfolio_df['total_value'].pct_change()

            metrics = {
                'total_return': (portfolio_df['total_value'].iloc[-1] / portfolio_df['total_value'].iloc[0] - 1) * 100,
                'volatility': portfolio_df['returns'].std() * (252 ** 0.5) * 100,  # Annualized
                'sharpe_ratio': (portfolio_df['returns'].mean() / portfolio_df['returns'].std()) * (252 ** 0.5),
                'max_drawdown': ((portfolio_df['total_value'] / portfolio_df['total_value'].cummax()) - 1).min() * 100,
                'win_rate': self._calculate_win_rate(),
                'total_trades': self._count_trades()
            }

            # Save metrics to database
            with self.get_session() as session:
                perf_metrics = PerformanceMetrics(
                    portfolio_value=portfolio_df['total_value'].iloc[-1],
                    benchmark_value=100,  # Placeholder for benchmark
                    sharpe_ratio=metrics.get('sharpe_ratio'),
                    max_drawdown=metrics.get('max_drawdown'),
                    win_rate=metrics.get('win_rate'),
                    total_trades=metrics.get('total_trades', 0)
                )
                session.add(perf_metrics)
                session.commit()

            return metrics

        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {}

    def _calculate_win_rate(self) -> float:
        """Calculate win rate from trades"""
        try:
            with self.get_session() as session:
                trades = session.query(Trade).all()
                if not trades:
                    return 0.0

                winning_trades = 0
                total_trades = 0

                # Group trades by symbol to calculate P&L
                trade_pairs = {}
                for trade in trades:
                    if trade.symbol not in trade_pairs:
                        trade_pairs[trade.symbol] = []
                    trade_pairs[trade.symbol].append(trade)

                for symbol, symbol_trades in trade_pairs.items():
                    symbol_trades.sort(key=lambda x: x.timestamp)
                    position = 0
                    entry_price = 0

                    for trade in symbol_trades:
                        if trade.action == 'BUY':
                            if position == 0:
                                entry_price = trade.price
                            position += trade.quantity
                        elif trade.action == 'SELL' and position > 0:
                            pnl = (trade.price - entry_price) * trade.quantity
                            if pnl > 0:
                                winning_trades += 1
                            total_trades += 1
                            position = max(0, position - trade.quantity)

                return (winning_trades / total_trades * 100) if total_trades > 0 else 0.0

        except Exception as e:
            logger.error(f"Error calculating win rate: {e}")
            return 0.0

    def _count_trades(self) -> int:
        """Count total number of trades"""
        try:
            with self.get_session() as session:
                return session.query(Trade).count()
        except Exception as e:
            logger.error(f"Error counting trades: {e}")
            return 0

    def close_position(self, symbol: str) -> None:
        """Mark position as closed"""
        try:
            with self.get_session() as session:
                position = session.query(Position).filter_by(
                    symbol=symbol, is_active=True
                ).first()
                if position:
                    position.is_active = False
                    position.last_updated = datetime.now()
                    session.commit()
                    logger.info(f"Position closed: {symbol}")
        except Exception as e:
            logger.error(f"Error closing position: {e}")

    def get_unexecuted_recommendations(self) -> List[AIRecommendation]:
        """Get AI recommendations that haven't been executed"""
        try:
            with self.get_session() as session:
                return session.query(AIRecommendation).filter_by(executed=False).all()
        except Exception as e:
            logger.error(f"Error getting unexecuted recommendations: {e}")
            return []

    def mark_recommendation_executed(self, recommendation_id: int) -> None:
        """Mark recommendation as executed"""
        try:
            with self.get_session() as session:
                rec = session.query(AIRecommendation).get(recommendation_id)
                if rec:
                    rec.executed = True
                    session.commit()
        except Exception as e:
            logger.error(f"Error marking recommendation executed: {e}")

# Global database manager instance
db_manager = DatabaseManager()

def get_db_manager() -> DatabaseManager:
    """Get the global database manager instance"""
    return db_manager
