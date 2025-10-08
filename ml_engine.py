"""
Machine Learning Recommendation Engine for AI Trading System
Enhanced with Profitable Stock Analysis and Debug Logging
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from loguru import logger
from config import get_config
from database import get_db_manager
from data_manager import EnhancedDataManager, PRODUCTION_MICRO_CAPS

config = get_config()

class ProfitabilityAnalyzer:
    """Advanced analyzer for determining most profitable stock opportunities"""
    
    def __init__(self, data_manager: EnhancedDataManager):
        self.data_manager = data_manager
    
    def analyze_profitability(self, symbol: str) -> Optional[Dict[str, Any]]:
        try:
            stock_data = self.data_manager.get_stock_data(symbol, validate=False)
            if not stock_data:
                return None
            
            hist_data = self.data_manager.get_historical_data(symbol, period="6mo")
            if hist_data.empty or len(hist_data) < 50:
                return None
            
            technical_score = self._calculate_technical_score(hist_data, stock_data)
            momentum_score = self._calculate_momentum_score(hist_data)
            value_score = self._calculate_value_score(stock_data, hist_data)
            volume_score = self._calculate_volume_score(hist_data, stock_data)
            volatility_score = self._calculate_volatility_score(hist_data)
            
            composite_score = (
                technical_score * 0.25 +
                momentum_score * 0.25 +
                value_score * 0.20 +
                volume_score * 0.15 +
                volatility_score * 0.15
            )
            
            return {
                'symbol': symbol,
                'composite_score': composite_score,
                'technical_score': technical_score,
                'momentum_score': momentum_score,
                'value_score': value_score,
                'volume_score': volume_score,
                'volatility_score': volatility_score,
                'current_price': stock_data.price,
                'expected_return': self._estimate_expected_return(composite_score),
                'confidence': min(composite_score, 0.95),
                'reasoning': self._generate_reasoning(symbol, technical_score, momentum_score, value_score)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing profitability for {symbol}: {e}")
            return None
    
    def _calculate_technical_score(self, hist_data: pd.DataFrame, stock_data) -> float:
        try:
            delta = hist_data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]
            
            if current_rsi < 30:
                rsi_score = 0.9
            elif current_rsi < 40:
                rsi_score = 0.7
            elif current_rsi > 70:
                rsi_score = 0.2
            else:
                rsi_score = 0.5
            
            sma_20 = hist_data['Close'].rolling(window=20).mean().iloc[-1]
            sma_50 = hist_data['Close'].rolling(window=50).mean().iloc[-1]
            current_price = stock_data.price
            
            if current_price > sma_20 > sma_50:
                ma_score = 0.8
            elif current_price < sma_20 < sma_50 and current_price < sma_50 * 0.95:
                ma_score = 0.7
            else:
                ma_score = 0.4
            
            recent_high = hist_data['High'].rolling(window=20).max().iloc[-1]
            recent_low = hist_data['Low'].rolling(window=20).min().iloc[-1]
            price_position = (current_price - recent_low) / (recent_high - recent_low)
            
            if price_position < 0.3:
                support_score = 0.8
            elif price_position > 0.8:
                support_score = 0.3
            else:
                support_score = 0.5
            
            return np.mean([rsi_score, ma_score, support_score])
        except Exception as e:
            logger.error(f"Error calculating technical score: {e}")
            return 0.5
    
    def _calculate_momentum_score(self, hist_data: pd.DataFrame) -> float:
        try:
            current_price = hist_data['Close'].iloc[-1]
            momentum_5d = (current_price / hist_data['Close'].iloc[-6] - 1) if len(hist_data) > 5 else 0
            momentum_20d = (current_price / hist_data['Close'].iloc[-21] - 1) if len(hist_data) > 20 else 0
            
            avg_volume_20d = hist_data['Volume'].tail(20).mean()
            recent_volume_5d = hist_data['Volume'].tail(5).mean()
            volume_momentum = recent_volume_5d / avg_volume_20d if avg_volume_20d > 0 else 1
            
            momentum_score = 0.5
            if momentum_5d > 0.02:
                momentum_score += 0.2
            elif momentum_5d < -0.05:
                momentum_score += 0.1
            
            if momentum_20d > 0.05:
                momentum_score += 0.15
            elif momentum_20d < -0.15:
                momentum_score += 0.1
            
            if volume_momentum > 1.5:
                momentum_score += 0.15
            
            return min(momentum_score, 1.0)
        except Exception as e:
            logger.error(f"Error calculating momentum score: {e}")
            return 0.5
    
    def _calculate_value_score(self, stock_data, hist_data: pd.DataFrame) -> float:
        try:
            current_price = stock_data.price
            high_52w = hist_data['High'].max()
            low_52w = hist_data['Low'].min()
            if high_52w > low_52w:
                price_position = (current_price - low_52w) / (high_52w - low_52w)
                if price_position < 0.3:
                    value_score = 0.8
                elif price_position > 0.8:
                    value_score = 0.2
                else:
                    value_score = 0.5
            else:
                value_score = 0.5
            
            if stock_data.market_cap and stock_data.market_cap < 100_000_000:
                value_score += 0.1
            
            return min(value_score, 1.0)
        except Exception as e:
            logger.error(f"Error calculating value score: {e}")
            return 0.5
    
    def _calculate_volume_score(self, hist_data: pd.DataFrame, stock_data) -> float:
        try:
            avg_volume_30d = hist_data['Volume'].tail(30).mean()
            current_volume = stock_data.volume
            
            volume_ratio = current_volume / avg_volume_30d if avg_volume_30d > 0 else 1
            
            if volume_ratio > 2.0:
                volume_score = 0.9
            elif volume_ratio > 1.5:
                volume_score = 0.7
            elif volume_ratio < 0.5:
                volume_score = 0.3
            else:
                volume_score = 0.5
            
            if avg_volume_30d < config.MIN_VOLUME:
                volume_score *= 0.5
            
            return volume_score
        except Exception as e:
            logger.error(f"Error calculating volume score: {e}")
            return 0.5
    
    def _calculate_volatility_score(self, hist_data: pd.DataFrame) -> float:
        try:
            returns = hist_data['Close'].pct_change().dropna()
            if len(returns) < 10:
                return 0.5
            volatility = returns.std()
            if 0.02 <= volatility <= 0.05:
                volatility_score = 0.8
            elif 0.01 <= volatility <= 0.07:
                volatility_score = 0.6
            else:
                volatility_score = 0.4
            return volatility_score
        except Exception as e:
            logger.error(f"Error calculating volatility score: {e}")
            return 0.5
    
    def _estimate_expected_return(self, composite_score: float) -> float:
        base_return = 0.05
        bonus_return = (composite_score - 0.5) * 0.4
        return max(0.02, base_return + bonus_return)
    
    def _generate_reasoning(self, symbol: str, technical_score: float, momentum_score: float, value_score: float) -> str:
        reasons = []
        if technical_score > 0.7:
            reasons.append("strong technical indicators")
        elif technical_score > 0.6:
            reasons.append("favorable technical setup")
        if momentum_score > 0.7:
            reasons.append("positive price momentum")
        elif momentum_score > 0.6:
            reasons.append("building momentum")
        if value_score > 0.7:
            reasons.append("attractive valuation opportunity")
        elif value_score > 0.6:
            reasons.append("reasonable value proposition")
        if not reasons:
            reasons.append("balanced risk-reward profile")
        return f"Selected {symbol} based on: " + ", ".join(reasons) + " with profit potential analysis."

class MLRecommendationEngine:
    def __init__(self):
        self.config = config
        self.db_manager = get_db_manager()
        self.data_manager = EnhancedDataManager()
        self.profitability_analyzer = ProfitabilityAnalyzer(self.data_manager)
        self.return_model = None

    def generate_recommendations(self, max_recommendations: int = None) -> List[Dict[str, Any]]:
        try:
            max_recs = max_recommendations or self.config.MAX_DAILY_RECOMMENDATIONS
            logger.info(f"Generating {max_recs} ML recommendations...")

            trade_history = self.db_manager.get_trade_history(days=365)
            if trade_history.empty or len(trade_history) < self.config.MIN_TRAINING_SAMPLES:
                logger.info("No sufficient trade history - generating profit-optimized initial recommendations.")
                return self._generate_profitable_initial_recommendations(max_recs)

            candidates = self._get_candidate_stocks()
            logger.info(f"[DEBUG ML] Candidate stocks count: {len(candidates)}")

            scored_candidates = []
            for symbol in candidates:
                try:
                    analysis = self.profitability_analyzer.analyze_profitability(symbol)
                    if analysis:
                        logger.info(f"[DEBUG ML] Profitability of {symbol}: Composite Score = {analysis['composite_score']:.3f}")
                        scored_candidates.append(analysis)
                    else:
                        logger.info(f"[DEBUG ML] Profitability analysis returned None for {symbol}")
                except Exception as e:
                    logger.error(f"[ERROR ML] Error scoring {symbol}: {e}")
                    continue

            logger.info(f"[DEBUG ML] Profitable candidates count: {len(scored_candidates)}")

            scored_candidates.sort(key=lambda x: x['composite_score'], reverse=True)
            top_recommendations = scored_candidates[:max_recs]

            recommendations = []
            for rec in top_recommendations:
                recommendations.append({
                    'symbol': rec['symbol'],
                    'action': 'BUY',
                    'confidence': rec['confidence'],
                    'reasoning': rec['reasoning'],
                    'price_target': rec['current_price'] * (1 + rec['expected_return']),
                    'expected_return': rec['expected_return'],
                    'risk_score': 1 - rec['composite_score'],
                    'ml_features': {
                        'technical_score': rec['technical_score'],
                        'momentum_score': rec['momentum_score'],
                        'value_score': rec['value_score']
                    },
                    'sentiment_score': 0.0
                })

            logger.info(f"Generated {len(recommendations)} ML recommendations.")
            return recommendations

        except Exception as e:
            logger.error(f"Error generating ML recommendations: {e}")
            return []

    def _generate_profitable_initial_recommendations(self, max_recs: int) -> List[Dict[str, Any]]:
        try:
            logger.info("Analyzing current market for most profitable opportunities...")

            candidates = self._get_candidate_stocks()
            if not candidates:
                logger.warning("No candidate stocks found")
                return []

            profitable_stocks = []
            for symbol in candidates:
                analysis = self.profitability_analyzer.analyze_profitability(symbol)
                if analysis:
                    profitable_stocks.append(analysis)

            if not profitable_stocks:
                logger.warning("No profitable stock analyses completed")
                return []

            profitable_stocks.sort(key=lambda x: x['composite_score'], reverse=True)
            top_profitable = profitable_stocks[:max_recs]

            recommendations = []
            for analysis in top_profitable:
                recommendations.append({
                    'symbol': analysis['symbol'],
                    'action': 'BUY',
                    'confidence': analysis['confidence'],
                    'reasoning': analysis['reasoning'],
                    'price_target': analysis['current_price'] * (1 + analysis['expected_return']),
                    'expected_return': analysis['expected_return'],
                    'risk_score': 1 - analysis['composite_score'],
                    'ml_features': {
                        'technical_score': analysis['technical_score'],
                        'momentum_score': analysis['momentum_score'],
                        'value_score': analysis['value_score'],
                        'volume_score': analysis['volume_score'],
                        'volatility_score': analysis['volatility_score']
                    },
                    'sentiment_score': 0.0
                })

            logger.info(f"Generated {len(recommendations)} profit-optimized initial recommendations")
            return recommendations

        except Exception as e:
            logger.error(f"Error generating profitable initial recommendations: {e}")
            return []

    def _get_candidate_stocks(self) -> List[str]:
        try:
            candidates = self.data_manager.fetch_etf_holdings()
            if not candidates:
                candidates = PRODUCTION_MICRO_CAPS

            filtered_candidates = []
            for symbol in candidates:
                stock_data = self.data_manager.get_stock_data(symbol, validate=False)
                if stock_data and stock_data.is_micro_cap and stock_data.has_sufficient_volume:
                    filtered_candidates.append(symbol)

            return filtered_candidates[:50]

        except Exception as e:
            logger.error(f"Error getting candidate stocks: {e}")
            return []

    def learn_from_trades(self, retrain: bool = False) -> None:
        logger.info("learn_from_trades called - no training implemented yet.")
        pass

    def get_model_info(self) -> Dict[str, Any]:
        return {
            'return_model_exists': self.return_model is not None,
            'models_directory': str(self.config.MODELS_DIR),
            'last_training_date': None,
            'profitability_analyzer_available': True
        }
