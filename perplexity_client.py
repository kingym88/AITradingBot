"""
Perplexity API Client for AI Trading System
Replaces ChatGPT with Perplexity for better real-time financial data access
"""
import httpx
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import json
from loguru import logger
from config import get_config

config = get_config()

@dataclass
class TradingRecommendation:
    """Data class for trading recommendations"""
    action: str  # 'BUY', 'SELL', 'HOLD'
    symbol: str
    reasoning: str
    confidence: float  # 0.0 to 1.0
    price_target: Optional[float] = None
    stop_loss: Optional[float] = None
    position_size: Optional[float] = None
    time_horizon: Optional[str] = None  # 'short', 'medium', 'long'
    risk_level: Optional[str] = None    # 'low', 'medium', 'high'
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class PerplexityClient:
    """Enhanced Perplexity API client for financial analysis"""

    def __init__(self):
        self.api_key = config.PERPLEXITY_API_KEY
        self.base_url = config.PERPLEXITY_BASE_URL
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.client = httpx.AsyncClient(timeout=30.0)

    async def get_trading_recommendation(
        self, 
        portfolio_data: Dict[str, Any],
        market_data: Dict[str, Any],
        research_context: Optional[str] = None
    ) -> List[TradingRecommendation]:
        """
        Get trading recommendations from Perplexity

        Args:
            portfolio_data: Current portfolio information
            market_data: Current market data
            research_context: Additional research context

        Returns:
            List of trading recommendations
        """
        try:
            prompt = self._build_trading_prompt(portfolio_data, market_data, research_context)

            response = await self._make_api_request(prompt)
            recommendations = self._parse_recommendations(response)

            return recommendations

        except Exception as e:
            logger.error(f"Error getting trading recommendation: {e}")
            return []

    def _build_trading_prompt(
        self, 
        portfolio_data: Dict[str, Any],
        market_data: Dict[str, Any],
        research_context: Optional[str] = None
    ) -> str:
        """Build comprehensive trading prompt"""

        prompt = f"""
        You are a professional portfolio strategist managing a micro-cap stock portfolio.

        CURRENT PORTFOLIO STATUS:
        - Total Value: ${portfolio_data.get('total_value', 0):.2f}
        - Cash Available: ${portfolio_data.get('cash', 0):.2f}
        - Number of Positions: {portfolio_data.get('num_positions', 0)}
        - Current Holdings: {portfolio_data.get('holdings', {})}
        - Daily P&L: {portfolio_data.get('daily_pnl', 0):.2f}%
        - Total Return: {portfolio_data.get('total_return', 0):.2f}%

        MARKET CONDITIONS:
        - Market Status: {'Open' if market_data.get('market_open', False) else 'Closed'}
        - S&P 500 Change: {market_data.get('sp500_change', 0):.2f}%
        - VIX Level: {market_data.get('vix', 'N/A')}
        - Market Sentiment: {market_data.get('sentiment', 'Neutral')}

        INVESTMENT CONSTRAINTS:
        - Only US-listed micro-cap stocks (market cap < $300M)
        - Maximum position size: 30% of portfolio
        - Stop-loss at 15% loss
        - Take profit at 25% gain
        - Minimum daily volume: 100,000 shares

        CURRENT STOCK CANDIDATES:
        {self._format_stock_candidates(market_data.get('candidates', []))}

        {f"ADDITIONAL RESEARCH CONTEXT: {research_context}" if research_context else ""}

        TASK:
        Analyze the current portfolio and market conditions. Provide specific trading recommendations 
        in JSON format with the following structure for each recommendation:

        {{
            "action": "BUY|SELL|HOLD",
            "symbol": "STOCK_SYMBOL",
            "reasoning": "Detailed reasoning for the recommendation",
            "confidence": 0.85,
            "price_target": 25.50,
            "stop_loss": 20.00,
            "position_size": 0.15,
            "time_horizon": "short|medium|long",
            "risk_level": "low|medium|high"
        }}

        Focus on:
        1. Stocks with strong fundamentals and upcoming catalysts
        2. Technical analysis and momentum indicators
        3. Risk management and position sizing
        4. Market timing considerations
        5. Sector diversification

        Provide 1-5 recommendations based on current opportunities.
        """

        return prompt.strip()

    def _format_stock_candidates(self, candidates: List[Dict[str, Any]]) -> str:
        """Format stock candidates for the prompt"""
        if not candidates:
            return "No specific candidates provided - please research and suggest micro-cap opportunities."

        formatted = []
        for stock in candidates:
            formatted.append(
                f"- {stock.get('symbol', 'N/A')}: "
                f"Price ${stock.get('price', 0):.2f}, "
                f"Volume {stock.get('volume', 0):,}, "
                f"Change {stock.get('change_percent', 0):.2f}%"
            )

        return "\n".join(formatted)

    async def _make_api_request(self, prompt: str) -> str:
        """Make API request to Perplexity"""
        try:
            payload = {
                "model": "llama-3.1-sonar-large-128k-online",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a professional financial analyst and portfolio manager specializing in micro-cap stocks. Provide accurate, data-driven investment recommendations."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 2000,
                "search_domain_filter": ["finance.yahoo.com", "seekingalpha.com", "marketwatch.com", "bloomberg.com"],
                "search_recency_filter": "day"
            }

            response = await self.client.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload
            )

            if response.status_code != 200:
                logger.error(f"API request failed: {response.status_code} - {response.text}")
                return ""

            result = response.json()
            return result.get("choices", [{}])[0].get("message", {}).get("content", "")

        except Exception as e:
            logger.error(f"Error making API request: {e}")
            return ""

    def _parse_recommendations(self, response: str) -> List[TradingRecommendation]:
        """Parse recommendations from API response"""
        recommendations = []

        try:
            # Try to find JSON blocks in the response
            import re
            json_pattern = r'\{[^{}]*"action"[^{}]*\}'
            matches = re.findall(json_pattern, response, re.DOTALL)

            for match in matches:
                try:
                    data = json.loads(match)
                    recommendation = TradingRecommendation(
                        action=data.get("action", "HOLD").upper(),
                        symbol=data.get("symbol", "").upper(),
                        reasoning=data.get("reasoning", ""),
                        confidence=float(data.get("confidence", 0.5)),
                        price_target=data.get("price_target"),
                        stop_loss=data.get("stop_loss"),
                        position_size=data.get("position_size"),
                        time_horizon=data.get("time_horizon"),
                        risk_level=data.get("risk_level")
                    )
                    recommendations.append(recommendation)

                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    logger.warning(f"Failed to parse recommendation JSON: {e}")
                    continue

            # If no JSON found, try to parse from text
            if not recommendations:
                recommendations = self._parse_text_recommendations(response)

        except Exception as e:
            logger.error(f"Error parsing recommendations: {e}")

        return recommendations

    def _parse_text_recommendations(self, response: str) -> List[TradingRecommendation]:
        """Fallback parser for text-based recommendations"""
        recommendations = []

        # This is a basic text parser - in production you'd want more sophisticated NLP
        lines = response.split('\n')
        current_rec = {}

        for line in lines:
            line = line.strip()
            if any(action in line.upper() for action in ['BUY', 'SELL', 'HOLD']):
                if current_rec:
                    # Save previous recommendation
                    try:
                        rec = TradingRecommendation(
                            action=current_rec.get('action', 'HOLD'),
                            symbol=current_rec.get('symbol', ''),
                            reasoning=current_rec.get('reasoning', ''),
                            confidence=0.5
                        )
                        recommendations.append(rec)
                    except:
                        pass

                # Start new recommendation
                current_rec = {'reasoning': line}

                # Extract action and symbol
                for action in ['BUY', 'SELL', 'HOLD']:
                    if action in line.upper():
                        current_rec['action'] = action
                        # Try to extract symbol
                        words = line.split()
                        for word in words:
                            if word.isalpha() and len(word) <= 5 and word.isupper():
                                current_rec['symbol'] = word
                                break
                        break

        return recommendations

    async def get_market_research(self, symbols: List[str]) -> str:
        """Get comprehensive market research for given symbols"""
        try:
            symbols_str = ", ".join(symbols)
            prompt = f"""
            Provide comprehensive research analysis for these micro-cap stocks: {symbols_str}

            For each stock, include:
            1. Recent news and developments
            2. Financial health and key metrics
            3. Growth catalysts and risks
            4. Technical analysis
            5. Analyst opinions and price targets
            6. Industry trends affecting the company

            Focus on recent developments (last 30 days) and upcoming catalysts.
            Provide a balanced view including both opportunities and risks.
            """

            response = await self._make_api_request(prompt)
            return response

        except Exception as e:
            logger.error(f"Error getting market research: {e}")
            return ""

    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()

# Synchronous wrapper for easier usage
class PerplexityClientSync:
    """Synchronous wrapper for PerplexityClient"""

    def __init__(self):
        self.async_client = PerplexityClient()

    def get_trading_recommendation(
        self, 
        portfolio_data: Dict[str, Any],
        market_data: Dict[str, Any],
        research_context: Optional[str] = None
    ) -> List[TradingRecommendation]:
        """Synchronous wrapper for get_trading_recommendation"""
        return asyncio.run(
            self.async_client.get_trading_recommendation(
                portfolio_data, market_data, research_context
            )
        )

    def get_market_research(self, symbols: List[str]) -> str:
        """Synchronous wrapper for get_market_research"""
        return asyncio.run(self.async_client.get_market_research(symbols))

    def __del__(self):
        try:
            asyncio.run(self.async_client.close())
        except:
            pass
