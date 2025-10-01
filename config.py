"""
Configuration management for AI Trading System with ML
"""
import os
from pathlib import Path
from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import validator
from dotenv import load_dotenv

load_dotenv()

class TradingConfig(BaseSettings):
    """Main configuration class using Pydantic for validation"""

    # API Configuration
    PERPLEXITY_API_KEY: str
    PERPLEXITY_BASE_URL: str = "https://api.perplexity.ai"

    # Optional API keys for enhanced data
    ALPHA_VANTAGE_KEY: Optional[str] = None
    FRED_API_KEY: Optional[str] = None

    # Data Sources
    PRIMARY_DATA_SOURCE: str = "yahoo"
    BACKUP_DATA_SOURCES: List[str] = ["alpha_vantage", "finnhub"]

    # Trading Parameters
    INITIAL_CAPITAL: float = 100.0
    MAX_POSITION_SIZE: float = 0.30  # 30% max per position
    STOP_LOSS_PERCENTAGE: float = 0.15  # 15% stop loss
    TAKE_PROFIT_PERCENTAGE: float = 0.25  # 25% take profit
    MAX_MARKET_CAP: int = 300_000_000_000_000  # $300M for micro-caps

    # Risk Management
    MAX_DAILY_LOSS: float = 0.05  # 5% max daily loss
    MAX_POSITIONS: int = 10
    MIN_VOLUME: int = 100_000  # Minimum daily volume

    # ML Configuration
    MAX_DAILY_RECOMMENDATIONS: int = 5
    ML_RETRAIN_FREQUENCY: int = 7  # Retrain ML model every 7 days
    FEATURE_WINDOW: int = 30  # Look back 30 days for features
    MIN_TRAINING_SAMPLES: int = 50  # Minimum samples before ML training

    # News and Sentiment
    NEWS_LOOKBACK_DAYS: int = 7
    SENTIMENT_THRESHOLD: float = 0.1  # Minimum sentiment score for consideration

    # Database
    DATABASE_URL: str = "sqlite:///trading_system.db"

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/trading_system.log"

    # Market Hours (Eastern Time)
    MARKET_OPEN: str = "09:30"
    MARKET_CLOSE: str = "16:00"

    # Research Settings
    DEEP_RESEARCH_DAYS: List[int] = [5, 6]  # Saturday = 5, Sunday = 6 (0 = Monday)
    MAX_RESEARCH_STOCKS: int = 50

    # Weekend Analysis
    WEEKEND_ANALYSIS_ENABLED: bool = True
    WEEKEND_PORTFOLIO_REVIEW: bool = True
    WEEKEND_MARKET_ANALYSIS: bool = True

    # File Paths
    DATA_DIR: Path = Path("data")
    LOGS_DIR: Path = Path("logs")
    REPORTS_DIR: Path = Path("reports")
    MODELS_DIR: Path = Path("models")

    @validator("PERPLEXITY_API_KEY")
    def validate_api_key(cls, v):
        if not v or v == "your_perplexity_api_key_here":
            raise ValueError("PERPLEXITY_API_KEY is required - please set it in your .env file")
        return v

    @validator("INITIAL_CAPITAL")
    def validate_capital(cls, v):
        if v <= 0:
            raise ValueError("INITIAL_CAPITAL must be positive")
        return v

    @validator("MAX_DAILY_RECOMMENDATIONS")
    def validate_recommendations(cls, v):
        if v < 1 or v > 10:
            raise ValueError("MAX_DAILY_RECOMMENDATIONS must be between 1 and 10")
        return v

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create directories if they don't exist
        for directory in [self.DATA_DIR, self.LOGS_DIR, self.REPORTS_DIR, self.MODELS_DIR]:
            directory.mkdir(exist_ok=True, parents=True)

    class Config:
        env_file = ".env"
        case_sensitive = True

# Global config instance
config = None

def get_config() -> TradingConfig:
    """Get the global configuration instance"""
    global config
    if config is None:
        try:
            config = TradingConfig()
        except ValueError as e:
            print(f"Configuration Error: {e}")
            print("Please check your .env file and ensure PERPLEXITY_API_KEY is set")
            raise
    return config

def update_config(**kwargs) -> None:
    """Update configuration values"""
    global config
    if config is None:
        config = get_config()

    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown configuration key: {key}")
