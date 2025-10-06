"""
Configuration management for AI Trading System
Handles environment variables, API keys, and system settings
"""
import os
from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import BaseSettings, validator
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class TradingConfig(BaseSettings):
    """Main configuration class using Pydantic for validation"""

    # API Configuration
    PERPLEXITY_API_KEY: str
    PERPLEXITY_BASE_URL: str = "https://api.perplexity.ai"

    # Data Sources
    PRIMARY_DATA_SOURCE: str = "yahoo"
    BACKUP_DATA_SOURCES: list = ["alpha_vantage", "finnhub"]

    # Trading Parameters
    INITIAL_CAPITAL: float = 100.0
    MAX_POSITION_SIZE: float = 0.30  # 30% max per position
    STOP_LOSS_PERCENTAGE: float = 0.15  # 15% stop loss
    TAKE_PROFIT_PERCENTAGE: float = 0.25  # 25% take profit
    MAX_MARKET_CAP: int = 300_000_000  # $300M for micro-caps

    # Risk Management
    MAX_DAILY_LOSS: float = 0.05  # 5% max daily loss
    MAX_POSITIONS: int = 10
    MIN_VOLUME: int = 100_000  # Minimum daily volume

    # Database
    DATABASE_URL: str = "sqlite:///trading_system.db"

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/trading_system.log"

    # Market Hours (Eastern Time)
    MARKET_OPEN: str = "09:30"
    MARKET_CLOSE: str = "16:00"

    # Research Settings
    DEEP_RESEARCH_DAYS: list = [4]  # Friday = 4 (0 = Monday)
    MAX_RESEARCH_STOCKS: int = 50

    # File Paths
    DATA_DIR: Path = Path("data")
    LOGS_DIR: Path = Path("logs")
    REPORTS_DIR: Path = Path("reports")

    @validator("PERPLEXITY_API_KEY")
    def validate_api_key(cls, v):
        if not v:
            raise ValueError("PERPLEXITY_API_KEY is required")
        return v

    @validator("INITIAL_CAPITAL")
    def validate_capital(cls, v):
        if v <= 0:
            raise ValueError("INITIAL_CAPITAL must be positive")
        return v

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create directories if they don't exist
        for directory in [self.DATA_DIR, self.LOGS_DIR, self.REPORTS_DIR]:
            directory.mkdir(exist_ok=True, parents=True)

    class Config:
        env_file = ".env"
        case_sensitive = True

# Global config instance
config = TradingConfig()

def get_config() -> TradingConfig:
    """Get the global configuration instance"""
    return config

def update_config(**kwargs) -> None:
    """Update configuration values"""
    global config
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown configuration key: {key}")
