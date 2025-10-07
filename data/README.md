# Data Storage Directory

This directory stores cached data files and ETF holdings information.

## Files Stored Here:
- `etf_holdings_microcap.csv` - Cached iShares Micro-Cap ETF holdings
- Stock data cache files (temporary, auto-cleaned)
- Trade export files when requested
- Market data snapshots

## ETF Holdings Cache:
- Downloaded automatically from iShares IWC ETF
- Updated each time system runs
- Contains 200+ current micro-cap stocks
- Used as primary stock universe for analysis

## Data Refresh:
- ETF holdings: Downloaded fresh each session
- Stock prices: Cached for 5 minutes
- Market data: Real-time via Yahoo Finance
