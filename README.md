# stocks


# 🧠 Daily Option Strategy Analyzer

This project fetches 5 years of historical stock data from Yahoo Finance, computes technical indicators, scores directional signals across multiple timeframes, and recommends option strike prices using delta-based Black-Scholes inversion.

## 📈 What It Does

- Fetches 5-year daily OHLCV data for up to 3 stocks and 1 benchmark index
- Computes:
  - VWMA20, VWMA50
  - Bollinger Bands (20, 2)
  - RSI(14), MACD(12,26,9)
  - VWAP20, Vol20
- Calculates composite scores for:
  - Next Day
  - 2 Weeks
  - 3 Months
  - 6 Months
- Recommends option strikes using delta inversion
- Saves:
  - CSVs of raw data
  - Annotated PNG plots
  - JSON summary of scores and strike recommendations

## 🛠️ Setup

### 1. Clone the repo
```bash
git clone https://github.com/your-username/daily-option-analysis.git
cd daily-option-analysis


## folder structure
├── auto_analysis.py          # Main Python script with full logic
├── run_analysis.sh           # Shell script to automate daily execution
├── requirements.txt          # Python dependencies
├── README.md                 # Project overview and setup instructions
├── data/                     # Folder for raw CSVs (auto-created)
└── output/                   # Folder for plots and JSON summaries (auto-created)