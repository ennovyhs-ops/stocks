
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
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

## 🏃‍♀️ How to Run

There are two ways to run the analysis:

### 1. Web Interface

Run the Flask web server:

```bash
flask run
# or
python app.py
```

Then, open your web browser to `http://127.0.0.1:5000` to view the analysis results.

### 2. Command Line

Run the analysis script directly from the command line:

```bash
./run_analysis.sh
# or
python auto_analysis.py TICKER1,TICKER2 BENCHMARK
```

Example:
```bash
python auto_analysis.py 9988.HK,0005.HK ^HSI
```

## 📁 Folder Structure

```
├── app.py                    # Flask web server
├── auto_analysis.py          # Main Python script with full logic
├── run_analysis.sh           # Shell script to automate daily execution
├── requirements.txt          # Python dependencies
├── vercel.json               # Vercel deployment configuration
├── .vercelignore             # Vercel ignore file
├── README.md                 # Project overview and setup instructions
├── templates/                # HTML templates for the web interface
│   ├── index.html
│   └── error.html
└── static/
    └── output/               # Folder for plots and JSON summaries (auto-created)
```

## ⚙️ Configuration

To change the stock tickers and benchmark index, edit the `config.py` file:

```python
# List your tickers here
STOCK_TICKERS = ["AAPL", "GOOGL"]
BENCHMARK_INDEX = "^GSPC"
```
