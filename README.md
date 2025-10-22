# Daily Option Strategy Analyzer

This project is a comprehensive tool for analyzing stock and option data to generate daily trading strategy recommendations. It fetches historical stock data, calculates a variety of technical indicators, and uses this information to score potential trading opportunities and suggest option strike prices.

## Key Features

- **Data Fetching:** Fetches 2 years of historical daily OHLCV (Open, High, Low, Close, Volume) data for up to 3 stocks and 1 benchmark index from Yahoo Finance, to be input by user at interface page.
- **Technical Analysis:** Computes a wide range of technical indicators, including:
    - VWMA (Volume Weighted Moving Average) (20 and 50-day)
    - Bollinger Bands (20-day with 2 standard deviations)
    - RSI (Relative Strength Index) (14-day)
    - MACD (Moving Average Convergence Divergence) (12, 26, 9-day)
    - VWAP (Volume Weighted Average Price) (20-day)
    - 20-day Average Volume
- **Multi-Timeframe Scoring:** Calculates composite scores for various timeframes:
    - Next Day
    - 2 Weeks
    - 3 Months
    - 6 Months
- **Option Strike Recommendations:** Recommends option strike prices using a delta-based Black-Scholes inversion model for each of the timeframes. Recommendations should be clear and explicit on option play, namely call/put, long or short. If recommendations is dependent on an event or price to be happening, earmark to wait for current price or premium to reach certain levels. 
- **Output Generation:** Display the analysis results below entered input with plots of the technical analysis, and summary of the scores and strike recommendations.

## Technologies Used

- **Python:** The core programming language for the project.
- **Flask:** A lightweight web framework for creating the web interface.
- **yfinance:** A Python library for fetching historical market data from Yahoo Finance.
- **pandas:** A powerful library for data manipulation and analysis.
- **polars:** A fast DataFrame library for in-memory analytics.
- **numpy:** A fundamental package for scientific computing with Python.
- **scipy:** A Python-based ecosystem of open-source software for mathematics, science, and engineering.
- **plotly:** A graphing library for creating interactive, publication-quality graphs.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ennovyhs-ops/stocks.git
   cd stocks
   ```
2. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

You can run the analysis in two ways:

### Web Interface

1. **Run the Flask web server:**
   ```bash
   flask run
   ```
   or
   ```bash
   python app.py
   ```
2. **Open your web browser** and navigate to `http://127.0.0.1:5000` to view the analysis results.

### Command Line

1. **Run the analysis script directly from the command line:**
   ```bash
   ./run_analysis.sh
   ```
   or
   ```bash
   python auto_analysis.py TICKER1,TICKER2 BENCHMARK
   ```
   For example:
   ```bash
   python auto_analysis.py 9988.HK,0005.HK ^HSI
   ```

## Deployment

This project is deployed on Vercel. You can access the live application here: [stocks-eight-tau.vercel.app](https://stocks-eight-tau.vercel.app)

## Project Structure

```
.
├── app.py                  # Flask web server for the web interface
├── auto_analysis.py        # Main Python script with the full analysis logic
├── config.py               # Configuration file for stock tickers and benchmark index
├── interactive_runner.py   # Interactive runner for the analysis
├── requirements.txt        # Python dependencies for the project
├── run_analysis.sh         # Shell script to automate the daily execution of the analysis
├── templates
│   ├── error.html          # HTML template for displaying errors
│   ├── index.html          # HTML template for the main page
│   └── results.html        # HTML template for displaying the analysis results
└── README.md               # This file
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
