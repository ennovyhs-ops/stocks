# Daily Option Strategy Analyzer

This project is a comprehensive tool for analyzing stock and option data to generate daily trading strategy recommendations. It fetches historical stock data, calculates a variety of technical indicators, and uses this information to score potential trading opportunities and suggest option strike prices.

## Key Features

- **Data Fetching:** Fetches 2 years of historical daily OHLCV (Open, High, Low, Close, Volume) real live data for up to 3 stocks and 1 benchmark index from Yahoo Finance (https://finance.yahoo.com/quote/...), selected stocks and index to be input by user at interface page.
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
- **Option Strike Recommendations:** Recommends option strike prices using a delta-based Black-Scholes inversion model for each of the timeframes. Recommendations should be clear and explicit on option play, namely call/put, long or short. Recommedations should be according to the day of request. If recommendations is dependent on an event or price yet to happen, earmark to wait for current price or premium to reach certain levels. 
- **Output Generation:** Display the analysis results below entered input on the same webpage interface with plots of the technical analysis, and summary of the scores and strike recommendations.
- **Output Storage:** It is unnecesary to record and store output data.

## Technologies Used

- **Python:** The core programming language for the project.
- **yfinance:** A Python library for fetching historical market data from Yahoo Finance.
- **pandas:** A powerful library for data manipulation and analysis.
- **numpy:** A fundamental package for scientific computing with Python.
- **scipy:** A Python-based ecosystem of open-source software for mathematics, science, and engineering.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ennovyhs-ops/stocks.git
   cd stocks
   ```

## Usage

You can run the analysis in by web interface 

1. **Open your web browser** and navigate to `hstocks-eight-tau.vercel.app` to view the analysis results.

This project is deployed on Vercel. You can access the live application here: [stocks-eight-tau.vercel.app](https://stocks-eight-tau.vercel.app)
