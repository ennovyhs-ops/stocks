import yfinance as yf
from config import STOCK_TICKERS, BENCHMARK_INDEX
from analysis_module import run_analysis

def fetch_data(ticker):
    df = yf.Ticker(ticker).history(period="5y", interval="1d", auto_adjust=False)
    df = df.reset_index()
    df.to_csv(f"data/{ticker.replace('^','_')}_5y.csv", index=False)
    return df

def main():
    index_df = fetch_data(BENCHMARK_INDEX)
    for ticker in STOCK_TICKERS:
        stock_df = fetch_data(ticker)
        run_analysis(stock_df, index_df, ticker)

if __name__ == "__main__":
    main()
