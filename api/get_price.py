
import sys
import json
import pandas as pd
from datetime import datetime, timedelta

try:
    from alpha_vantage.timeseries import TimeSeries
except ImportError:
    print(json.dumps({"error": "Alpha Vantage library not found. Please install it using: pip install alpha_vantage"}))
    sys.exit(1)

API_KEY = 'JJ2HHF8X7MY9DTRB'

def get_historical_prices_alpha_vantage(ticker, start_date, end_date):
    """
    Gets historical stock prices from Alpha Vantage.
    """
    try:
        ts = TimeSeries(key=API_KEY, output_format='pandas')
        data, meta_data = ts.get_daily_adjusted(symbol=ticker, outputsize='full')
        data = data.rename(columns={
            '1. open': 'Open',
            '2. high': 'High',
            '3. low': 'Low',
            '4. close': 'Close',
            '5. adjusted close': 'Adj. Close',
            '6. volume': 'Volume'
        })
        data.index = pd.to_datetime(data.index)
        data = data[(data.index >= pd.to_datetime(start_date)) & (data.index <= pd.to_datetime(end_date))]
        return data.sort_index(ascending=True)
    except Exception as e:
        print(f"Error fetching historical data from Alpha Vantage: {e}", file=sys.stderr)
        return pd.DataFrame()

def get_latest_price_alpha_vantage(ticker):
    """
    Gets the latest price for a ticker from Alpha Vantage.
    """
    try:
        ts = TimeSeries(key=API_KEY, output_format='pandas')
        data, meta_data = ts.get_quote_endpoint(symbol=ticker)
        if not data.empty:
            return {'latest_price': data['05. price'].iloc[0]}
    except Exception as e:
        print(f"Error fetching latest price from Alpha Vantage: {e}", file=sys.stderr)
    return None

def get_option_chain_alpha_vantage(ticker):
    """
    Gets option chain for a given ticker from Alpha Vantage.
    This is a placeholder as the free tier does not provide extensive option chain data.
    A paid plan is needed for more comprehensive data.
    """
    print("Warning: Option chain data from Alpha Vantage is limited on the free plan.", file=sys.stderr)
    return {}

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(json.dumps({"error": "Usage: get_price.py [historical|options|latest] <ticker>"}), file=sys.stderr)
        sys.exit(1)

    data_type, ticker = sys.argv[1], sys.argv[2]

    result = {}
    if data_type == "historical":
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=2*365)).strftime('%Y-%m-%d')
        df = get_historical_prices_alpha_vantage(ticker, start_date, end_date)
        result = df.to_json(orient='split') if not df.empty else json.dumps({"error": f"Could not retrieve historical data for {ticker}"})
    elif data_type == "options":
        chain = get_option_chain_alpha_vantage(ticker)
        result = json.dumps(chain) if chain else json.dumps({"error": f"Could not retrieve option chain for {ticker}"})
    elif data_type == "latest":
        price_info = get_latest_price_alpha_vantage(ticker)
        result = json.dumps(price_info) if price_info else json.dumps({"error": f"Could not retrieve latest price for {ticker}"})
    else:
        result = json.dumps({"error": f"Invalid data type: {data_type}"})

    print(result)
