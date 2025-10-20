
import json
from auto_analysis import analyze_stocks

def main():
    """
    This script provides an interactive way to run the stock analysis.
    It prompts the user for stock tickers and a benchmark index,
    then runs the analysis and prints the results.
    """
    try:
        tickers_str = input("Enter stock tickers separated by commas (e.g., 9988.HK,0005.HK): ")
        if not tickers_str:
            print("No tickers provided. Exiting.")
            return

        tickers = [t.strip() for t in tickers_str.split(',')]

        benchmark_index = input("Enter benchmark index (e.g., ^HSI): ")
        if not benchmark_index:
            print("No benchmark index provided. Exiting.")
            return

        print(f"Running analysis for tickers: {tickers} and benchmark: {benchmark_index}")
        summary = analyze_stocks(tickers, benchmark_index)
        print("\nAnalysis complete. Summary:")
        print(json.dumps(summary, indent=2, default=float))

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
