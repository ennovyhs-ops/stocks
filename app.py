import json
import asyncio
from flask import Flask, render_template, request, redirect, url_for
from auto_analysis import analyze_stocks

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    tickers = request.form.get('tickers')
    benchmark = request.form.get('benchmark')

    if not tickers or not benchmark:
        return redirect(url_for('index'))

    ticker_list = [t.strip() for t in tickers.split(',')]
    
    try:
        # Since analyze_stocks is now async, we need to run it in an event loop
        summary = asyncio.run(analyze_stocks(ticker_list, benchmark))
        # Assuming the first ticker is the one we want to show the plot for
        main_ticker = ticker_list[0]
        plot_html = summary.get(main_ticker, {}).get('plot')
        
        return render_template('results.html', 
                             summary=json.dumps(summary, indent=2, default=float),
                             plot_html=plot_html)
    except Exception as e:
        return render_template('error.html', error=str(e))

def create_app():
    # Factory for WSGI servers if needed in future
    return app

if __name__ == "__main__":
    # Local development server
    app.run(debug=True)
