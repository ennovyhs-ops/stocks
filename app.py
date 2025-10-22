import json
import asyncio
from flask import Flask, render_template, request
from auto_analysis import analyze_stocks

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        tickers = request.form.get('tickers')
        benchmark = request.form.get('benchmark')

        if not tickers or not benchmark:
            return render_template('index.html', error="Please provide tickers and a benchmark.")

        ticker_list = [t.strip() for t in tickers.split(',')]
        if len(ticker_list) > 3:
            return render_template('index.html', error="You can analyze up to 3 stocks at a time.", tickers=tickers, benchmark=benchmark)

        try:
            summary = asyncio.run(analyze_stocks(ticker_list, benchmark))
            
            rendered_summary = {}
            for ticker, analysis in summary.items():
                plot_html = analysis.pop('plot', '')
                rendered_summary[ticker] = {
                    'summary': json.dumps(analysis, indent=2, default=float),
                    'plot': plot_html
                }

            return render_template('index.html', 
                                 results=rendered_summary,
                                 tickers=tickers,
                                 benchmark=benchmark)
        except Exception as e:
            return render_template('error.html', error=str(e))
    
    return render_template('index.html')

def create_app():
    # Factory for WSGI servers if needed in future
    return app

if __name__ == "__main__":
    # Local development server
    app.run(debug=True)