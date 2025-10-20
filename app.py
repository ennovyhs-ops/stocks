
import json
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from auto_analysis import analyze_stocks
import os

app = Flask(__name__)

# Ensure the output directory exists
if not os.path.exists('/tmp/output'):
    os.makedirs('/tmp/output')

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
        summary = analyze_stocks(ticker_list, benchmark)
        # Assuming the first ticker is the one we want to show the plot for
        main_ticker = ticker_list[0]
        plot_file = summary.get(main_ticker, {}).get('plot')
        
        return render_template('results.html', summary=json.dumps(summary, indent=2, default=float), plot_file=plot_file)
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/tmp/output/<filename>')
def output_file(filename):
    return send_from_directory('/tmp/output', filename)

if __name__ == "__main__":
    app.run(debug=True)

