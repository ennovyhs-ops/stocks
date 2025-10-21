
import json
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from auto_analysis import analyze_stocks
import os

app = Flask(__name__)

# Ensure the output directory exists
if not os.path.exists('static/output'):
    os.makedirs('static/output')

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
        
        # Read the plotly HTML file content
        if plot_file and os.path.exists(plot_file):
            with open(plot_file, 'r') as f:
                plot_html = f.read()
        else:
            plot_html = None
        
        return render_template('results.html', 
                             summary=json.dumps(summary, indent=2, default=float),
                             plot_html=plot_html)
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/static/output/<filename>')
def output_file(filename):
    return send_from_directory('static/output', filename)

def create_app():
    # Factory for WSGI servers if needed in future
    return app

if __name__ == "__main__":
    # Local development server
    app.run(debug=True)
