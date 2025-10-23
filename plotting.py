import io
import base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot_base64(df, ticker):
    fig, ax = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [3, 1]})
    ax0, ax1 = ax
    ax0.plot(df['Date'], df['Close'], color='black', linewidth=1)
    if 'VWMA50' in df: ax0.plot(df['Date'], df['VWMA50'], color='blue', linewidth=0.8)
    if 'VWMA20' in df: ax0.plot(df['Date'], df['VWMA20'], color='cyan', linewidth=0.7)
    if 'BB_upper' in df and 'BB_lower' in df:
        ax0.fill_between(df['Date'], df['BB_lower'], df['BB_upper'], color='gray', alpha=0.12)
    ax0.set_title(f"{ticker} Close and VWMA/Bollinger")
    if 'RSI' in df:
        ax1.plot(df['Date'], df['RSI'], color='purple', linewidth=0.9)
        ax1.axhline(70, color='red', linestyle='--'); ax1.axhline(30, color='green', linestyle='--')
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode('ascii')
    return f"data:image/png;base64,{data}"
