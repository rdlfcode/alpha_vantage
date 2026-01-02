import duckdb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import threading
import queue
from lightweight_charts import Chart
from datetime import timedelta

# Set aesthetic style for Seaborn
sns.set_theme(style="darkgrid", palette="muted")

def get_db_connection(db_path='data/alpha_vantage.db'):
    """Establishes a connection to the DuckDB database."""
    return duckdb.connect(db_path)

def load_ohlcv(conn, symbol):
    """
    Loads OHLCV data for a specific symbol from the TIME_SERIES_DAILY_ADJUSTED table.
    Fetches up to 5 years to satisfy the 4-year history requirement.
    """
    query = f"""
        SELECT 
            dt as time, 
            open, 
            high, 
            low, 
            close, 
            volume 
        FROM TIME_SERIES_DAILY_ADJUSTED
        WHERE symbol = '{symbol}' 
        ORDER BY time
    """
    try:
        df = conn.execute(query).df()
        return df
    except Exception as e:
        print(f"Error loading OHLCV: {e}")
        return pd.DataFrame()

def load_options(db_path, symbol, result_queue):
    """
    Loads historical options data for a specific symbol.
    Designed to be run in a separate thread.
    """
    try:
        # Create a new connection for the thread
        conn = duckdb.connect(db_path)
        # We process 'future' expirations relative to the dataset's notion of 'now'.
        # For a full landscape, we just load everything and filter in python.
        query = f"""
            SELECT expiration, strike, type, openInterest, dt, volume
            FROM HISTORICAL_OPTIONS 
            WHERE symbol = '{symbol}'
        """
        df = conn.execute(query).df()
        conn.close()
        result_queue.put(df)
    except Exception as e:
        print(f"Error loading Options in background: {e}")
        result_queue.put(pd.DataFrame())

def plot_interactive_ohlcv(df, symbol):
    """
    Renders the interactive OHLCV chart.
    """
    if df.empty:
        print(f"No OHLCV data found for {symbol}")
        return

    print(f"Displaying Interactive Chart for {symbol}. Close the window to proceed to prediction analysis.")
    chart = Chart()
    chart.legend(visible=True)
    chart.topbar.textbox('symbol', symbol)
    
    # Lightweight charts expects specific columns
    chart_data = df[['time', 'open', 'high', 'low', 'close', 'volume']].copy()
    chart.set(chart_data)
    chart.show(block=True) 

def plot_prediction_heatmap(ohlcv_df, options_df, symbol):
    """
    Plots 4 years of OHLCV history and a 1-year future prediction heatmap based on Options Open Interest.
    
    The Heatmap:
    - X-Axis: Time (Past 4 years + 1 year Future)
    - Y-Axis: Price / Strike
    - Color: Open Interest Intensity (for future expirations)
    - Line: Historic Close Price
    """
    if ohlcv_df.empty or options_df.empty:
        print("Insufficient data for prediction heatmap.")
        return

    # 1. Setup Data Ranges
    current_date = ohlcv_df['time'].max()
    start_date_history = current_date - timedelta(days=4*365) # 4 Years back
    end_date_prediction = current_date + timedelta(days=365)  # 1 Year forward

    # Filter History
    history_df = ohlcv_df[ohlcv_df['time'] >= start_date_history].copy()
    
    # Filter Options for Future Landscape
    # We want valid options that expire in the future (relative to the last known date)
    options_df['expiration'] = pd.to_datetime(options_df['expiration'])
    future_options = options_df[
        (options_df['expiration'] > current_date) & 
        (options_df['expiration'] <= end_date_prediction)
    ].copy()

    if future_options.empty:
        print("No future options data available for prediction.")
        return

    # 2. Calculate Prediction Line (Weighted Avg Strike per Expiration)
    # Group by expiration to get a single 'target price' for each future date
    def get_weighted_strike(group):
        total_oi = group['openInterest'].sum()
        if total_oi == 0: return None
        return (group['strike'] * group['openInterest']).sum() / total_oi

    predictions = future_options.groupby('expiration').apply(get_weighted_strike).reset_index()
    predictions.columns = ['expiration', 'predicted_price']
    predictions = predictions.dropna().sort_values('expiration')
    
    # Prepend the last historical point to connect the line seamlessly
    last_point = pd.DataFrame({'expiration': [current_date], 'predicted_price': [history_df['close'].iloc[-1]]})
    predictions = pd.concat([last_point, predictions])

    # 3. Plotting
    fig, ax = plt.subplots(figsize=(14, 8))

    # A. Historical Price Line
    ax.plot(history_df['time'], history_df['close'], label='Historical Close (4yr)', color='blue', linewidth=2)

    # B. Options Heatmap (Scatter)
    # We plot every strike/expiration tuple weighted by OI. 
    # To reduce points, filter low OI.
    threshold = future_options['openInterest'].quantile(0.50) # Show top 50%
    heatmap_data = future_options[future_options['openInterest'] > threshold]
    
    # Normalize OI for sizing/alpha
    max_oi = heatmap_data['openInterest'].max()
    sizes = (heatmap_data['openInterest'] / max_oi) * 100 
    
    sc = ax.scatter(
        heatmap_data['expiration'], 
        heatmap_data['strike'], 
        c=heatmap_data['openInterest'], 
        s=sizes, 
        cmap='inferno_r', 
        alpha=0.6,
        label='Options OI Intensity'
    )
    plt.colorbar(sc, label='Open Interest')

    # C. Prediction Line
    ax.plot(predictions['expiration'], predictions['predicted_price'], label='Market Implied Forecast', color='lime', linestyle='--', linewidth=2.5, marker='o', markersize=4)

    # Formatting
    ax.set_title(f"{symbol} 4-Year History & 1-Year Option-Implied Prediction", fontsize=16)
    ax.set_ylabel("Price / Strike")
    ax.set_xlabel("Date")
    
    # Date Formatting
    ax.set_xlim(start_date_history, end_date_prediction)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

def main():
    symbol = 'GOOG' 
    db_path = 'data/alpha_vantage.db'

    print(f"Processing {symbol}...")
    
    # 1. Start Async Options Load
    options_queue = queue.Queue()
    # We pass db_path strings because DuckDB connections aren't thread-safe across threads in the same way 
    # checking docs: DuckDB allows cursor sharing but safer to connect per thread for read-only.
    loader_thread = threading.Thread(target=load_options, args=(db_path, symbol, options_queue))
    loader_thread.start()
    print("Started background loading of Options data...")

    # 2. Main Thread: Load and Show OHLCV
    try:
        conn = get_db_connection(db_path)
        ohlcv_df = load_ohlcv(conn, symbol)
        conn.close()
        
      #   if not ohlcv_df.empty:
      #       plot_interactive_ohlcv(ohlcv_df, symbol)
      #   else:
      #       print("No OHLCV data found.")
            
    except Exception as e:
        print(f"Error in main thread: {e}")

    # 3. Wait for Options Data
    print("Waiting for background data to finish loading...")
    loader_thread.join()
    options_df = options_queue.get()
    
    if options_df.empty:
        print("Failed to load options data or no data found.")
        return

    # 4. Show Consolidated Heatmap Prediction
    print("Plotting Consolidated Heatmap Prediction...")
    plot_prediction_heatmap(ohlcv_df, options_df, symbol)

if __name__ == "__main__":
    main()
