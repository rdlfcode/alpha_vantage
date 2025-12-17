
import duckdb
import pandas as pd
from pathlib import Path
from settings import settings
from alpha_vantage_schema import TABLE_SCHEMAS

def get_db_summary():
    db_path = Path(settings.get("db_path", "data/alpha_vantage.db"))
    conn = duckdb.connect(str(db_path))
    
    # Get all tables
    tables = conn.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='main' ORDER BY table_name").fetchall()
    tables = [t[0] for t in tables]
    
    summary_data = []
    
    for table in tables:
        # Check columns to find a likely date column
        columns_info = conn.execute(f"DESCRIBE {table}").fetchall()
        column_names = [c[0] for c in columns_info]
        
        # Prioritize 'dt' but fallback to others if needed
        date_col = None
        for candidate in ['dt', 'date', 'timestamp', 'latest_trading_day']:
            if candidate in column_names:
                date_col = candidate
                break
        
        try:
            if date_col:
                query = f"SELECT COUNT(*) as row_count, MIN({date_col}) as min_date, MAX({date_col}) as max_date FROM {table}"
            else:
                query = f"SELECT COUNT(*) as row_count, NULL as min_date, NULL as max_date FROM {table}"
                
            stats = conn.execute(query).fetchone()
            
            summary_data.append({
                "Table": table,
                "Rows": stats[0],
                "Min Date": stats[1],
                "Max Date": stats[2],
                "Date Col": date_col or "N/A"
            })
        except Exception as e:
            print(f"Error processing {table}: {e}")
            summary_data.append({
                "Table": table,
                "Rows": -1,
                "Min Date": None,
                "Max Date": None,
                "Date Col": "Error"
            })

    conn.close()
    
    if summary_data:
        df = pd.DataFrame(summary_data)
        # Format dates if they exist
        return df
    else:
        return pd.DataFrame()

if __name__ == "__main__":
    df = get_db_summary()
    if not df.empty:
        # Adjust display settings to make sure we see everything
        pd.set_option('display.max_rows', None)
        pd.set_option('display.width', 1000)
        pd.set_option('display.max_columns', None)
        print(df.to_string(index=False))
    else:
        print("No tables found in database.")
