
import duckdb
import pandas as pd
from pathlib import Path
from data.settings import settings

def build_dataset():
    db_path = Path(settings.get("data_dir"), settings.get("db_name"))
    conn = duckdb.connect(str(db_path), read_only=False)

    opt_days = settings.get("option_expiration_days", [30, 91, 182, 365])

    # Dynamic Binnning Logic for Options
    # We define a window of tolerance (e.g. 25% of the target duration)
    # This groups raw contracts into bucketed "Constant Maturity" features
    case_binnning_parts = []
    for d in opt_days:
        window = max(5, int(d * 0.25))
        case_binnning_parts.append(f"WHEN abs(dte - {d}) <= {window} THEN {d}")
    
    case_binnning = "\n               ".join(case_binnning_parts)

    # Dynamic Aggregation (Manual Pivot)
    # DuckDB PIVOT is restrictive with multiple aggregates, so we build manual columns
    agg_cols = []
    for d in opt_days:
        agg_cols.extend([
            f"AVG(CASE WHEN target_dte = {d} THEN impliedVolatility END) as iv{d}",
            f"SUM(CASE WHEN target_dte = {d} THEN volume ELSE 0 END) as vol{d}",
            f"SUM(CASE WHEN target_dte = {d} THEN openInterest ELSE 0 END) as oi{d}",
            f"SUM(CASE WHEN target_dte = {d} AND type='put' THEN volume ELSE 0 END) * 1.0 / NULLIF(SUM(CASE WHEN target_dte = {d} THEN volume ELSE 0 END), 0) as pcrVol{d}",
            f"SUM(CASE WHEN target_dte = {d} AND type='put' THEN openInterest ELSE 0 END) * 1.0 / NULLIF(SUM(CASE WHEN target_dte = {d} THEN openInterest ELSE 0 END), 0) as pcrOi{d}",
            f"AVG(CASE WHEN target_dte = {d} THEN vega END) as vega{d}",
            f"AVG(CASE WHEN target_dte = {d} THEN gamma END) as gamma{d}"
        ])
    
    agg_sql = ",\n            ".join(agg_cols)

    query = f"""
    WITH fundamental_pivoted AS (
        -- Turn long-form fundamentals into wide-form (one column per metric)
        PIVOT (
            SELECT symbol, dt, metric, value 
            FROM FUNDAMENTALS 
            WHERE periodType = 'QUARTERLY'
        )
        ON metric 
        USING AVG(value) 
        GROUP BY symbol, dt
    ),
    options_calc AS (
        SELECT 
            symbol, 
            dt, 
            impliedVolatility,
            volume,
            openInterest,
            type,
            vega,
            gamma,
            -- Calculate Days to Expiry
            date_diff('day', CAST(dt AS DATE), CAST(expiration AS DATE)) as dte
        FROM HISTORICAL_OPTIONS
        WHERE impliedVolatility > 0
    ),
    options_binned AS (
        SELECT 
            symbol, 
            dt,
            impliedVolatility,
            volume,
            openInterest,
            type,
            vega,
            gamma,
            -- Assign to nearest target bucket
            CASE 
               {case_binnning}
               ELSE NULL 
            END as target_dte
        FROM options_calc
        WHERE target_dte IS NOT NULL
    ),
    options_aggregated AS (
        SELECT 
            symbol, 
            dt,
            {agg_sql}
        FROM options_binned
        GROUP BY symbol, dt
    )
    SELECT 
        tsda.symbol,
        tsda.dt,
        -- Exclude join keys and known duplicates to avoid collision
        tsda.* EXCLUDE (symbol, dt),
        -- Exclude fields likely present in FUNDAMENTALS or TSDA to prevent _1, _2 duplicates
        over.* EXCLUDE (symbol, dt, ebitda, eps, sharesOutstanding, peRatio, dividendYield, profitMargin),
        fp.* EXCLUDE (symbol, dt),
        op.* EXCLUDE (symbol, dt),
        m.* EXCLUDE (dt)
    FROM TIME_SERIES_DAILY_ADJUSTED tsda
    LEFT JOIN OVERVIEW over 
        ON tsda.symbol = over.symbol
    -- Join Options (Exact Match on Date)
    LEFT JOIN options_aggregated op
        ON tsda.symbol = op.symbol AND tsda.dt = op.dt
    -- ASOF Join Fundamentals (Forward Fill from past)
    ASOF LEFT JOIN fundamental_pivoted fp 
        ON tsda.symbol = fp.symbol AND tsda.dt >= fp.dt
    -- ASOF Join Macro (Forward Fill from past)
    ASOF LEFT JOIN MACRO m 
       ON tsda.dt >= m.dt
    ORDER BY tsda.symbol, tsda.dt
    """

    print("Executing Query...")
    # print(query) # Debug
    
    try:
        df = conn.execute(query).df()
        
        # Cleaning Logic
        df.ffill(inplace=True)
        cols_to_drop = [c for c in df.columns if df[c].nunique() <= 1 and c not in ['symbol', 'dt']]
        df.drop(columns=cols_to_drop, inplace=True)
        df.dropna(axis=0, how="any", inplace=True)
        
        return df
    except Exception as e:
        print(f"Query Failed: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

if __name__ == "__main__":
   df = build_dataset()
   print(df)
   df.to_clipboard()
