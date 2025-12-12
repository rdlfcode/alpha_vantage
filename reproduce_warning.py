
import sys
# Flush immediately
sys.stdout.reconfigure(line_buffering=True)
print("Starting script...")
try:
    print("Importing pandas...")
    import pandas as pd
    print("Importing warnings...")
    import warnings
    print("Importing duckdb...")
    import duckdb
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Unexpected error during import: {e}")
    sys.exit(1)

print("Imports done.")

warnings.simplefilter('always', FutureWarning)

def test_dtype_inference():
    print("Testing pd.concat mixed index...")
    try:
        df1 = pd.DataFrame({"A": [1, 2]}, index=[1, 2])
        df2 = pd.DataFrame({"A": [3, 4]}, index=["3", "4"])
        # This triggers: FutureWarning: The behavior of DataFrame concatenation with empty or all-NA entries is deprecated... 
        # But we are looking for Dtype inference on Index constructor.
        res = pd.concat([df1, df2]) 
    except Exception as e:
        print(f"Failed: {e}")

    print("Testing explicit Index construction from mixed list...")
    try:
        # This triggered it in older reproduction: 
        idx = pd.Index([1, "2"]) # warning?
    except Exception as e:
        print(e)
    
    print("Testing DataFrame.from_dict...")
    data = {"2021-01-01": {"val": 1}, "2021-01-02": {"val": 2}}
    df = pd.DataFrame.from_dict(data, orient="index")
    # This creates index from string keys.
    print("Finished.")

if __name__ == "__main__":
    test_dtype_inference()
