import pandas as pd
import warnings

# Filter warnings to ensure we catch them
warnings.simplefilter('always', FutureWarning)

print(f"Pandas version: {pd.__version__}")

print("Attempting to trigger warning with Index([1, 'a'])...")
try:
    # This is a classic mixed-type inference case
    idx = pd.Index([1, 'a'])
    print(f"Index created: {idx}")
except Exception as e:
    print(f"Error: {e}")

print("\nAttempting to trigger warning with DataFrame.from_dict...")
data = {"a": {"x": 1}, "b": {"x": 2}}
df = pd.DataFrame.from_dict(data, orient='index')
print(f"DataFrame created:\n{df}")

print("\nAttempting to trigger warning with Index(Series)...")
try:
    s = pd.Series([1, 'a'])
    idx = pd.Index(s)
    print(f"Index from Series created: {idx}")
except Exception as e:
    print(f"Error: {e}")

print("\nAttempting to trigger warning with read_csv numeric headers...")
from io import StringIO
csv_data = "1,2\n3,4"
df_csv = pd.read_csv(StringIO(csv_data))
print(f"CSV Columns: {df_csv.columns}")

print("\nAttempting to trigger warning with Index(dict_keys)...")
d = {1: 'a', 'b': 'c'}
idx = pd.Index(d.keys())
print(f"Index from dict_keys created: {idx}")


