import pandas as pd

print("Testing date parsing:")
test_dates = ['2023-01-01', '2023-01-01T00:00', '2023-01-01T00:00:00']

for d in test_dates:
    try:
        result = pd.to_datetime(d)
        print(f"{d} -> {result}")
    except Exception as e:
        print(f"{d} -> Error: {e}")