
import pandas as pd
import os

if os.path.exists("test_classification.csv"):
    with open("test_classification.csv", "r") as f:
        print("First 5 lines of CSV:")
        for i in range(5):
            print(f.readline().strip())
            
    try:
        df = pd.read_csv("test_classification.csv")
        print("\nPandas Read Success!")
        print(df.head())
    except Exception as e:
        print(f"\nPandas Read Failed: {e}")
else:
    print("test_classification.csv does not exist.")
