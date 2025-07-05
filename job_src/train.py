import argparse
import pandas as pd
import os

parser = argparse.ArgumentParser()
parser.add_argument('--input_data', type=str, required=True, help='Path to input CSV file')
args = parser.parse_args()

print(f"Reading CSV file from: {args.input_data}")
print(f"File size: {os.path.getsize(args.input_data)} bytes")

df = pd.read_csv(args.input_data)
print(df.head())

# TODO: Add GNN modeling code using torch_geometric
