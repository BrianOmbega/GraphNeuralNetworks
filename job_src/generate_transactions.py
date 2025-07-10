import pandas as pd
import numpy as np
import random
import pickle
from datetime import datetime, timedelta
from tqdm import trange, tqdm
import os
import argparse

random.seed(42)
np.random.seed(42)

# Settings
n_transactions = 300_000
undeclared_ratio = 0.05
n_declared = int(n_transactions * (1 - undeclared_ratio))
n_undeclared = n_transactions - n_declared

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--taxpayers_csv', type=str, default='all_taxpayers.csv', help='Path to taxpayers CSV file')
args = parser.parse_args()

taxpayers_path = args.taxpayers_csv
sector_details_path = 'sector_details.pkl'
all_taxpayers_df = pd.read_csv(taxpayers_path)
with open(sector_details_path, 'rb') as f:
    sector_details = pickle.load(f)

# Allowed sector links (as in notebook)
allowed_links = {
    'Retail': ['Wholesale Trade', 'Manufacturing', 'Agriculture', 'Transportation & Logistics', 'ICT', 'Finance'],
    'Manufacturing': ['Agriculture', 'Mining & Quarrying', 'Energy', 'Transportation & Logistics', 'Wholesale Trade', 'Construction'],
    'Agriculture': ['Manufacturing', 'Wholesale Trade', 'Retail', 'Water & Sanitation', 'Finance'],
    'Construction': ['Manufacturing', 'Wholesale Trade', 'Energy', 'Transportation & Logistics', 'Finance', 'Legal & Professional Services'],
    'ICT': ['Finance', 'Telecommunications', 'Education', 'Healthcare', 'Legal & Professional Services', 'Retail', 'Entertainment & Media'],
    'Finance': ['All'],
    'Hospitality': ['Agriculture', 'Wholesale Trade', 'Retail', 'Transportation & Logistics', 'Entertainment & Media'],
    'Healthcare': ['Pharmaceuticals', 'Manufacturing', 'ICT', 'Education', 'Energy', 'Waste Management'],
    'Education': ['ICT', 'Finance', 'Publishing', 'Retail', 'Public Administration'],
    'Real Estate': ['Construction', 'Finance', 'Legal & Professional Services', 'Public Administration'],
    'Transportation & Logistics': ['Wholesale Trade', 'Retail', 'Manufacturing', 'Energy', 'Agriculture', 'Mining & Quarrying'],
    'Telecommunications': ['ICT', 'Finance', 'Education', 'Entertainment & Media'],
    'Energy': ['Manufacturing', 'Mining & Quarrying', 'Construction', 'Transportation & Logistics', 'Water & Sanitation'],
    'Legal & Professional Services': ['Finance', 'Real Estate', 'Construction', 'Public Administration', 'Healthcare'],
    'Mining & Quarrying': ['Manufacturing', 'Construction', 'Energy', 'Transportation & Logistics'],
    'Entertainment & Media': ['Retail', 'ICT', 'Arts & Culture', 'Telecommunications', 'Public Administration'],
    'Public Administration': ['All'],
    'Water & Sanitation': ['Construction', 'Agriculture', 'Public Administration', 'Healthcare'],
    'Waste Management': ['Healthcare', 'Construction', 'Public Administration', 'Water & Sanitation'],
    'Security Services': ['Retail', 'Finance', 'Public Administration', 'Real Estate', 'Healthcare'],
    'Wholesale Trade': ['Manufacturing', 'Agriculture', 'Retail', 'Transportation & Logistics'],
    'Arts & Culture': ['Education', 'Entertainment & Media', 'Retail', 'Public Administration'],
    'Transportation': ['Retail', 'Wholesale Trade', 'Agriculture', 'Construction', 'Healthcare']
}

invoice_statuses = ['Paid', 'Pending', 'Cancelled', 'Disputed']
payment_methods = ['Bank Transfer', 'Mobile Money', 'Cash', 'Cheque', 'Credit']

# Helper functions
def is_valid_transaction(seller_sector, buyer_sector):
    allowed = allowed_links.get(seller_sector, [])
    return buyer_sector in allowed or 'All' in allowed or random.random() < 0.05

def sample_seasonal_date(sector):
    # Simple random date in 2022 for this script
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    return datetime(2022, month, day)

def generate_transaction(seller, buyer, declared=1):
    goods_list = sector_details.get(seller['economic_sector'], {}).get('outputs', ['General Item'])
    description = random.choice(goods_list)
    sales_amount = round(random.uniform(500, 100000), 2)
    invoice_date = sample_seasonal_date(seller['economic_sector'])
    invoice_status = random.choices(invoice_statuses, weights=[0.7, 0.2, 0.05, 0.05])[0]
    payment_method = random.choice(payment_methods)
    return {
        'seller_id': seller['taxpayer_id'],
        'buyer_id': buyer['taxpayer_id'],
        'description_of_goods': description,
        'sales_amount': sales_amount,
        'invoice_date': invoice_date,
        'invoice_status': invoice_status,
        'payment_method': payment_method,
        'location': seller['location'],
        'economic_sector': seller['economic_sector'],
        'buyer_size_category': buyer['size_category'],
        'seller_size_category': seller['size_category'],
        'declared': declared
    }

# Generate declared transactions
buyers = all_taxpayers_df.copy()
sellers = all_taxpayers_df.copy()
high_degree_buyers = buyers.nlargest(100, 'size')['taxpayer_id'].tolist()
transactions = []
invalid_transactions = set()

declared_transactions = []
for _ in trange(n_declared, desc='Declared transactions'):
    buyer_id = random.choice(high_degree_buyers)
    buyer = buyers[buyers['taxpayer_id'] == buyer_id].iloc[0]
    seller = sellers.sample(weights=sellers['norm_size']).iloc[0]
    if seller['taxpayer_id'] == buyer['taxpayer_id']:
        continue
    if (seller['taxpayer_id'], buyer['taxpayer_id']) in invalid_transactions:
        continue
    if not is_valid_transaction(seller['economic_sector'], buyer['economic_sector']):
        invalid_transactions.add((seller['taxpayer_id'], buyer['taxpayer_id']))
        continue
    declared_transactions.append(generate_transaction(seller, buyer, 1))

# Undeclared for declared transactions
undeclared_transactions = []
for _ in trange(n_undeclared, desc='Undeclared for declared'):
    ref_tx = random.choice(declared_transactions)
    buyer_id = ref_tx['buyer_id']
    buyer = all_taxpayers_df[all_taxpayers_df['taxpayer_id'] == buyer_id].iloc[0]
    small_sellers = all_taxpayers_df[
        (all_taxpayers_df['size'] < all_taxpayers_df['size'].quantile(0.3)) &
        (all_taxpayers_df['location'] == ref_tx['location']) &
        (all_taxpayers_df['economic_sector'] == ref_tx['economic_sector']) &
        (all_taxpayers_df['taxpayer_id'] != buyer_id)
    ]
    if small_sellers.empty:
        continue
    seller = small_sellers.sample(1).iloc[0]
    undeclared_transactions.append(generate_transaction(seller, buyer, 0))

# Generate power-law transactions (buyers skewed toward large firms)
power_law_transactions = []
for _ in trange(n_declared // 2, desc='Power-law transactions'):
    buyer_id = random.choice(high_degree_buyers)
    buyer = buyers[buyers['taxpayer_id'] == buyer_id].iloc[0]
    seller = sellers.sample(weights=sellers['norm_size']).iloc[0]
    if seller['taxpayer_id'] == buyer['taxpayer_id']:
        continue
    if (seller['taxpayer_id'], buyer['taxpayer_id']) in invalid_transactions:
        continue
    if not is_valid_transaction(seller['economic_sector'], buyer['economic_sector']):
        invalid_transactions.add((seller['taxpayer_id'], buyer['taxpayer_id']))
        continue
    power_law_transactions.append(generate_transaction(seller, buyer, 1))

# Undeclared for power-law transactions
undeclared_power_law_transactions = []
power_law_declared_df = pd.DataFrame(power_law_transactions)
for _ in trange(n_undeclared // 2, desc='Undeclared for power-law'):
    if power_law_declared_df.empty:
        break
    ref_tx = power_law_declared_df.sample(1).iloc[0]
    buyer_id = ref_tx['buyer_id']
    buyer = all_taxpayers_df[all_taxpayers_df['taxpayer_id'] == buyer_id].iloc[0]
    small_sellers = all_taxpayers_df[
        (all_taxpayers_df['size'] < all_taxpayers_df['size'].quantile(0.3)) &
        (all_taxpayers_df['location'] == ref_tx['location']) &
        (all_taxpayers_df['economic_sector'] == ref_tx['economic_sector']) &
        (all_taxpayers_df['taxpayer_id'] != buyer_id)
    ]
    if small_sellers.empty:
        continue
    seller = small_sellers.sample(1).iloc[0]
    undeclared_power_law_transactions.append(generate_transaction(seller, buyer, 0))

# Location-sector transactions
declared_location_sector_transactions = []
location_sector_flows = {
    ('Thika', 'Nairobi'): [ 'Manufacturing', 'Agriculture', 'Retail', 'Wholesale', 'Construction'],
    ('Nakuru', 'Nairobi'): ['Agriculture', 'Retail', 'Construction'],
    ('Kisumu', 'Nairobi'): ['Retail', 'ICT', 'Wholesale'],
    ('Nyeri', 'Kericho'): ['Manufacturing', 'Agriculture', 'Energy'],
    ('Eldoret', 'Thika'): ['Agriculture', 'Manufacturing', 'Transport'],
    ('Machakos', 'Nairobi'): ['Construction', 'Logistics'],
    ('Nairobi', 'Mombasa'): ['ICT', 'Finance', 'Trade'],
    ('Mombasa', 'Nairobi'): ['Logistics', 'Wholesale', 'Hospitality', 'Manufacturing', 'Construction', 'Energy', 'Transport', 'Trade']
}
for _ in trange(n_declared // 4, desc='Location-sector transactions'):
    seller = sellers.sample(weights=sellers['norm_size']).iloc[0]
    origin = seller['location']
    matches = [k for k in location_sector_flows if k[0] == origin]
    if matches:
        dest_pair = random.choice(matches)
        dest_location = dest_pair[1]
        expected_sectors = location_sector_flows[dest_pair]
        if seller['economic_sector'] not in expected_sectors:
            continue
        buyers_in_dest = all_taxpayers_df[
            (all_taxpayers_df['location'] == dest_location) &
            (all_taxpayers_df['taxpayer_id'] != seller['taxpayer_id'])
        ]
    else:
        buyers_in_dest = all_taxpayers_df[
            (all_taxpayers_df['location'] != origin) &
            (all_taxpayers_df['taxpayer_id'] != seller['taxpayer_id'])
        ]
    if buyers_in_dest.empty:
        continue
    buyer = buyers_in_dest.sample(weights=buyers_in_dest['norm_size']).iloc[0]
    declared_location_sector_transactions.append(generate_transaction(seller, buyer, 1))

# Undeclared for location-sector transactions
undeclared_location_sector_transactions = []
declared_loc_sec_df = pd.DataFrame(declared_location_sector_transactions)
for _ in trange(n_undeclared // 4, desc='Undeclared for location-sector'):
    if declared_loc_sec_df.empty:
        break
    ref_tx = declared_loc_sec_df.sample(1).iloc[0]
    buyer_id = ref_tx['buyer_id']
    buyer = all_taxpayers_df[all_taxpayers_df['taxpayer_id'] == buyer_id].iloc[0]
    small_sellers = all_taxpayers_df[
        (all_taxpayers_df['size'] < all_taxpayers_df['size'].quantile(0.3)) &
        (all_taxpayers_df['location'] == ref_tx['location']) &
        (all_taxpayers_df['economic_sector'] == ref_tx['economic_sector']) &
        (all_taxpayers_df['taxpayer_id'] != buyer_id)
    ]
    if small_sellers.empty:
        continue
    seller = small_sellers.sample(1).iloc[0]
    undeclared_location_sector_transactions.append(generate_transaction(seller, buyer, 0))

# Sector burst weights (seasonality)
sector_burst_weights = {
    'Retail': [0.12, 0.07, 0.06, 0.06, 0.06, 0.10, 0.05, 0.05, 0.06, 0.08, 0.12, 0.17],
    'Agriculture': [0.05, 0.05, 0.10, 0.10, 0.10, 0.05, 0.05, 0.05, 0.07, 0.13, 0.12, 0.08],
    'Hospitality': [0.08, 0.06, 0.06, 0.12, 0.07, 0.07, 0.07, 0.12, 0.06, 0.06, 0.06, 0.17],
    'Transport & Logistics': [0.06, 0.06, 0.06, 0.12, 0.08, 0.06, 0.06, 0.12, 0.06, 0.06, 0.06, 0.16],
    'Construction': [0.10, 0.10, 0.10, 0.06, 0.06, 0.06, 0.10, 0.10, 0.10, 0.05, 0.04, 0.03],
    'Entertainment & Media': [0.10, 0.06, 0.05, 0.08, 0.08, 0.06, 0.06, 0.08, 0.06, 0.08, 0.10, 0.19],
    'Energy': [0.08, 0.08, 0.08, 0.09, 0.09, 0.08, 0.08, 0.08, 0.08, 0.09, 0.09, 0.08],
    'Real Estate': [0.09, 0.09, 0.09, 0.07, 0.07, 0.07, 0.10, 0.10, 0.10, 0.07, 0.08, 0.07],
    'Other': [0.05] * 12
}

# Function to sample date with seasonality and burst variations
def sample_sector_burst_date(sector):
    monthly_weights = sector_burst_weights.get(sector, sector_burst_weights['Other'])
    month = random.choices(range(12), weights=monthly_weights)[0]
    day = random.randint(1, 28)
    return datetime(2022, month + 1, day)

# Generate sector burst transactions
sector_burst_transactions = []
for sector, weights in sector_burst_weights.items():
    sector_taxpayers = all_taxpayers_df[all_taxpayers_df['economic_sector'] == sector]
    if sector_taxpayers.empty:
        continue
    small_taxpayers = sector_taxpayers[sector_taxpayers['size'] < sector_taxpayers['size'].quantile(0.5)]
    if len(small_taxpayers) < 2:
        continue
    n_sector_txns = int(n_transactions * (len(sector_taxpayers) / len(all_taxpayers_df)))
    n_declared = int(n_sector_txns * (1 - undeclared_ratio))
    n_undeclared = n_sector_txns - n_declared
    # Valid pairs for declared txns
    valid_pairs = [
        (s['taxpayer_id'], b['taxpayer_id'])
        for i, s in small_taxpayers.iterrows()
        for j, b in small_taxpayers.iterrows()
        if s['taxpayer_id'] != b['taxpayer_id']
        and is_valid_transaction(s['economic_sector'], b['economic_sector'])
    ]
    if not valid_pairs:
        continue
    sampled_declared = random.choices(valid_pairs, k=n_declared)
    for seller_id, buyer_id in tqdm(sampled_declared, desc=f"{sector} declared"):
        seller = small_taxpayers[small_taxpayers['taxpayer_id'] == seller_id].iloc[0]
        buyer = small_taxpayers[small_taxpayers['taxpayer_id'] == buyer_id].iloc[0]
        tx = generate_transaction(seller, buyer, 1)
        tx['invoice_date'] = sample_sector_burst_date(sector)
        sector_burst_transactions.append(tx)
    sampled_undeclared = random.choices(valid_pairs, k=n_undeclared)
    for seller_id, buyer_id in tqdm(sampled_undeclared, desc=f"{sector} undeclared"):
        seller = small_taxpayers[small_taxpayers['taxpayer_id'] == seller_id].iloc[0]
        buyer = small_taxpayers[small_taxpayers['taxpayer_id'] == buyer_id].iloc[0]
        tx = generate_transaction(seller, buyer, 0)
        tx['invoice_date'] = sample_sector_burst_date(sector)
        sector_burst_transactions.append(tx)

# Add all transactions to main transaction list
transactions = declared_transactions + undeclared_transactions + power_law_transactions + undeclared_power_law_transactions + declared_location_sector_transactions + undeclared_location_sector_transactions + sector_burst_transactions

# Save to CSV
os.makedirs('outputs', exist_ok=True)
output_path = os.path.join('outputs', 'synthetic_transactions_1M.csv')
pd.DataFrame(transactions).to_csv(output_path, index=False)
print(f"Generated {len(transactions)} transactions and saved to {output_path}")
