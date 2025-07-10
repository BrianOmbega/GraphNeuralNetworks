import pandas as pd
import numpy as np
import random
import pickle
from datetime import datetime, timedelta
from tqdm import trange, tqdm
import os

random.seed(42)
np.random.seed(42)

# Settings
n_transactions = 1_000_000
undeclared_ratio = 0.05
n_declared = int(n_transactions * (1 - undeclared_ratio))
n_undeclared = n_transactions - n_declared

# Load taxpayers and sector details
taxpayers_path = 'all_taxpayers.csv'
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
    transactions.append(generate_transaction(seller, buyer, 1))

# Generate undeclared transactions
for _ in trange(n_undeclared, desc='Undeclared transactions'):
    buyer = buyers.sample().iloc[0]
    small_sellers = sellers[sellers['size'] < sellers['size'].quantile(0.3)]
    if small_sellers.empty:
        continue
    seller = small_sellers.sample().iloc[0]
    if seller['taxpayer_id'] == buyer['taxpayer_id']:
        continue
    if not is_valid_transaction(seller['economic_sector'], buyer['economic_sector']):
        continue
    transactions.append(generate_transaction(seller, buyer, 0))

# Save to CSV
os.makedirs('outputs', exist_ok=True)
output_path = os.path.join('outputs', 'synthetic_transactions_1M.csv')
pd.DataFrame(transactions).to_csv(output_path, index=False)
print(f"Generated {len(transactions)} transactions and saved to {output_path}")
