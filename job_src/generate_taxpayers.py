import pandas as pd
import random
from datetime import datetime, timedelta
import numpy as np
import os

random.seed(42)

# Settings
n_non_individuals = 50000
locations = [
    'Nairobi', 'Mombasa', 'Kisumu', 'Eldoret', 'Nakuru', 'Thika', 'Machakos',
    'Kericho', 'Nyeri', 'Garissa', 'Meru', 'Kitale'
]

# Example sector list (customize as needed)
sectors = [
    'Retail', 'Manufacturing', 'Agriculture', 'Construction', 'ICT', 'Finance',
    'Hospitality', 'Healthcare', 'Education', 'Real Estate', 'Transportation & Logistics',
    'Telecommunications', 'Energy', 'Legal & Professional Services', 'Mining & Quarrying',
    'Entertainment & Media', 'Public Administration', 'Water & Sanitation', 'Waste Management',
    'Security Services', 'Wholesale Trade', 'Arts & Culture', 'Transportation'
]

def random_registration_date(start_year=2005, end_year=2023):
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)
    delta = end_date - start_date
    return start_date + timedelta(days=random.randint(0, delta.days))

# Generate synthetic directors
n_directors = 100
list_of_directors = [f"A{str(i).zfill(4)}" for i in range(1, n_directors + 1)]

non_individuals = []
for i in range(1, n_non_individuals + 1):
    taxpayer_id = f"P{str(i).zfill(4)}"
    owners = random.sample(list_of_directors, k=random.randint(1, 3))
    non_individuals.append({
        'taxpayer_id': taxpayer_id,
        'location': random.choice(locations),
        'economic_sector': random.choice(sectors),
        'entity_type': 'Non-Individual',
        'beneficial_owners': owners,
        'registration_date': random_registration_date().date()
    })

non_individuals_df = pd.DataFrame(non_individuals)

# Add a 'size' column (simulate with lognormal distribution)
np.random.seed(42)
non_individuals_df['size'] = np.random.lognormal(mean=12, sigma=1, size=len(non_individuals_df))

# Normalize sizes for probability distribution
non_individuals_df['norm_size'] = non_individuals_df['size'] / non_individuals_df['size'].sum()

# Add size category to taxpayers
quantiles = non_individuals_df['size'].quantile([0.25, 0.5, 0.75])
def categorize_size(size):
    if size <= quantiles[0.25]:
        return 'micro'
    elif size <= quantiles[0.5]:
        return 'small'
    elif size <= quantiles[0.75]:
        return 'medium'
    else:
        return 'large'
non_individuals_df['size_category'] = non_individuals_df['size'].apply(categorize_size)

# Ensure outputs directory exists
os.makedirs('outputs', exist_ok=True)
output_csv_path = os.path.join('outputs', 'non_individuals_50000.csv')
non_individuals_df.to_csv(output_csv_path, index=False)
print(f'Generated 50000 non-individual taxpayers and saved to {output_csv_path}')
