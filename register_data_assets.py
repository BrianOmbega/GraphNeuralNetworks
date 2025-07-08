import os
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data

# ---
# To delete a data asset from Azure ML, uncomment and use the following lines:
# try:
#     ml_client.data.delete(name='all_taxpayers_csv', version='1')
#     print('Deleted data asset: all_taxpayers_csv version 1')
# except Exception as e:
#     print(f'Error deleting data asset: {e}')
# ---

# Load environment variables from .env file
load_dotenv()

subscription_id = os.getenv('AZURE_SUBSCRIPTION_ID')
resource_group = os.getenv('AZURE_RESOURCE_GROUP')
workspace_name = os.getenv('AZURE_WORKSPACE_NAME')

credential = DefaultAzureCredential()
ml_client = MLClient(
    credential,
    subscription_id,
    resource_group,
    workspace_name
)

# List of data assets to register (add more as needed)
data_assets = [
    {
        'name': 'all_taxpayers_csv',
        'path': 'data assets/all_taxpayers_337.csv',
        'description': 'VAT data CSV',
        'version': '1',
        'type': 'uri_file'
    },
    {
        'name': 'all_transactions_337',
        'path': 'data assets/all_transactions_337.csv',
        'description': 'VAT transaction data CSV',
        'version': '1',
        'type': 'uri_file'
    },
    {
    'name': 'transaction_graph',
    'path': 'data assets/graph_data.pt',
    'description': 'VAT transaction graph ',
    'version': '1',
    'type': 'uri_file'
    }

    # Add more assets here
]

for asset in data_assets:
    try:
        existing = ml_client.data.get(name=asset['name'], version=asset['version'])
        print(f"Data asset already exists: {existing.id}")
    except Exception:
        if not os.path.exists(asset['path']):
            print(f"File not found: {asset['path']}")
            continue
        data_asset = Data(
            path=asset['path'],
            type=asset['type'],
            description=asset['description'],
            name=asset['name'],
            version=asset['version']
        )
        registered = ml_client.data.create_or_update(data_asset)
        print(f"Registered data asset: {registered.id}")

print("Data asset registration complete.")
