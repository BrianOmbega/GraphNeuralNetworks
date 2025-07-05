import os
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml import command
from azure.ai.ml.entities import Data
from azure.ai.ml.entities import Environment
from azure.ai.ml import Input

# Load environment variables from .env file
load_dotenv()

# Azure ML workspace details (now loaded from .env)
subscription_id = os.getenv('AZURE_SUBSCRIPTION_ID')
resource_group = os.getenv('AZURE_RESOURCE_GROUP')
workspace_name = os.getenv('AZURE_WORKSPACE_NAME')
compute_name = os.getenv('AZURE_COMPUTE_NAME')

# Connect to Azure ML workspace
credential = DefaultAzureCredential()
print(f"DefaultAzureCredential: {credential}")
ml_client = MLClient(
    credential,
    subscription_id,
    resource_group,
    workspace_name
)
print(f"Connected to Azure ML Workspace: {workspace_name}")

# Verify connection by listing computes
try:
    computes = list(ml_client.compute.list())
    print(f"Successfully connected! Found {len(computes)} compute resources in the workspace.")
except Exception as e:
    print(f"Failed to connect or list computes: {e}")

# Register all_taxpayers.xlsx as a data asset only if it does not exist
try:
    existing_asset = ml_client.data.get(name="all_taxpayers_csv", version="6")
    print(f"Data asset already exists: {existing_asset.id}")
    excel_asset = existing_asset
except Exception:
    excel_asset = Data(
        path="all_taxpayers.csv",
        type="uri_file",
        description="VAT data CSV",
        name="all_taxpayers_csv",
        version="6"
    )
    excel_asset = ml_client.data.create_or_update(excel_asset)
    print(f"Registered data asset: {excel_asset.id}")

# Print local file size for verification
if os.path.exists("all_taxpayers.csv"):
    print(f"Local all_taxpayers.csv size: {os.path.getsize('all_taxpayers.csv')} bytes")
else:
    print("Local all_taxpayers.csv not found!")

# Register custom environment if not exists
try:
    custom_env = ml_client.environments.get(name="gnn-env", version="2")
    print(f"Environment already exists: {custom_env.id}")
except Exception:
    custom_env = Environment(
        name="gnn-env",
        version="2",
        description="Environment with pandas, xlrd, openpyxl, azure-ai-ml, etc.",
        conda_file="conda.yaml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest"
    )
    custom_env = ml_client.environments.create_or_update(custom_env)
    print(f"Registered environment: {custom_env.id}")

# Example: submit train.py as a job to your Azure ML compute
print("Preparing to submit job...")
job = command(
    code="./job_src",  # only upload minimal files
    command="python train.py --input_data ${{inputs.data}}",
    environment=custom_env.id,
    compute='bombega-ci',
    display_name="gnn-training-job-mufasa",
    inputs=dict(
        data=Input(
            type="uri_file",
            path="azureml:all_taxpayers_csv:6"  # Use the correct version or 'latest'
        )
    )
)
print("Job object created. Submitting to Azure ML...")
returned_job = ml_client.jobs.create_or_update(job)
print(f"Job submitted. View at: {returned_job.studio_url}")
