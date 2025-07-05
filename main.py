import os
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import CommandJob

# Load environment variables from .env file
load_dotenv()

# Azure ML workspace details (now loaded from .env)
subscription_id = os.getenv('AZURE_SUBSCRIPTION_ID')
resource_group = os.getenv('AZURE_RESOURCE_GROUP')
workspace_name = os.getenv('AZURE_WORKSPACE_NAME')
compute_name = os.getenv('AZURE_COMPUTE_NAME')

# Connect to Azure ML workspace
ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id,
    resource_group,
    workspace_name
)
print(f"Connected to Azure ML Workspace: {workspace_name}")

# Example: submit train.py as a job to your Azure ML compute
job = CommandJob(
    code=".",  # current directory
    command="python train.py",
    environment="AzureML-sklearn-1.0-ubuntu20.04-py38-cpu:1",  # or your custom environment
    compute=compute_name,
    display_name="gnn-training-job",
)

returned_job = ml_client.jobs.create_or_update(job)
print(f"Job submitted. View at: {returned_job.studio_url}")
