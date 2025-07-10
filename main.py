import os
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml import command
from azure.ai.ml.entities import Environment
from azure.ai.ml import Input, Output
import time

# NOTE: Data asset registration has been moved to register_data_assets.py.
# Please run that script before submitting jobs to ensure all assets are registered.

# Load environment variables from .env file
load_dotenv()

# Azure ML workspace details (now loaded from .env)
subscription_id = os.getenv('AZURE_SUBSCRIPTION_ID')
resource_group = os.getenv('AZURE_RESOURCE_GROUP')
workspace_name = os.getenv('AZURE_WORKSPACE_NAME')
compute_name = os.getenv('AZURE_COMPUTE_NAME')
compute_instance = os.getenv('AZURE_COMPUTE_INSTANCE')

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

# Register custom environment if not exists
try:
    custom_env = ml_client.environments.get(name="gnn-env", version="3")
    print(f"Environment already exists: {custom_env.id}")
except Exception:
    custom_env = Environment(
        name="gnn-env",
        version="3",
        description="Environment with torch, torch-geometric, pandas, scikit-learn, azure-ai-ml, etc.",
        conda_file="conda.yaml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest"
    )
    custom_env = ml_client.environments.create_or_update(custom_env)
    print(f"Registered environment: {custom_env.id}")

# Submit train.py job to create the graph and save description
# print("Preparing to submit graph creation job...")
# graph_job = command(
#     code="./job_src",
#     command="python generate_graph.py --input_data ${{inputs.data}} --taxpayers_data ${{inputs.taxpayers}}",
#     environment=custom_env.id,
#     compute='bombega-ci',
#     display_name="gnn-create-graph-job",
#     inputs=dict(
#         data=Input(
#             type="uri_file",
#             path="azureml:all_transactions_337:2"
#         ),
#         taxpayers=Input(
#             type="uri_file",
#             path="azureml:all_taxpayers_csv:2"
#         )
#     )
# )
# print("Graph creation job object created. Submitting to Azure ML...")
# graph_job_result = ml_client.jobs.create_or_update(graph_job)
# print(f"Graph creation job submitted. View at: {graph_job_result.studio_url}")

# # Wait for the graph creation job to complete
# print("Waiting for graph creation job to complete...")
# while True:
#     job_status = ml_client.jobs.get(graph_job_result.name).status
#     print(f"Current status: {job_status}")
#     if job_status in ["Completed", "Failed", "Cancelled"]:
#         break
#     time.sleep(30)
# if job_status != "Completed":
#     raise RuntimeError(f"Graph creation job did not complete successfully. Status: {job_status}")
# print("Graph creation job completed successfully.")

# Example: submit load_and_train_graph.py as a job to your Azure ML compute
print("Preparing to submit load_and_train_graph job...")
job = command(
    code="./job_src",  # only upload minimal files
    command="python load_and_train_graph.py --graph_data ${{inputs.graph}}",
    environment=custom_env.id,
    compute='bombega-ci',
    display_name="gnn-train-gnn-job",
    inputs=dict(
        graph=Input(
            type="uri_file",
            path="azureml:transaction_graph:2"  # Use the correct version
        )
    )
)
print("Job object created. Submitting to Azure ML...")
returned_job = ml_client.jobs.create_or_update(job)
print(f"Job submitted. View at: {returned_job.studio_url}")

# # Submit generate_taxpayers.py job to generate synthetic taxpayers
# print("Preparing to submit generate_taxpayers job...")
# generate_taxpayers_job = command(
#     code="./job_src",
#     command="python generate_taxpayers.py",
#     environment=custom_env.id,
#     compute='bombega-ci',
#     display_name="gnn-generate-taxpayers-job",
#     outputs={
#         "output_csv": Output(type="uri_file")
#     }
# )
# print("Job object created. Submitting to Azure ML...")
# generate_taxpayers_job_result = ml_client.jobs.create_or_update(generate_taxpayers_job)
# print(f"Generate taxpayers job submitted. View at: {generate_taxpayers_job_result.studio_url}")

# Submit generate_1M_transactions.py job to generate 1M synthetic transactions
# print("Preparing to submit generate_1M_transactions job...")
# generate_transactions_job = command(
#     code="./job_src",
#     command="python generate_transactions.py --taxpayers_csv ${{inputs.taxpayers}}",
#     environment=custom_env.id,
#     compute=compute_instance,
#     display_name="gnn-generate-1M-transactions-job",
#     inputs={
#         "taxpayers": Input(
#             type="uri_file",
#             path="azureml:all_taxpayers_csv:2"
#         )
#     },
#     outputs={
#         "output_csv": Output(type="uri_file")
#     }
# )
# print("Job object created. Submitting to Azure ML...")
# generate_transactions_job_result = ml_client.jobs.create_or_update(generate_transactions_job)
# print(f"Generate 1M transactions job submitted. View at: {generate_transactions_job_result.studio_url}")
