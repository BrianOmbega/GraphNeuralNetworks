import argparse
import pandas as pd
import os
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--input_data', type=str, required=True, help='Path to input CSV file')
parser.add_argument('--taxpayers_data', type=str, required=True, help='Path to taxpayers CSV file')
args = parser.parse_args()

print(f"Reading transactions CSV file from: {args.input_data}")

print(f"Reading taxpayers CSV file from: {args.taxpayers_data}")
print(f"File size: {os.path.getsize(args.taxpayers_data)} bytes")



# Load transactions data
all_transactions_df = pd.read_csv(args.input_data)
print(len(all_transactions_df))
print(all_transactions_df.head())


# Load taxpayers data
all_taxpayers_df = pd.read_csv(args.taxpayers_data)
print(f"File size: {os.path.getsize(args.input_data)} bytes")
print(f"len(all_taxpayers_df): {len(all_taxpayers_df)}")
print(f"len(all_transactions_df): {len(all_transactions_df)}")


# Encode categorical edge features
for col in ['description_of_goods', 'invoice_status', 'payment_method']:
    encoder = LabelEncoder()
    all_transactions_df[col + '_encoded'] = encoder.fit_transform(all_transactions_df[col])

all_transactions_df['day_of_week'] = pd.to_datetime(all_transactions_df['invoice_date'], dayfirst=True).dt.weekday
all_transactions_df['month'] = pd.to_datetime(all_transactions_df['invoice_date'], dayfirst=True).dt.month
all_transactions_df['month_sin'] = np.sin(2 * np.pi * all_transactions_df['month'] / 12)
all_transactions_df['month_cos'] = np.cos(2 * np.pi * all_transactions_df['month'] / 12)

edge_features = [
    'description_of_goods_encoded', 'invoice_status_encoded',
    'payment_method_encoded', 'day_of_week', 'month', 'month_sin',
    'month_cos'
]

# Encode node IDs
all_taxpayers = pd.unique(all_transactions_df[['buyer_id', 'seller_id']].values.ravel())
node_encoder = {tid: i for i, tid in enumerate(all_taxpayers)}
n_nodes = len(all_taxpayers)

# Edge index
edge_index_np = np.array([
    all_transactions_df['seller_id'].map(node_encoder).values,
    all_transactions_df['buyer_id'].map(node_encoder).values
])
edge_index = torch.tensor(edge_index_np, dtype=torch.long)

# Edge labels
edge_label = torch.tensor(all_transactions_df['declared'].values, dtype=torch.float)

# Node features
taxpayer_info = all_taxpayers_df.set_index('taxpayer_id').loc[all_taxpayers]
node_features = pd.get_dummies(taxpayer_info[['location', 'economic_sector', 'size_category']])
node_features['size'] = taxpayer_info['size'].values
scaler = StandardScaler()
x = torch.tensor(scaler.fit_transform(node_features.values), dtype=torch.float)

# Edge features
graph_edge_features = all_transactions_df[edge_features]
edge_attr = torch.tensor(graph_edge_features.values, dtype=torch.float)

# Create PyG Data object
graph_data = Data(
    x=x,                 # node features
    edge_index=edge_index,
    edge_attr=edge_attr, # edge features
    edge_label=edge_label # edge labels for classification
)

# Save the graph object
os.makedirs('outputs', exist_ok=True)
output_path = os.path.join('outputs', 'graph_data.pt')
torch.save(graph_data, output_path)
print(f"Graph data saved to: {output_path}")

# Describe graph function
def describe_graph(G):
    import numpy as np
    import networkx as nx
    output = []
    output.append('📊 Basic Graph Properties')
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    density = nx.density(G)
    degrees = [deg for _, deg in G.degree()]
    avg_degree = np.mean(degrees)
    degree_distribution = np.bincount(degrees)
    output.append(f"- Number of nodes: {num_nodes}")
    output.append(f"- Number of edges: {num_edges}")
    output.append(f"- Density: {density:.4f}")
    output.append(f"- Average degree: {avg_degree:.2f}")
    output.append(f"- Degree distribution (sample): {dict(zip(*np.unique(degrees, return_counts=True)))}")
    output.append("\n🔗 Connectivity Metrics")
    if nx.is_connected(G):
        diameter = nx.diameter(G)
        avg_path_length = nx.average_shortest_path_length(G)
        output.append(f"- Diameter: {diameter}")
        output.append(f"- Average path length: {avg_path_length:.2f}")
    else:
        num_components = nx.number_connected_components(G)
        largest_cc = max(nx.connected_components(G), key=len)
        subgraph = G.subgraph(largest_cc)
        diameter = nx.diameter(subgraph)
        avg_path_length = nx.average_shortest_path_length(subgraph)
        output.append(f"- Connected components: {num_components}")
        output.append(f"- Diameter (of largest component): {diameter}")
        output.append(f"- Average path length (largest component): {avg_path_length:.2f}")
    clustering_coeff = nx.average_clustering(G)
    transitivity = nx.transitivity(G)
    output.append(f"- Average clustering coefficient: {clustering_coeff:.4f}")
    output.append(f"- Transitivity: {transitivity:.4f}")
    return '\n'.join(output)

# After saving the graph object, describe and save the graph
from torch_geometric.utils import to_networkx
G = to_networkx(graph_data, to_undirected=True)
graph_description = describe_graph(G)
with open(os.path.join('outputs', 'graph_description.txt'), 'w') as f:
    f.write(graph_description)
print("Graph description saved to outputs/graph_description.txt")
