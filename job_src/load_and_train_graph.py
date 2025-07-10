# This file was moved to job_src/load_and_train_graph.py. Please use that version for Azure ML jobs.

import os
import sys
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F
import torch.nn as nn
import argparse
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, average_precision_score
)
import pandas as pd
from models import GAT, GCN, GraphSAGE, EdgeClassifier
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser()
parser.add_argument('--graph_data', type=str, required=True, help='Path to graph_data.pt file')
args = parser.parse_args()

graph_path = args.graph_data
print("[INFO] Starting GNN training job in Azure ML...")
print(f"[INFO] Loading graph from: {graph_path}")
data = torch.load(graph_path, weights_only=False)
print("[INFO] Graph loaded. Splitting edges into train/val/test...")

# Split the edges into training, validation, and testing
edges = data.edge_index.t().numpy()              # shape: [num_edges, 2]
labels = data.edge_label.numpy()                 # shape: [num_edges]
edge_attrs = data.edge_attr.numpy()              # shape: [num_edges, num_edge_features]

# Train/val/test split with edge_attr included
X_train, X_temp, y_train, y_temp, attr_train, attr_temp = train_test_split(
    edges, labels, edge_attrs, test_size=0.3, stratify=labels, random_state=42
)
X_val, X_test, y_val, y_test, attr_val, attr_test = train_test_split(
    X_temp, y_temp, attr_temp, test_size=0.5, stratify=y_temp, random_state=42
)
print(f"[INFO] Edge splits: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

def to_torch_edges(edges, labels, attrs):
    edge_index = torch.tensor(edges).t().contiguous()                # shape: [2, num_edges]
    edge_label = torch.tensor(labels, dtype=torch.float)             # shape: [num_edges]
    edge_attr = torch.tensor(attrs, dtype=torch.float)               # shape: [num_edges, num_edge_features]
    return edge_index, edge_label, edge_attr

train_edge_index, train_edge_label, train_edge_attr = to_torch_edges(X_train, y_train, attr_train)
val_edge_index, val_edge_label, val_edge_attr = to_torch_edges(X_val, y_val, attr_val)
test_edge_index, test_edge_label, test_edge_attr = to_torch_edges(X_test, y_test, attr_test)

# Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)

# Move edge splits to device
train_edge_index = train_edge_index.to(device)
train_edge_label = train_edge_label.to(device)
train_edge_attr = train_edge_attr.to(device)
val_edge_index = val_edge_index.to(device)
val_edge_label = val_edge_label.to(device)
val_edge_attr = val_edge_attr.to(device)

# Model configs
model_configs = [
    ("GraphSAGE", GraphSAGE, {"in_channels": data.num_node_features, "hidden_channels": 64}),
    ("GAT", GAT, {"in_channels": data.num_node_features, "hidden_channels": 64, "heads": 2}),
    ("GCN", GCN, {"in_channels": data.num_node_features, "hidden_channels": 64})
#    ("GIN", GIN, {"in_channels": data.num_node_features, "hidden_channels": 64}),
]

all_metrics = []

# Ensure outputs directory exists and print absolute path
outputs_dir = os.path.abspath('outputs')
os.makedirs(outputs_dir, exist_ok=True)
print(f"[INFO] Outputs will be saved to: {outputs_dir}")

for model_name, model_class, model_kwargs in model_configs:
    print(f"[INFO] Training {model_name}...")
    start_time = time.time()
    # Re-initialize model and optimizer for each run
    model = model_class(**model_kwargs).to(device)
    edge_classifier = EdgeClassifier(emb_dim=64, edge_feat_dim=train_edge_attr.shape[1]).to(device)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(edge_classifier.parameters()), lr=0.005)
    loss_fn = nn.BCELoss()
    train_log = []
    # Training loop
    for epoch in range(1, 101):
        model.train()
        optimizer.zero_grad()
        z = model(data.x, data.edge_index)
        pred = edge_classifier(z, train_edge_index, train_edge_attr)
        loss = loss_fn(pred, train_edge_label)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                val_pred = edge_classifier(z, val_edge_index, val_edge_attr)
                val_loss = loss_fn(val_pred, val_edge_label)
                val_acc = ((val_pred > 0.5) == val_edge_label).float().mean()
                print(f"[INFO] {model_name} Epoch {epoch:03d} | Train Loss: {loss.item():.4f} | Val Acc: {val_acc:.4f}")
                train_log.append((epoch, loss.item(), val_acc.item()))
    elapsed = time.time() - start_time
    print(f"[INFO] {model_name} training completed in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    # Inference on test set
    model.eval()
    with torch.no_grad():
        z = model(data.x, data.edge_index)
        test_pred = edge_classifier(z, test_edge_index, test_edge_attr)
    true_labels = test_edge_label.cpu().numpy().astype(int)
    predicted_probs = test_pred.cpu().numpy()
    predicted_labels = (predicted_probs > 0.5).astype(int)
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, pos_label=0)
    recall = recall_score(true_labels, predicted_labels, pos_label=0)
    f1 = f1_score(true_labels, predicted_labels, pos_label=0)
    roc_auc = roc_auc_score(true_labels, predicted_probs)
    pr_auc = average_precision_score(true_labels, predicted_probs, pos_label=0)
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    all_metrics.append({
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "conf_matrix": conf_matrix,
        "train_log": train_log
    })
    # Save per-model log
    log_txt_path = os.path.join('outputs', f'train_log_{model_name}.txt')
    with open(log_txt_path, 'w') as f:
        for epoch, train_loss, val_acc in train_log:
            f.write(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}\n")
        f.write(f"Training time (seconds): {elapsed:.2f}\n")
        f.write(f"Training time (minutes): {elapsed/60:.2f}\n")
    print(f"[INFO] Training log saved to {log_txt_path}")
    # Save per-model metrics
    metrics_txt_path = os.path.join('outputs', f'test_metrics_{model_name}.txt')
    with open(metrics_txt_path, 'w') as f:
        f.write(f"Accuracy:           {accuracy:.4f}\n")
        f.write(f"Precision (class 0):{precision:.4f}\n")
        f.write(f"Recall (class 0):   {recall:.4f}\n")
        f.write(f"F1 Score (class 0): {f1:.4f}\n")
        f.write(f"ROC AUC:            {roc_auc:.4f}\n")
        f.write(f"PR AUC (class 0):   {pr_auc:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        conf_df = pd.DataFrame(
            conf_matrix,
            index=["Actual: Undeclared", "Actual: Declared"],
            columns=["Predicted: Undeclared", "Predicted: Declared"]
        )
        conf_df.to_csv(f, sep='\t')
    print(f"[INFO] Test metrics saved to {metrics_txt_path}")

# Save summary table
summary_path = os.path.join('outputs', 'model_comparison_summary.txt')
with open(summary_path, 'w') as f:
    f.write("Model\tAccuracy\tPrecision\tRecall\tF1\tROC_AUC\tPR_AUC\n")
    for m in all_metrics:
        f.write(f"{m['model']}\t{m['accuracy']:.4f}\t{m['precision']:.4f}\t{m['recall']:.4f}\t{m['f1']:.4f}\t{m['roc_auc']:.4f}\t{m['pr_auc']:.4f}\n")
print(f"[INFO] Model comparison summary saved to {summary_path}")

print("[INFO] GNN training job finished.")
