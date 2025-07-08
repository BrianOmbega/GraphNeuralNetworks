# This file was moved to job_src/load_and_train_graph.py. Please use that version for Azure ML jobs.

import os
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
from models import GAT, GCN, GIN
import time

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

# Define GraphSAGE model
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

# Define Edge Classifier
class EdgeClassifier(nn.Module):
    def __init__(self, emb_dim, edge_feat_dim):
        super(EdgeClassifier, self).__init__()
        self.fc1 = nn.Linear(2 * emb_dim + edge_feat_dim, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, z, edge_index, edge_attr):
        src = z[edge_index[0]]
        dst = z[edge_index[1]]
        x = torch.cat([src, dst, edge_attr], dim=1)
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x)).squeeze()

# Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GraphSAGE(in_channels=data.num_node_features, hidden_channels=64).to(device)
edge_classifier = EdgeClassifier(emb_dim=64, edge_feat_dim=train_edge_attr.shape[1]).to(device)
optimizer = torch.optim.Adam(list(model.parameters()) + list(edge_classifier.parameters()), lr=0.005)
loss_fn = nn.BCELoss()
data = data.to(device)

# Move edge splits to device
train_edge_index = train_edge_index.to(device)
train_edge_label = train_edge_label.to(device)
train_edge_attr = train_edge_attr.to(device)
val_edge_index = val_edge_index.to(device)
val_edge_label = val_edge_label.to(device)
val_edge_attr = val_edge_attr.to(device)

print("[INFO] Initializing models and optimizer...")
# Training loop
print("[INFO] Starting training loop...")
train_log = []  # Collect (epoch, train_loss, val_acc)
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
            print(f"[INFO] Epoch {epoch:03d} | Train Loss: {loss.item():.4f} | Val Acc: {val_acc:.4f}")
            train_log.append((epoch, loss.item(), val_acc.item()))
print("[INFO] Training complete. Running inference on test set...")

# Inference on test set
model.eval()
with torch.no_grad():
    z = model(data.x, data.edge_index)
    test_pred = edge_classifier(z, test_edge_index, test_edge_attr)

# Compute and save test metrics

# Convert to numpy
true_labels = test_edge_label.cpu().numpy().astype(int)
predicted_probs = test_pred.cpu().numpy()
predicted_labels = (predicted_probs > 0.5).astype(int)

# Compute metrics for undeclared transactions (class 0 as positive)
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, pos_label=0)
recall = recall_score(true_labels, predicted_labels, pos_label=0)
f1 = f1_score(true_labels, predicted_labels, pos_label=0)
roc_auc = roc_auc_score(true_labels, predicted_probs)
pr_auc = average_precision_score(true_labels, predicted_probs, pos_label=0)
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# Display results
print(f"Accuracy:           {accuracy:.4f}")
print(f"Precision (class 0):{precision:.4f}")
print(f"Recall (class 0):   {recall:.4f}")
print(f"F1 Score (class 0): {f1:.4f}")
print(f"ROC AUC:            {roc_auc:.4f}")
print(f"PR AUC (class 0):   {pr_auc:.4f}")
conf_df = pd.DataFrame(
    conf_matrix,
    index=["Actual: Undeclared", "Actual: Declared"],
    columns=["Predicted: Undeclared", "Predicted: Declared"]
)
print("Confusion Matrix:\n", conf_df)

# Save metrics to outputs/test_metrics.txt
metrics_txt_path = os.path.join('outputs', 'test_metrics.txt')
with open(metrics_txt_path, 'w') as f:
    f.write(f"Accuracy:           {accuracy:.4f}\n")
    f.write(f"Precision (class 0):{precision:.4f}\n")
    f.write(f"Recall (class 0):   {recall:.4f}\n")
    f.write(f"F1 Score (class 0): {f1:.4f}\n")
    f.write(f"ROC AUC:            {roc_auc:.4f}\n")
    f.write(f"PR AUC (class 0):   {pr_auc:.4f}\n\n")
    f.write("Confusion Matrix:\n")
    conf_df.to_csv(f, sep='\t')
print(f"[INFO] Test metrics saved to {metrics_txt_path}")

print("[INFO] Saving model and graph to outputs directory...")
# Save results to outputs directory for Azure ML
os.makedirs('outputs', exist_ok=True)
output_model_path = os.path.join('outputs', 'model_and_graph.pt')
torch.save({
    'model_state_dict': model.state_dict(),
    'graph_data': data,
}, output_model_path)
print(f"[INFO] Model and graph saved to {output_model_path}")

print("[INFO] Saving training data to outputs/train_edges.txt ...")
train_txt_path = os.path.join('outputs', 'train_edges.txt')
with open(train_txt_path, 'w') as f:
    # Write header
    attr_dim = train_edge_attr.shape[1] if len(train_edge_attr.shape) > 1 else 1
    header = ["src", "dst", "label"] + [f"attr_{i}" for i in range(attr_dim)]
    f.write("\t".join(header) + "\n")
    # Write each edge
    for i in range(train_edge_index.shape[1]):
        src = train_edge_index[0, i].item()
        dst = train_edge_index[1, i].item()
        label = train_edge_label[i].item()
        attrs = train_edge_attr[i].tolist() if attr_dim > 1 else [train_edge_attr[i].item()]
        row = [str(src), str(dst), str(label)] + [str(a) for a in attrs]
        f.write("\t".join(row) + "\n")
print(f"[INFO] Training data saved to {train_txt_path}")

# Save training log to outputs/train_log.txt
log_txt_path = os.path.join('outputs', 'train_log.txt')
with open(log_txt_path, 'w') as f:
    for epoch, train_loss, val_acc in train_log:
        f.write(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}\n")
print(f"[INFO] Training log saved to {log_txt_path}")

# Model configs
model_configs = [
    ("GraphSAGE", GraphSAGE, {"in_channels": data.num_node_features, "hidden_channels": 64}),
    ("GAT", GAT, {"in_channels": data.num_node_features, "hidden_channels": 64, "heads": 2}),
    ("GCN", GCN, {"in_channels": data.num_node_features, "hidden_channels": 64}),
    ("GIN", GIN, {"in_channels": data.num_node_features, "hidden_channels": 64}),
]

all_metrics = []

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
