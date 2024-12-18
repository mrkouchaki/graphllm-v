#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import json
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
import torch.nn as nn

class HeteroGraphAttentionLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, edge_types):
        super(HeteroGraphAttentionLayer, self).__init__()
        self.edge_types = edge_types
        self.attn_layers = nn.ModuleDict({
            etype: nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)
            for etype in edge_types
        })
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
        )
        
    def forward(self, G, h):
        outputs = {}
        for etype in self.edge_types:
            subgraph = G.edge_type_subgraph([etype])
            h_src = h[subgraph.ndata[dgl.NTYPE] == subgraph.ntypes[0]]
            h_dst = h[subgraph.ndata[dgl.NTYPE] == subgraph.ntypes[1]]
            attn_layer = self.attn_layers[etype]
            h_updated, _ = attn_layer(h_src.unsqueeze(1), h_dst.unsqueeze(1))        
        h_combined = torch.cat([outputs[etype] for etype in outputs], dim=-1)

        ff_output = self.feed_forward(h_combined)
        return h_combined + ff_output

class HeteroGraphTransformer(nn.Module):
    def __init__(self, hidden_dim, num_classes, num_heads, num_layers, edge_types):
        super(HeteroGraphTransformer, self).__init__()
        self.token_embedding = BertModel.from_pretrained('bert-base-uncased')
        self.transformer_layers = nn.ModuleList([
            HeteroGraphAttentionLayer(hidden_dim, num_heads, edge_types)
        ])
        self.classifier = nn.Linear(hidden_dim * len(edge_types), num_classes)
        
    def forward(self, input_ids, attention_mask, G, h):
        x = self.token_embedding(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        logits = self.classifier(cls_representation)
        return logits

    
class Transformer(nn.Module):
    def __init__(self, hidden_dim, num_classes, num_heads=8, num_layers=6, max_distance=5):
        super(GraphAwareTransformer, self).__init__()
        self.token_embedding = BertModel.from_pretrained('bert-base-uncased')
        self.transformer_layers = nn.ModuleList([
            GraphAttentionLayer(hidden_dim, num_heads, max_distance)
            for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, input_ids, attention_mask, distance_matrix):
        x = self.token_embedding(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        for layer in self.transformer_layers:
            x = layer(x, attention_mask, distance_matrix)
        cls_representation = x[:, 0, :]
        logits = self.classifier(cls_representation)
        return logits

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, max_distance):
        super(GraphAttentionLayer, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)
        self.distance_embeddings = nn.Embedding(max_distance + 2, hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
        )
    
    def forward(self, x, attention_mask, distance_matrix):
        # x shape: [batch_size, seq_length, hidden_dim]
        batch_size, seq_length, hidden_dim = x.size()
        x = x.transpose(0, 1)
        distance_embeddings = self.distance_embeddings(distances)
        

        attn_output, _ = self.multihead_attn(
            x, x, x,
            key_padding_mask=~attention_mask.bool(),
            need_weights=False
        )
        x = x + attn_output
        ff_output = self.feed_forward(x)
        x = x + ff_output
        x = self.layer_norm(x)
        x = x.transpose(0, 1) 
        return x

    

input_ids = encoded_inputs['input_ids']
attention_mask = encoded_inputs['attention_mask']


dataset = TensorDataset(input_ids, attention_mask, labels)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=8)
test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=8)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HeteroGraphTransformer(
    hidden_dim=768,
    num_classes=2,
    num_heads=8,
    num_layers=2,
    edge_types=[etype[1] for etype in G.canonical_etypes]  # Extract relation names only
)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()


for ntype in G.ntypes:
    G.nodes[ntype].data['feat'] = torch.randn((G.num_nodes(ntype), hidden_dim))  # Replace with actual features if available


for epoch in range(1, 4):
    print(f"Epoch {epoch} starting...")
    model.train()
    total_loss = 0
    
    # Training
    for batch_idx, batch in enumerate(train_dataloader, 1):
        optimizer.zero_grad()
        
        batch_input_ids, batch_attention_mask, batch_labels = [b.to(device) for b in batch]
        print(f"Batch {batch_idx} | Input IDs shape: {batch_input_ids.shape} | Attention Mask shape: {batch_attention_mask.shape}")
        

        outputs = model(batch_input_ids, batch_attention_mask, G, G.ndata['feat'])
        loss = criterion(outputs, batch_labels)
        

        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        print(f"Batch {batch_idx} | Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(train_dataloader)
    print(f'Epoch {epoch} completed. Average Loss: {avg_loss:.4f}')
    
    model.eval()
    total_correct = 0
    total_count = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataloader, 1):
            batch_input_ids, batch_attention_mask, batch_labels = [b.to(device) for b in batch]
            outputs = model(batch_input_ids, batch_attention_mask, G, G.ndata['feat'])
            predictions = torch.argmax(outputs, dim=1)
            total_correct += (predictions == batch_labels).sum().item()
            total_count += batch_labels.size(0)
            print(f"Test Batch {batch_idx} | Correct Predictions: {(predictions == batch_labels).sum().item()} / {batch_labels.size(0)}")
    
    accuracy = total_correct / total_count
    print(f'Test Accuracy after Epoch {epoch}: {accuracy:.4f}')

    


plt.figure(figsize=(8, 5))
plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', label="Training Loss", color='blue')
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.title("Training Loss Over Epochs", fontsize=14)
plt.legend(fontsize=12)
plt.grid(alpha=0.5, linestyle='--')
plt.show()


plt.figure(figsize=(8, 5))
plt.plot(range(1, len(eval_losses) + 1), eval_losses, marker='x', linestyle='--', label="Evaluation Loss", color='red')
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.title("Evaluation Loss Over Epochs", fontsize=14)
plt.legend(fontsize=12)
plt.grid(alpha=0.5, linestyle='--')
plt.show()


plt.figure(figsize=(8, 5))
plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, marker='o', label="Training Accuracy", color='green')
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.title("Training Accuracy Over Epochs", fontsize=14)
plt.legend(fontsize=12)
plt.grid(alpha=0.5, linestyle='--')
plt.show()


plt.figure(figsize=(8, 5))
plt.plot(range(1, len(eval_accuracies) + 1), eval_accuracies, marker='x', linestyle='--', label="Evaluation Accuracy", color='purple')
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.title("Evaluation Accuracy Over Epochs", fontsize=14)
plt.legend(fontsize=12)
plt.grid(alpha=0.5, linestyle='--')
plt.show()




cm = confusion_matrix(all_labels_roc, predicted_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.title("Confusion Matrix (Predicted Classes)", fontsize=14)
plt.show()

cm2 = confusion_matrix(all_labels, predicted_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm2, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.title("Confusion Matrix (Binary Predictions)", fontsize=14)
plt.show()




fpr, tpr, thresholds = roc_curve(all_labels_roc, all_preds_roc)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC-ROC = {roc_auc:.4f}", color='tab:blue')
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.title("ROC Curve", fontsize=14)
plt.legend(fontsize=12)
plt.grid(alpha=0.5, linestyle='--')
plt.show()



metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]
scores = [eval_accuracies[-1], precision, recall, f1, roc_auc]

plt.figure(figsize=(8, 5))
sns.barplot(x=metrics, y=scores, palette="viridis")
plt.ylim(0, 1)
plt.ylabel("Score", fontsize=12)
plt.title("Evaluation Metrics", fontsize=14)
plt.show()


epochs = 100
best_eval_loss = float('inf')  # saving the best model
for epoch in range(epochs):
    model.train()
    total_train_loss = 0
    correct_train = 0
    total_train = 0
    for _, input_ids, attention_mask, labels in train_dataloader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(G, input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

        # Accuracy tracking
        preds = torch.argmax(logits, dim=1)
        correct_train += (preds == labels).sum().item()
        total_train += labels.size(0)

    avg_train_loss = total_train_loss / len(train_dataloader)
    train_accuracy = correct_train / total_train

    train_losses.append(avg_train_loss)
    train_accuracies.append(train_accuracy)

    print(f"Epoch {epoch + 1}: Training Loss: {avg_train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}")
    
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', label="Training Loss", color='blue')
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.title("Training Loss Over Epochs", fontsize=14)
plt.legend(fontsize=12)
plt.grid(alpha=0.5, linestyle='--')
plt.show()


plt.figure(figsize=(8, 5))
plt.plot(range(1, len(eval_losses) + 1), eval_losses, marker='x', linestyle='--', label="Evaluation Loss", color='red')
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.title("Evaluation Loss Over Epochs", fontsize=14)
plt.legend(fontsize=12)
plt.grid(alpha=0.5, linestyle='--')
plt.show()


plt.figure(figsize=(8, 5))
plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, marker='o', label="Training Accuracy", color='green')
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.title("Training Accuracy Over Epochs", fontsize=14)
plt.legend(fontsize=12)
plt.grid(alpha=0.5, linestyle='--')
plt.show()


plt.figure(figsize=(8, 5))
plt.plot(range(1, len(eval_accuracies) + 1), eval_accuracies, marker='x', linestyle='--', label="Evaluation Accuracy", color='purple')
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.title("Evaluation Accuracy Over Epochs", fontsize=14)
plt.legend(fontsize=12)
plt.grid(alpha=0.5, linestyle='--')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




