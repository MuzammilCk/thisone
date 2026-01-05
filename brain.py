
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# === HYBRID ARCHITECTURE: Attention + ResNet ===
class SelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim):
        super(SelfAttentionBlock, self).__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** -0.5

    def forward(self, x):
        Q = self.query(x); K = self.key(x); V = self.value(x)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        context = torch.matmul(attention_weights, V)
        return context + x  # Residual connection

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout_rate=0.2):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.GELU()

    def forward(self, x):
        residual = x
        out = self.activation(self.bn1(self.linear1(x)))
        out = self.dropout(out)
        out = self.bn2(self.linear2(out))
        out += residual
        return self.activation(out)

class AdvancedMetaNet(nn.Module):
    def __init__(self, input_dim):
        super(AdvancedMetaNet, self).__init__()
        hidden_dim = 128
        self.embedding = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.GELU())
        self.attention = SelfAttentionBlock(hidden_dim)
        self.backbone = nn.Sequential(ResidualBlock(hidden_dim), ResidualBlock(hidden_dim), nn.Linear(hidden_dim, 64), nn.GELU())
        self.head_lr = nn.Linear(64, 1)
        self.head_reg = nn.Linear(64, 1)
        self.head_batch = nn.Linear(64, 1)
        self.head_drop = nn.Linear(64, 1)
        self.head_optim = nn.Linear(64, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.attention(x)
        feat = self.backbone(x)
        return torch.cat([self.head_lr(feat), self.head_reg(feat), self.head_batch(feat), self.head_drop(feat), self.head_optim(feat)], dim=1)

# === META-LEARNER CONTROLLER ===
class MetaLearner:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_features = ['n_instances', 'n_features', 'n_numerical', 'n_categorical', 'dimensionality', 'missing_ratio', 'mean_skewness', 'max_skewness', 'mean_kurtosis', 'avg_correlation', 'max_correlation', 'coefficient_variation', 'avg_cardinality', 'class_imbalance_ratio', 'target_entropy', 'normalized_entropy', 'sparsity']
        self.output_params = ['learning_rate', 'weight_decay_l2', 'batch_size', 'dropout', 'optimizer_type']
        self.scaler = StandardScaler()
        self.model = None
        self.is_trained = False
        self.knowledge_base_path = "knowledge_base.csv"
        print(f"🧠 Meta-Learner Brain initialized on {self.device}")

    def _bootstrap_heuristics(self, dna):
        """Step A: Cold Start Heuristics (The Old 'Brain') used when no data exists."""
        print("🧊 Cold Start: Using Heuristics to bootstrap...")
        lr = np.clip(0.005 * np.exp(-dna.get('target_entropy', 1.0)), 0.0001, 0.01)
        l2 = 1e-3 * dna.get('mean_skewness', 0) if dna.get('mean_skewness', 0) > 2.0 else 1e-5
        l2 = np.clip(l2, 1e-6, 0.1)
        
        n_inst = dna.get('n_instances', 1000)
        bs = 128 if n_inst > 10000 else (64 if n_inst > 5000 else 32)
        if dna.get('class_imbalance_ratio', 1) > 5: bs = 16
        
        drop = np.clip(0.1 + (dna.get('sparsity', 0) * 0.4), 0.0, 0.5)
        optim_type = 'adam' if dna.get('target_entropy', 0) > 0.5 else 'sgd' # Check this logic later, simpler is often better for simple tasks
        
        return {
            'learning_rate': float(lr),
            'weight_decay_l2': float(l2),
            'batch_size': int(bs),
            'dropout': float(drop),
            'optimizer_type': optim_type
        }

    def store_experience(self, dna, hyperparameters, metric):
        """Step B: The Feedback Loop - Save run results to Knowledge Base."""
        # Flatten everything into one row
        row = dna.copy()
        row.update(hyperparameters)
        row['final_metric'] = metric
        
        # Optimizer mapping for regression readiness
        row['optimizer_type_code'] = 1 if hyperparameters['optimizer_type'] == 'adam' else 0
        
        df_row = pd.DataFrame([row])
        
        if not os.path.exists(self.knowledge_base_path):
            df_row.to_csv(self.knowledge_base_path, index=False)
        else:
            # Reorder columns to match existing file
            try:
                # Read only header
                existing_columns = pd.read_csv(self.knowledge_base_path, nrows=0).columns.tolist()
                # Add missing columns with default 0
                for col in existing_columns:
                    if col not in df_row.columns:
                        df_row[col] = 0
                # Ensure order matches
                df_row = df_row[existing_columns]
                df_row.to_csv(self.knowledge_base_path, mode='a', header=False, index=False)
            except pd.errors.EmptyDataError:
                # File exists but empty?
                df_row.to_csv(self.knowledge_base_path, index=False)
        print(f"💾 Experience stored to '{self.knowledge_base_path}'")

    def train(self, epochs=50):
        """Step C: Evolutionary Selection (Survival of the Fittest)."""
        if not os.path.exists(self.knowledge_base_path):
            print("⚠️ No Knowledge Base found. Skipping training.")
            return

        df = pd.read_csv(self.knowledge_base_path)
        if len(df) < 5:
            print(f"⚠️ Not enough data to train (Found {len(df)} records, need 5). Using heuristics.")
            return

        # === EVOLUTIONARY SELECTION ===
        # Filter for the Top 50% of runs based on final_metric
        # We assume higher metric is better (Accuracy, R2).
        median_perf = df['final_metric'].median()
        df_elite = df[df['final_metric'] >= median_perf]
        
        print(f"\n🎓 Training Meta-Brain on ELITE History ({len(df_elite)}/{len(df)} records > {median_perf:.4f})...")
        
        if len(df_elite) < 2:
            df_elite = df # Fallback if filtering is too aggressive
        
        # Prepare Data
        for f in self.input_features: 
            if f not in df_elite.columns: df_elite[f] = 0
            
        X = self.scaler.fit_transform(df_elite[self.input_features].values)
        
        target_cols = ['learning_rate', 'weight_decay_l2', 'batch_size', 'dropout', 'optimizer_type_code']
        y = df_elite[target_cols].values
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        self.model = AdvancedMetaNet(input_dim=X.shape[1]).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=0.003)
        criterion = nn.MSELoss()
        
        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            preds = self.model(X_tensor)
            loss = criterion(preds, y_tensor)
            loss.backward()
            optimizer.step()
            
        self.is_trained = True
        print(f"✅ Brain Training Complete on Elite Data (Loss: {loss.item():.4f})")

    def predict(self, dataset_dna):
        """Step D: Prediction with Evolutionary Exploration (Mutation)."""
        if not self.is_trained:
            if os.path.exists("meta_brain_weights.pth"):
                pass 
            return self._bootstrap_heuristics(dataset_dna)

        feats = [dataset_dna.get(f, 0) for f in self.input_features]
        X = self.scaler.transform(np.array(feats).reshape(1, -1))
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            raw_preds = self.model(X_tensor).cpu().numpy()[0]
            
        # === EVOLUTIONARY MUTATION (EXPLORATION) ===
        # Add 10% Gaussian noise to encourage exploring new hyperparameters
        # This prevents getting stuck in local optima
        noise = np.random.normal(0, 0.1, size=raw_preds.shape)
        raw_preds += noise
            
        return {
            'learning_rate': float(np.abs(raw_preds[0])), 
            'weight_decay_l2': float(np.abs(raw_preds[1])),
            'batch_size': int(np.clip(raw_preds[2], 16, 256)), 
            'dropout': float(np.clip(np.abs(raw_preds[3]), 0, 0.5)),
            'optimizer_type': 'adam' if raw_preds[4] > 0.5 else 'sgd'
        }
        
    def save(self, path="meta_brain.pkl"):
        with open(path, 'wb') as f: pickle.dump(self, f)
    
    @staticmethod
    def load(path="meta_brain.pkl"):
        with open(path, 'rb') as f: return pickle.load(f)


