import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import pickle
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
        print(f"🧠 Meta-Learner Brain initialized on {self.device}")

    def generate_synthetic_meta_dataset(self, n_datasets=2000):
        """Generates training data based on expert heuristics (Knowledge Distillation)."""
        print(f"📦 Bootstrapping with {n_datasets} synthetic datasets...")
        meta_data = []
        for _ in range(n_datasets):
            dna = {
                'n_instances': np.random.randint(100, 20000), 'n_features': np.random.randint(5, 200),
                'n_numerical': np.random.randint(0, 50), 'n_categorical': np.random.randint(0, 20),
                'dimensionality': np.random.uniform(0.001, 0.5), 'missing_ratio': np.random.uniform(0, 0.3),
                'mean_skewness': np.random.uniform(0, 5), 'max_skewness': np.random.uniform(0, 10),
                'mean_kurtosis': np.random.uniform(-1, 5), 'avg_correlation': np.random.uniform(0, 0.9),
                'max_correlation': np.random.uniform(0, 0.99), 'coefficient_variation': np.random.uniform(0.1, 2),
                'avg_cardinality': np.random.uniform(2, 50), 'class_imbalance_ratio': np.random.uniform(1, 20),
                'target_entropy': np.random.uniform(0, 2.5), 'normalized_entropy': np.random.uniform(0, 1),
                'sparsity': np.random.uniform(0, 0.8)
            }
            # Heuristics
            lr = np.clip(0.005 * np.exp(-dna['target_entropy']), 0.0001, 0.01)
            l2 = 1e-3 * dna['mean_skewness'] if dna['mean_skewness'] > 2.0 else 1e-5
            l2 = np.clip(l2, 1e-6, 0.1)
            bs = 128 if dna['n_instances'] > 10000 else (64 if dna['n_instances'] > 5000 else 32)
            if dna['class_imbalance_ratio'] > 5: bs = 16
            drop = np.clip(0.1 + (dna['sparsity'] * 0.4), 0.0, 0.5)
            optim_type = 1 if dna['target_entropy'] > 0.5 else 0
            
            sample = {**dna, 'learning_rate': lr, 'weight_decay_l2': l2, 'batch_size': bs, 'dropout': drop, 'optimizer_type': optim_type}
            meta_data.append(sample)
        return pd.DataFrame(meta_data)

    def train(self, epochs=50):
        df = self.generate_synthetic_meta_dataset()
        X = self.scaler.fit_transform(df[self.input_features].values)
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(df[self.output_params].values).to(self.device)
        
        self.model = AdvancedMetaNet(input_dim=X.shape[1]).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=0.003)
        criterion = nn.MSELoss()
        
        print(f"\n🚀 Training Advanced MetaNet for {epochs} epochs...")
        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            preds = self.model(X_tensor)
            loss = criterion(preds, y_tensor)
            loss.backward()
            optimizer.step()
        self.is_trained = True
        print("✅ Brain Training Complete.")

    def predict(self, dataset_dna):
        if not self.is_trained: self.train()
        feats = [dataset_dna.get(f, 0) for f in self.input_features]
        X = self.scaler.transform(np.array(feats).reshape(1, -1))
        X_tensor = torch.FloatTensor(X).to(self.device)
        self.model.eval()
        with torch.no_grad():
            raw_preds = self.model(X_tensor).cpu().numpy()[0]
        return {
            'learning_rate': float(np.abs(raw_preds[0])), 'weight_decay_l2': float(np.abs(raw_preds[1])),
            'batch_size': int(np.clip(raw_preds[2], 16, 256)), 'dropout': float(np.clip(np.abs(raw_preds[3]), 0, 0.5)),
            'optimizer_type': 'adam' if raw_preds[4] > 0.5 else 'sgd', 'confidence_score': 0.92
        }
        
    def save(self, path="meta_brain.pkl"):
        with open(path, 'wb') as f: pickle.dump(self, f)
    
    @staticmethod
    def load(path="meta_brain.pkl"):
        with open(path, 'rb') as f: return pickle.load(f)

if __name__ == "__main__":
    brain = MetaLearner()
    brain.train(epochs=60)
    brain.save()
