import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, r2_score
import time
import warnings
from collections import deque

warnings.filterwarnings('ignore')

# ==========================================
# 1. THE MONITOR (Your "Priority 2" Logic)
# ==========================================
class AdaptiveTrainingMonitor:
    """
    Real-time Training Stability Analyzer.
    Detects Plateaus, Overfitting, and Gradient Anomalies.
    """
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.train_loss_buffer = deque(maxlen=window_size)
        self.val_loss_buffer = deque(maxlen=window_size)
        self.gradient_norm_buffer = deque(maxlen=window_size)
        self.plateau_count = 0
        self.overfitting_count = 0
        
    def update(self, train_loss, val_loss, gradient_norm):
        self.train_loss_buffer.append(train_loss)
        self.val_loss_buffer.append(val_loss)
        self.gradient_norm_buffer.append(gradient_norm)
    
    def detect_plateau(self, threshold=0.001):
        if len(self.train_loss_buffer) < self.window_size: return False
        loss_change = abs(self.train_loss_buffer[-1] - self.train_loss_buffer[0])
        is_plateau = loss_change < threshold
        if is_plateau: self.plateau_count += 1
        else: self.plateau_count = 0
        return self.plateau_count >= 2
    
    def detect_overfitting(self, threshold=0.05):
        if len(self.train_loss_buffer) < self.window_size: return False
        # Val loss up AND Train loss down
        val_trend = self.val_loss_buffer[-1] - self.val_loss_buffer[0]
        train_trend = self.train_loss_buffer[-1] - self.train_loss_buffer[0]
        gap = self.val_loss_buffer[-1] - self.train_loss_buffer[-1]
        
        is_overfitting = (val_trend > 0) and (train_trend < 0) and (gap > threshold)
        if is_overfitting: self.overfitting_count += 1
        else: self.overfitting_count = 0
        return self.overfitting_count >= 2

    def detect_vanishing_gradients(self, threshold=1e-4):
        if len(self.gradient_norm_buffer) < 3: return False
        return np.mean(self.gradient_norm_buffer) < threshold

# ==========================================
# 2. THE TRAINER (Merged with Streaming)
# ==========================================
class DynamicTrainer:
    def __init__(self, data_path, dataset_dna, hyperparameters, target_col=None):
        self.data_path = data_path
        self.dna = dataset_dna
        self.params = hyperparameters
        self.target_col = target_col
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize Monitor
        self.monitor = AdaptiveTrainingMonitor()
        self.logs = [] # Store adaptation messages

    def prepare_data(self):
        """Leakage-Free Data Prep with Categorical Handling"""
        df = pd.read_csv(self.data_path)
        if self.target_col is None: self.target_col = df.columns[-1]
        
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]

        # 1. SPLIT DATA FIRST (The Critical Fix)
        # We split raw data before doing ANY math on it
        X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 2. Process Numerical Columns
        # We learn the median/mean ONLY from the Training set
        num_cols = X_train_raw.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            medians = X_train_raw[num_cols].median()
            X_train_raw[num_cols] = X_train_raw[num_cols].fillna(medians)
            X_val_raw[num_cols] = X_val_raw[num_cols].fillna(medians)

        # 3. Process Categorical Columns
        cat_cols = X_train_raw.select_dtypes(exclude=[np.number]).columns
        if len(cat_cols) > 0:
            for col in cat_cols:
                mode = X_train_raw[col].mode()[0]
                X_train_raw[col] = X_train_raw[col].fillna(mode)
                X_val_raw[col] = X_val_raw[col].fillna(mode)
                
                # Simple One-Hot Encoding
                X_train_raw = pd.get_dummies(X_train_raw, columns=[col], drop_first=True)
                X_val_raw = pd.get_dummies(X_val_raw, columns=[col], drop_first=True)

        # Align columns (ensure train/val have same dummy columns)
        X_train_raw, X_val_raw = X_train_raw.align(X_val_raw, join='left', axis=1, fill_value=0)

        # 4. Scaling (Fit on Train, Transform on Val)
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train_raw)
        X_val = self.scaler.transform(X_val_raw)

        # 5. Target Encoding
        if self.dna['task_type'] == 'classification':
            self.le = LabelEncoder()
            y_train = self.le.fit_transform(y_train_raw)
            # Handle unseen labels in validation conservatively
            y_val_raw = y_val_raw.map(lambda s: s if s in self.le.classes_ else self.le.classes_[0])
            y_val = self.le.transform(y_val_raw)
            self.output_dim = len(self.le.classes_)
            self.criterion = nn.CrossEntropyLoss()
        else:
            y_train = y_train_raw.values.reshape(-1, 1)
            y_val = y_val_raw.values.reshape(-1, 1)
            self.output_dim = 1
            self.criterion = nn.MSELoss()

        # 6. Tensors
        dtype_y = torch.LongTensor if self.dna['task_type'] == 'classification' else torch.FloatTensor
        
        self.train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train).to(self.device), dtype_y(y_train).to(self.device)),
            batch_size=int(self.params['batch_size']), shuffle=True
        )
        self.val_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_val).to(self.device), dtype_y(y_val).to(self.device)),
            batch_size=int(self.params['batch_size'])
        )
        self.input_dim = X_train.shape[1]

    def build_model(self):
        layers = []
        in_dim = self.input_dim
        # MetaTune Intelligence: Deeper for high entropy
        depth = 4 if self.dna['target_entropy'] > 1.2 else 2
        
        for _ in range(depth):
            out_dim = max(in_dim // 2, 32)
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Dropout(self.params['dropout'])
            ])
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, self.output_dim))
        self.model = nn.Sequential(*layers).to(self.device)
        
        # Optimizer
        self.current_lr = self.params['learning_rate']
        self.current_wd = self.params['weight_decay_l2']
        
        if self.params['optimizer_type'] == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.current_lr, weight_decay=self.current_wd)
        else:
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.current_lr, momentum=0.9, weight_decay=self.current_wd)

    def _update_optimizer(self):
        """Applies new hyperparameters to the optimizer in real-time"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.current_lr
            param_group['weight_decay'] = self.current_wd

    def _compute_gradient_norm(self):
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

    def run(self, epochs=50):
        """
        Runs the training loop and YIELDS data for the UI.
        Contains the Logic for Dynamic Adaptation.
        """
        self.prepare_data()
        self.build_model()
        
        for epoch in range(epochs):
            # 1. Training Step
            self.model.train()
            batch_losses = []
            
            for X_b, y_b in self.train_loader:
                self.optimizer.zero_grad()
                out = self.model(X_b)
                loss = self.criterion(out, y_b)
                loss.backward()
                self.optimizer.step()
                batch_losses.append(loss.item())
            
            train_loss = np.mean(batch_losses)
            grad_norm = self._compute_gradient_norm() # Monitor Gradients
            
            # 2. Validation Step
            self.model.eval()
            val_losses = []
            preds, true = [], []
            with torch.no_grad():
                for X_b, y_b in self.val_loader:
                    out = self.model(X_b)
                    val_losses.append(self.criterion(out, y_b).item())
                    if self.dna['task_type'] == 'classification':
                        preds.extend(torch.argmax(out, 1).cpu().numpy())
                    else:
                        preds.extend(out.cpu().numpy().flatten())
                    true.extend(y_b.cpu().numpy().flatten())
            
            val_loss = np.mean(val_losses)
            metric = accuracy_score(true, preds) if self.dna['task_type'] == 'classification' else r2_score(true, preds)
            
            # 3. DYNAMIC ADAPTATION (The "Real-Time Feedback" Promise)
            self.monitor.update(train_loss, val_loss, grad_norm)
            adaptation_msg = None
            
            # Case A: Plateau -> Decrease Learning Rate (Fine-tuning)
            if self.monitor.detect_plateau():
                old_lr = self.current_lr
                self.current_lr *= 0.5
                self.current_lr = max(self.current_lr, 1e-5)
                self._update_optimizer()
                adaptation_msg = f"📉 Plateau Detected: Reduced LR to {self.current_lr:.1e}"
            
            # Case B: Overfitting -> Increase Regularization (Robustness)
            elif self.monitor.detect_overfitting():
                old_wd = self.current_wd
                self.current_wd *= 2.0 
                self.current_wd = min(self.current_wd, 0.1)
                self._update_optimizer()
                adaptation_msg = f"🛡️ Overfitting Risk: Increased L2 Reg to {self.current_wd:.1e}"
            
            # Case C: Vanishing Gradients -> Boost LR (Acceleration)
            elif self.monitor.detect_vanishing_gradients():
                self.current_lr *= 1.5
                self._update_optimizer()
                adaptation_msg = f"⚡ Vanishing Gradients: Boosted LR to {self.current_lr:.1e}"

            # 4. Yield Data to Dashboard
            # We send everything the UI needs to visualize the "Thinking" process
            yield {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "metric": metric,
                "current_l2": self.current_wd, # Mapped to Yellow Graph
                "current_lr": self.current_lr,
                "grad_norm": grad_norm,
                "adaptation": adaptation_msg
            }
