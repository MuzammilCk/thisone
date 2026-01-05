
# engine.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, r2_score
import time
import warnings
import joblib 

# Suppress warnings
warnings.filterwarnings('ignore')

class DynamicTrainer:
    def __init__(self, data_path, dataset_dna, hyperparameters, target_col=None, progress_callback=None):
        self.data_path = data_path
        self.dna = dataset_dna
        self.params = hyperparameters
        self.target_col = target_col
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.callback = progress_callback # For the Web UI
        
        # Tracking
        self.train_loss_history = []
        self.val_loss_history = []
        self.training_time = 0

    def prepare_data(self):
        """SCIENTIFICALLY ACCURATE PIPELINE (No Leakage)"""
        df = pd.read_csv(self.data_path)
        
        # 1. Identify Target
        if self.target_col is None: 
            self.target_col = df.columns[-1]
            
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]

        # 2. SPLIT DATA FIRST (The Critical Fix)
        # We split raw data before doing ANY math on it
        X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 3. Define Preprocessing Pipeline
        # Identify numeric and categorical columns
        num_cols = X_train_raw.select_dtypes(include=[np.number]).columns
        cat_cols = X_train_raw.select_dtypes(exclude=[np.number]).columns

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, num_cols),
                ('cat', categorical_transformer, cat_cols)
            ]
        )

        # 4. Fit and Transform
        # Fit ONLY on Training Data
        X_train = self.preprocessor.fit_transform(X_train_raw)
        # Transform Validation Data (using statistics from Train)
        X_val = self.preprocessor.transform(X_val_raw)

        # Save the pipeline for Inference
        joblib.dump(self.preprocessor, 'preprocessing_pipeline.pkl')

        # 5. Target Encoding
        if self.dna['task_type'] == 'classification':
            self.le = LabelEncoder()
            y_train = self.le.fit_transform(y_train_raw)
            # Handle unseen labels in validation conservatively by mapping to a safe default if needed
            # For simplicity in this demo, we assume validation labels are seen, or we use a mask
            # A more robust way is to filter valid indices or use a dedicated Unknown token
            
            # Simple check: map unseen to -1 then filter? 
            # Or just assume standard split covers it for this POC. 
            # Ideally LabelEncoder isn't great for unseen targets.
            
            # Safe transform:
            y_val_np = y_val_raw.to_numpy()
            y_val_masked = np.array([x if x in self.le.classes_ else self.le.classes_[0] for x in y_val_np])
            y_val = self.le.transform(y_val_masked)
            
            self.output_dim = len(self.le.classes_)
            self.criterion = nn.CrossEntropyLoss()
        else:
            y_train = y_train_raw.values.reshape(-1, 1).astype(np.float32)
            y_val = y_val_raw.values.reshape(-1, 1).astype(np.float32)
            self.output_dim = 1
            self.criterion = nn.MSELoss()

        # 6. Tensors
        self.X_train_T = torch.FloatTensor(X_train).to(self.device)
        self.X_val_T = torch.FloatTensor(X_val).to(self.device)
        
        if self.dna['task_type'] == 'classification':
            self.y_train_T = torch.LongTensor(y_train).to(self.device)
            self.y_val_T = torch.LongTensor(y_val).to(self.device)
        else:
            self.y_train_T = torch.FloatTensor(y_train).to(self.device)
            self.y_val_T = torch.FloatTensor(y_val).to(self.device)

        # 7. Loaders
        bs = int(self.params.get('batch_size', 32))
        self.train_loader = DataLoader(TensorDataset(self.X_train_T, self.y_train_T), batch_size=bs, shuffle=True)
        self.val_loader = DataLoader(TensorDataset(self.X_val_T, self.y_val_T), batch_size=bs)
        self.input_dim = X_train.shape[1]

    def build_model(self):
        # Dynamic Architecture Construction
        layers = []
        in_dim = self.input_dim
        
        # MetaTune Intelligence: Deeper networks for higher entropy
        depth = 2 if self.dna.get('target_entropy', 0) < 1.0 else 4
        
        for _ in range(depth):
            out_dim = max(in_dim // 2, 32)
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Dropout(self.params.get('dropout', 0.2))
            ])
            in_dim = out_dim
            
        layers.append(nn.Linear(in_dim, self.output_dim))
        self.model = nn.Sequential(*layers).to(self.device)
        
        # Optimizer from Meta-Brain
        lr = self.params.get('learning_rate', 0.001)
        wd = self.params.get('weight_decay_l2', 0.0)
        
        if self.params.get('optimizer_type', 'adam') == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        else:
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)

    def run(self, epochs=20):
        self.prepare_data()
        self.build_model()
        
        start_time = time.time()
        metric = 0.0
        metric_name = "Metric"

        for epoch in range(epochs):
            # Train Loop
            self.model.train()
            batch_loss = 0
            for X_b, y_b in self.train_loader:
                self.optimizer.zero_grad()
                pred = self.model(X_b)
                loss = self.criterion(pred, y_b)
                loss.backward()
                self.optimizer.step()
                batch_loss += loss.item()
            
            avg_train_loss = batch_loss / len(self.train_loader)
            
            # Validation Loop
            self.model.eval()
            val_loss = 0
            all_preds, all_true = [], []
            with torch.no_grad():
                for X_b, y_b in self.val_loader:
                    pred = self.model(X_b)
                    val_loss += self.criterion(pred, y_b).item()
                    if self.dna['task_type'] == 'classification':
                        all_preds.extend(torch.argmax(pred, 1).cpu().numpy())
                    else:
                        all_preds.extend(pred.cpu().numpy().flatten())
                    all_true.extend(y_b.cpu().numpy().flatten())
            
            avg_val_loss = val_loss / len(self.val_loader)
            
            # Metric Calculation
            if self.dna['task_type'] == 'classification':
                metric = accuracy_score(all_true, all_preds)
                metric_name = "Accuracy"
            else:
                metric = r2_score(all_true, all_preds)
                metric_name = "R2 Score"
                
            # Store History
            self.train_loss_history.append(avg_train_loss)
            self.val_loss_history.append(avg_val_loss)
            
            # UPDATE THE WEB UI (If callback exists)
            if self.callback:
                self.callback(epoch + 1, epochs, avg_train_loss, avg_val_loss, metric)
                time.sleep(0.1) # Visual effect for demo
                
        return {
            "status": "Optimization Complete",
            "final_metric": metric,
            "metric_name": metric_name,
            "training_time": time.time() - start_time
        }
