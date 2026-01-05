
import pandas as pd
import numpy as np

def generate_audit_data(n_rows=2000):
    print("🧪 Generating 'audit_data.csv' (The Stress Test)...")
    np.random.seed(42)
    
    data = {}
    
    # 1. Standard Numerical (baseline)
    data['feat_normal'] = np.random.randn(n_rows)
    
    # 2. Extreme Outliers (Skewness)
    outliers = np.random.randn(n_rows)
    outliers[0:10] = 1000000.0 # Extreme right tail
    data['feat_outlier'] = outliers
    
    # 3. Missing Values (Sparsity)
    sparse = np.random.randn(n_rows)
    sparse[np.random.rand(n_rows) < 0.3] = np.nan # 30% missing
    data['feat_missing'] = sparse
    
    # 4. Low Variance / Constant (Should be ignored or handled)
    data['feat_constant'] = np.ones(n_rows)
    
    # 5. High Cardinality Categorical (Stress test OneHotEncoder memory)
    # 500 unique values
    data['feat_high_card'] = [f"cat_{i%500}" for i in range(n_rows)]
    
    # 6. New Category in Validation (Simulation)
    # We can't simulate runtime future data here easily in one file, 
    # but the engine split will handle it if we ensure unique values exist.
    
    # 7. Mixed Type / Garbage (Robustness)
    # Numbers as strings
    data['feat_mixed'] = [str(x) for x in np.random.randint(0, 100, n_rows)]
    
    # 8. Correlated Features (Multicollinearity)
    data['feat_collinear_1'] = data['feat_normal'] * 2 + np.random.normal(0, 0.1, n_rows)
    
    # 9. Class Imbalance (Target)
    # 95% Class 0, 5% Class 1
    target_probs = [0.95, 0.05]
    data['target_audit'] = np.random.choice([0, 1], size=n_rows, p=target_probs)
    
    # Target Distortion (Manual) to ensure complex relationship
    # If feat_outlier is high -> Class 1
    # If feat_missing is NaN -> Class 0
    mask_outlier = data['feat_outlier'] > 100
    data['target_audit'][mask_outlier] = 1
    
    df = pd.DataFrame(data)
    
    # Save
    df.to_csv("audit_data.csv", index=False)
    print(f"✅ Generated 'audit_data.csv' with {n_rows} rows and {len(df.columns)} columns.")
    print("   - Includes: Nulls, Outliers, High Card, Imbalance, Constant Cols.")


