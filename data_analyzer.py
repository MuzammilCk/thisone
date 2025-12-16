import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, entropy
import sys
import os
import warnings

# Suppress warnings for cleaner output in production
warnings.filterwarnings('ignore')

class DatasetAnalyzer:
    """
    The 'Diagnostic Engine' of MetaTune.
    Role: Scans raw CSV data to extract a 'DNA' profile (Meta-Features) 
          that characterizes the dataset's complexity.
    """
    
    def __init__(self, file_path, target_col=None):
        self.file_path = file_path
        self.target_col = target_col
        self.data = None
        self.meta_features = {}

    def load_data(self):
        """Loads the CSV file into a Pandas DataFrame safely."""
        try:
            if not os.path.exists(self.file_path):
                print(f"❌ Error: File not found at {self.file_path}")
                return False
                
            self.data = pd.read_csv(self.file_path)
            print(f"✓ Success: Loaded '{os.path.basename(self.file_path)}'")
            print(f"  Shape: {self.data.shape[0]} rows x {self.data.shape[1]} cols")
            return True
        except Exception as e:
            print(f"❌ Error loading file: {e}")
            return False

    def analyze(self):
        """Performs forensic analysis on the dataset to extract its 'DNA'."""
        if self.data is None:
            print("❌ No data loaded. Call load_data() first.")
            return None

        # 1. Target Column Detection
        if self.target_col is None:
            self.target_col = self.data.columns[-1]
            print(f"  Auto-detected Target Column: '{self.target_col}'")
        elif self.target_col not in self.data.columns:
            print(f"❌ Error: Target column '{self.target_col}' not found in dataset.")
            return None

        # 2. Feature Separation
        features = self.data.drop(columns=[self.target_col])
        target = self.data[self.target_col]

        num_cols = features.select_dtypes(include=[np.number]).columns
        cat_cols = features.select_dtypes(exclude=[np.number]).columns

        # === DIMENSIONALITY & STRUCTURE ===
        self.meta_features['n_instances'] = len(self.data)
        self.meta_features['n_features'] = len(features.columns)
        self.meta_features['n_numerical'] = len(num_cols)
        self.meta_features['n_categorical'] = len(cat_cols)
        self.meta_features['dimensionality'] = len(features.columns) / len(self.data)

        # === DATA QUALITY ===
        missing_ratio = self.data.isnull().sum().sum() / self.data.size
        zero_ratio = 0.0
        if len(num_cols) > 0:
            zero_ratio = (features[num_cols] == 0).sum().sum() / features[num_cols].size
        
        self.meta_features['missing_ratio'] = missing_ratio
        self.meta_features['sparsity'] = max(missing_ratio, zero_ratio)

        # === STATISTICAL COMPLEXITY (Numerical) ===
        if len(num_cols) > 0:
            num_data = features[num_cols]
            skew_vals = num_data.apply(lambda x: skew(x.dropna()))
            self.meta_features['mean_skewness'] = skew_vals.mean()
            self.meta_features['max_skewness'] = skew_vals.abs().max()

            kurt_vals = num_data.apply(lambda x: kurtosis(x.dropna()))
            self.meta_features['mean_kurtosis'] = kurt_vals.mean()
            
            if len(num_cols) > 1:
                subset = num_data.iloc[:1000]
                corr_matrix = subset.corr().abs()
                np.fill_diagonal(corr_matrix.values, 0)
                self.meta_features['avg_correlation'] = corr_matrix.mean().mean()
                self.meta_features['max_correlation'] = corr_matrix.max().max()
            else:
                self.meta_features['avg_correlation'] = 0.0
                self.meta_features['max_correlation'] = 0.0
                
            cv = (num_data.std() / (num_data.mean().abs() + 1e-9)).mean()
            self.meta_features['coefficient_variation'] = cv
        else:
            for key in ['mean_skewness', 'max_skewness', 'mean_kurtosis', 
                       'avg_correlation', 'max_correlation', 'coefficient_variation']:
                self.meta_features[key] = 0.0

        # === CATEGORICAL COMPLEXITY ===
        if len(cat_cols) > 0:
            self.meta_features['avg_cardinality'] = features[cat_cols].nunique().mean()
            self.meta_features['max_cardinality'] = features[cat_cols].nunique().max()
        else:
            self.meta_features['avg_cardinality'] = 0.0
            self.meta_features['max_cardinality'] = 0.0

        # === TARGET DIFFICULTY (Task Profiling) ===
        n_unique_target = target.nunique()
        is_classification = (n_unique_target < 20) or (target.dtype == 'object')
        
        if is_classification:
            self.meta_features['task_type'] = 'classification'
            self.meta_features['n_classes'] = n_unique_target
            counts = target.value_counts()
            self.meta_features['class_imbalance_ratio'] = counts.max() / counts.min()
            prob_dist = counts / counts.sum()
            self.meta_features['target_entropy'] = entropy(prob_dist)
            max_ent = np.log(n_unique_target) if n_unique_target > 1 else 1
            self.meta_features['normalized_entropy'] = self.meta_features['target_entropy'] / max_ent
            self.meta_features['minority_class_pct'] = counts.min() / len(target)
        else:
            self.meta_features['task_type'] = 'regression'
            self.meta_features['n_classes'] = 0
            self.meta_features['class_imbalance_ratio'] = 0.0
            self.meta_features['target_entropy'] = 0.0
            self.meta_features['normalized_entropy'] = 0.0
            self.meta_features['minority_class_pct'] = 0.0
            
            if pd.api.types.is_numeric_dtype(target):
                self.meta_features['target_skewness'] = skew(target.dropna())
                self.meta_features['target_cv'] = target.std() / (abs(target.mean()) + 1e-9)

        return self.meta_features

    def print_summary(self):
        """Prints a formatted summary of results."""
        if not self.meta_features: return
        print("\n" + "="*60 + "\nDATASET DNA (Meta-Features)\n" + "="*60)
        def _print_row(key, val):
            print(f"  {key:25s}: {val:.4f}" if isinstance(val, float) else f"  {key:25s}: {val}")
        print("Dimensions:")
        _print_row("Instances", self.meta_features['n_instances'])
        _print_row("Features", self.meta_features['n_features'])
        print("\nComplexity Metrics:")
        _print_row("Target Entropy", self.meta_features.get('target_entropy', 0))
        _print_row("Mean Skewness", self.meta_features.get('mean_skewness', 0))
        _print_row("Sparsity/Missing", self.meta_features.get('sparsity', 0))
        _print_row("Imbalance Ratio", self.meta_features.get('class_imbalance_ratio', 0))
        print("="*60)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\n❌ USAGE ERROR\nUsage: python dataset_analyzer.py <path_to_csv> [target_column]")
        sys.exit(1)
    file_path = sys.argv[1]
    target_col = sys.argv[2] if len(sys.argv) > 2 else None
    print("\n" + "="*60 + f"\n🧬 ANALYZING DATASET: {os.path.basename(file_path)}\n" + "="*60)
    analyzer = DatasetAnalyzer(file_path, target_col=target_col)
    if analyzer.load_data():
        dna = analyzer.analyze()
        analyzer.print_summary()
        print("\n✅ Analysis Complete. DNA ready for Meta-Brain.")
