
import unittest
import pandas as pd
import numpy as np
import os
import sys
import shutil
from data_analyzer import DatasetAnalyzer
from brain import MetaLearner
from pipeline import MetaTunePipeline
import warnings
import io
import contextlib

# Force UTF-8 for Windows Console
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

class TestMetaTuneRigorous(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        print("\n" + "="*60)
        print("🛠️  STARTING RIGOROUS QA SUITE")
        print("="*60)
        # Cleanup
        for f in ["audit_data.csv", "knowledge_base.csv", "meta_brain.pkl", "preprocessing_pipeline.pkl"]:
            if os.path.exists(f): os.remove(f)

    def test_01_stress_data_generation(self):
        """Test 1: Can we generate the 'Dataset From Hell'?"""
        print("\n🧪 Test 01: Generating Stress Data...")
        try:
            import generate_audit_data
            generate_audit_data.generate_audit_data(n_rows=500)
            self.assertTrue(os.path.exists("audit_data.csv"))
            df = pd.read_csv("audit_data.csv")
            self.assertEqual(len(df), 500)
            print("   ✅ Stress Data Generated.")
        except Exception as e:
            self.fail(f"❌ Generation Failed: {e}")

    def test_02_pipeline_robustness(self):
        """Test 2: Can the pipeline handle the stress data without crashing?"""
        print("\n🧪 Test 02: Pipeline Robustness (Audit Data)...")
        # Redirect stdout to suppress logs during test
        try:
            pipeline = MetaTunePipeline("audit_data.csv", target_col="target_audit")
            results = pipeline.run(epochs=3)
            self.assertIsNotNone(results)
            self.assertIn("final_metric", results)
            print(f"   ✅ Pipeline Survived. Final Metric: {results['final_metric']:.4f}")
        except Exception as e:
            self.fail(f"❌ Pipeline CRASHED on Audit Data: {e}")

    def test_03_fake_code_detector(self):
        """Test 3: THE FAKE CODE DETECTOR. 
           Does the brain actually learn, or is it hardcoded?
           We check if predictions CHANGE after learning from history.
        """
        print("\n🧪 Test 03: Fake Brain Detector (Online Learning Verification)...")
        
        # 1. Get Baseline Prediction (Cold Start)
        brain = MetaLearner()
        # Mock DNA similar to audit data
        dna = {
            'n_instances': 500, 'n_features': 10, 
            'target_entropy': 1.5, 'mean_skewness': 0.5, 'task_type': 'classification'
        } 
        pred_cold = brain.predict(dna)
        print(f"   🧊 Cold Start Prediction (LR): {pred_cold['learning_rate']:.6f}")
        
        # 2. Inject Artificial History into Knowledge Base
        # We inject history that suggests a VERY high learning rate to see if Brain adapts.
        print("   💉 Injecting 'High LR' Memory into Knowledge Base...")
        history_data = []
        for _ in range(10):
            # Same DNA, but High Learning Rate was 'successful'
            row = dna.copy()
            row.update({
                'learning_rate': 0.05, # Much higher than typical cold start
                'weight_decay_l2': 0.001,
                'batch_size': 32,
                'dropout': 0.2,
                'optimizer_type': 'adam',
                'optimizer_type_code': 1,
                'final_metric': 0.99 # Good performance
            })
            history_data.append(row)
        
        pd.DataFrame(history_data).to_csv("knowledge_base.csv", index=False)
        
        # 3. Retrain Brain
        brain = MetaLearner() # Reload
        brain.train(epochs=20)
        
        # 4. Get New Prediction
        pred_warm = brain.predict(dna)
        print(f"   🔥 Warm Prediction (LR):       {pred_warm['learning_rate']:.6f}")
        
        # 5. The Verdict
        # If the brain ignores the history, pred_cold will likely equal pred_warm (or close).
        # We expect pred_warm to shift towards 0.05.
        
        diff = abs(pred_warm['learning_rate'] - pred_cold['learning_rate'])
        print(f"   Δ Change: {diff:.6f}")
        
        if diff < 0.001:
            self.fail("❌ FAKE BRAIN DETECTED: Prediction did not change after training on new history. The logic is likely hardcoded.")
        else:
            print("   ✅ Real Learning Detected. The brain adapted to the history.")

if __name__ == "__main__":
    unittest.main()
