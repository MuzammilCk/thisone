import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

try:
    from data_analyzer import DatasetAnalyzer
    from brain import MetaLearner 
    from engine import DynamicTrainer
except ImportError as e:
    print(f"❌ Error: Missing component files. {e}"); sys.exit(1)

class MetaTunePipeline:
    def __init__(self, data_path, target_col=None):
        self.data_path = data_path; self.target_col = target_col
        self.dataset_dna = None; self.predicted_params = None; self.training_results = None
        
    def run(self, train_brain=False, epochs=50):
        print("\n" + "="*70 + "\n🚀 METATUNE ENTERPRISE PIPELINE STARTING\n" + "="*70)
        
        # PHASE 1: DIAGNOSIS
        print("\n🔹 PHASE 1: Forensic Data Analysis")
        analyzer = DatasetAnalyzer(self.data_path, target_col=self.target_col)
        if not analyzer.load_data(): return None
        self.dataset_dna = analyzer.analyze()
        print(f"   ✓ Complexity Score: {self.dataset_dna['target_entropy']:.3f}")

        # PHASE 2: PRESCRIPTION
        print("\n🔹 PHASE 2: Neural Hyperparameter Prediction")
        brain = MetaLearner()
        if os.path.exists("meta_brain.pkl") and not train_brain: brain = MetaLearner.load("meta_brain.pkl")
        else: brain.train(epochs=50); brain.save("meta_brain.pkl")
        self.predicted_params = brain.predict(self.dataset_dna)
        print("\n   ✨ OPTIMIZED CONFIGURATION GENERATED:")
        for k, v in self.predicted_params.items(): print(f"   ► {k:20s}: {v}")

        # PHASE 3: EXECUTION
        print("\n🔹 PHASE 3: Training Deployment")
        trainer = DynamicTrainer(self.data_path, self.dataset_dna, self.predicted_params, target_col=self.target_col)
        self.training_results = trainer.run(epochs=epochs)
        
        self._generate_report(); self._visualize()
        return self.training_results

    def _generate_report(self):
        report = {"dna": {k: v.item() if hasattr(v, 'item') else v for k,v in self.dataset_dna.items()}, "params": self.predicted_params, "metrics": {"final_accuracy": float(self.training_results['final_metric']), "training_time": float(self.training_results['training_time'])}}
        with open("metatune_report.json", "w") as f: json.dump(report, f, indent=4)
        print("\n📄 Report generated: 'metatune_report.json'")

    def _visualize(self):
        try:
            plt.figure(figsize=(10, 5))
            plt.plot(self.training_results['train_loss_history'], label='Training Loss', linewidth=2)
            plt.plot(self.training_results['val_loss_history'], label='Validation Loss', linewidth=2, linestyle='--')
            plt.title(f"MetaTune Optimization Trajectory"); plt.xlabel("Epochs"); plt.ylabel("Loss"); plt.legend(); plt.grid(True, alpha=0.3)
            plt.savefig("metatune_graph.png"); print("📊 Visualization saved: 'metatune_graph.png'")
        except: pass

if __name__ == "__main__":
    pipeline = MetaTunePipeline(sys.argv[1] if len(sys.argv) > 1 else "demo_data.csv")
    pipeline.run(train_brain=False, epochs=30)
