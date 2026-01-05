
import unittest
import pandas as pd
import numpy as np
import os
import sys
import io
from contextlib import redirect_stdout
from data_analyzer import DatasetAnalyzer

class TestDataAnalyzer(unittest.TestCase):
    def setUp(self):
        # Create a dummy CSV for testing
        self.test_csv = "test_data_analyzer_dummy.csv"
        df = pd.DataFrame({
            'A': np.random.rand(10),
            'B': np.random.choice(['x', 'y'], 10),
            'target': np.random.choice([0, 1], 10)
        })
        df.to_csv(self.test_csv, index=False)

    def tearDown(self):
        if os.path.exists(self.test_csv):
            os.remove(self.test_csv)

    def test_analyzer_functionality(self):
        # Determine strict functionality without printing
        analyzer = DatasetAnalyzer(self.test_csv, target_col='target')
        self.assertTrue(analyzer.load_data())
        dna = analyzer.analyze()
        self.assertIsNotNone(dna)
        self.assertIn('n_instances', dna)
        self.assertEqual(dna['n_instances'], 10)
        
    def test_cli_behavior_simulation(self):
        # mimic the old main block logic
        f = io.StringIO()
        with redirect_stdout(f):
            analyzer = DatasetAnalyzer(self.test_csv, target_col='target')
            if analyzer.load_data():
                dna = analyzer.analyze()
                analyzer.print_summary()
                print("\n✅ Analysis Complete. DNA ready for Meta-Brain.")
        
        output = f.getvalue()
        self.assertIn("DATASET DNA", output)
        self.assertIn("Analysis Complete", output)

if __name__ == '__main__':
    unittest.main()
