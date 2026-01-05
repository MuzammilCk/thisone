
import unittest
import os
import pandas as pd
from generate_audit_data import generate_audit_data

class TestAuditGeneration(unittest.TestCase):
    def tearDown(self):
        if os.path.exists("audit_data.csv"):
            os.remove("audit_data.csv")

    def test_generation(self):
        generate_audit_data(n_rows=50)
        self.assertTrue(os.path.exists("audit_data.csv"))
        df = pd.read_csv("audit_data.csv")
        self.assertEqual(len(df), 50)
        self.assertIn('feat_outlier', df.columns)

if __name__ == '__main__':
    unittest.main()
