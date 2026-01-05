
import unittest
from brain import MetaLearner

class TestBrain(unittest.TestCase):
    def test_brain_predict(self):
        brain = MetaLearner()
        # Mock DNA with all features set to 0.5
        dna = {k: 0.5 for k in brain.input_features}
        
        prediction = brain.predict(dna)
        
        # Verify structure
        expected_keys = ['learning_rate', 'weight_decay_l2', 'batch_size', 'dropout', 'optimizer_type']
        for k in expected_keys:
            self.assertIn(k, prediction)
            
        print(f"\nPredicted Params: {prediction}")

if __name__ == '__main__':
    unittest.main()
