import os
import unittest
from water_bottle_challenge import classify_preprocessed_audio

class TestWaterBottleClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # assume tests are run from repo root
        cls.data_dir = os.path.join(os.path.dirname(__file__), 'data')
        cls.top_path = os.path.join(cls.data_dir, 'top.csv')
        cls.bottom_path = os.path.join(cls.data_dir, 'bottom.csv')

    def test_top_strike(self):
        """top.csv should be classified as 0 (TOP)"""
        result = classify_preprocessed_audio(self.top_path)
        print("predicition")
        print(result)
        self.assertEqual(result, 0, f"Expected top.csv → 0, got {result}")

    def test_bottom_strike(self):
        """bottom.csv should be classified as 1 (BOTTOM)"""
        result = classify_preprocessed_audio(self.bottom_path)
        print("predicition")
        print(result)
        self.assertEqual(result, 1, f"Expected bottom.csv → 1, got {result}")

    def test_unlabeled_returns_int_or_none(self):
        """Check that an unlabeled file returns either 0,1, or None without error"""
        # pick first unlabeled file
        for fname in os.listdir(self.data_dir):
            if fname not in ('top.csv', 'bottom.csv') and fname.endswith('.csv'):
                fpath = os.path.join(self.data_dir, fname)
                print(fpath)
                result = classify_preprocessed_audio(fpath)
                print("predicition")
                print(result)
                self.assertIn(result, (0, 1, None))

if __name__ == '__main__':
    unittest.main()