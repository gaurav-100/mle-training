import os
import unittest


class SimpleTest(unittest.TestCase):
    def test_model_file_exists(self):
        self.assertIsInstance(os.path.exists("artifacts"))


if __name__ == "__main__":
    unittest.main()
