import os
import sys
import unittest

# Ensure parent directory is on sys.path so 'server' package can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if __name__ == '__main__':
    tests = unittest.defaultTestLoader.discover('tests', pattern='test_pipeline.py')
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(tests)
    if not result.wasSuccessful():
        sys.exit(1)
