import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)                     # for `from tests.test_engine import ...`
for folder in ("core", "pipeline", "spike_short"):
    sys.path.insert(0, os.path.join(ROOT, folder))
