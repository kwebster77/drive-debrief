"""Put ``src`` on sys.path so tests import the package without installing.

Kept at repo root (not relying on pytest's ``pythonpath`` option, which
only exists in pytest >= 7) so the suite runs on any pytest version.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
