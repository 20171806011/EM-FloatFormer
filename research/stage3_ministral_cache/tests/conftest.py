"""Pytest configuration for Stage 3 tests."""
import os
import sys

# Prevent Windows MKL/PyTorch duplicate libiomp5md.dll abort when importing torch alongside numpy+mkl
if sys.platform.startswith("win"):
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
