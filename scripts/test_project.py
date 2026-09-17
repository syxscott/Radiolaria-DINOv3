#!/usr/bin/env python3
"""Test script for Radiolaria-DINOv3."""

import sys
from pathlib import Path


project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_imports():
	print("Testing imports...")

	ok = True
	try:
		import torch
		print(f"  torch {torch.__version__}")
	except ImportError as e:
		print(f"  Failed: {e}")
		ok = False

	try:
		import torchvision
		print(f"  torchvision {torchvision.__version__}")
	except ImportError as e:
		print(f"  Failed: {e}")
		ok = False

	try:
		import pandas
		print(f"  pandas {pandas.__version__}")
	except ImportError as e:
		print(f"  Failed: {e}")
		ok = False

	if ok:
		print("All imports successful!")
	return ok


def main():
	print("=" * 60)
	print("Radiolaria-DINOv3 Project Test")
	print("=" * 60)
	ok = test_imports()
	print("\nTest completed!")
	sys.exit(0 if ok else 1)


if __name__ == '__main__':
	main()
