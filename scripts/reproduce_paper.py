#!/usr/bin/env python3
"""Reproduce Paper Results: TAPT Ablation Study."""

import argparse
import sys
from pathlib import Path


project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
	parser = argparse.ArgumentParser(description='Reproduce TAPT ablation study')
	parser.add_argument('--data-root', type=str, default='./data')
	parser.add_argument('--output-root', type=str, default='./runs')
	parser.add_argument('--model-size', type=str, default='vitb16')
	parser.add_argument('--gpu-id', type=str, default='0')
	parser.add_argument('--dino-weights', type=str, required=True)
	args = parser.parse_args()

	print('=' * 60)
	print('Reproducing TAPT Ablation Study (Section 3.1)')
	print('=' * 60)
	print(f'Model: {args.model_size}')
	print(f'GPU: {args.gpu_id}')
	print(f'DINO weights: {args.dino_weights}')
	print(f'Data root: {args.data_root}')
	print(f'Output root: {args.output_root}')
	print('=' * 60)


if __name__ == '__main__':
	main()
