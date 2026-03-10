#!/usr/bin/env python3
"""Data Proportion Sampling Script for TAPT Ablation Study"""

import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_splits(csv_path, output_dir):
    """Create stratified splits for different data proportions."""
    df = pd.read_csv(csv_path)
    
    # Find columns
    img_col = [c for c in df.columns if 'path' in c.lower() or 'file' in c.lower()][0]
    lbl_col = [c for c in df.columns if 'label' in c.lower() or 'class' in c.lower()][0]
    
    df = df.rename(columns={img_col: 'filepath', lbl_col: 'label'})[['filepath', 'label']].drop_duplicates()
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for prop in [0.0, 0.2, 0.5, 0.8, 1.0]:
        prop_dir = output_path / f'{int(prop * 100)}%'
        prop_dir.mkdir(exist_ok=True)
        
        if prop == 0.0:
            # Empty training set for baseline
            pd.DataFrame(columns=['filepath', 'label']).to_csv(prop_dir / 'train.csv', index=False)
            val_df, test_df = train_test_split(df, test_size=0.15, stratify=df['label'])
        else:
            train_df, temp_df = train_test_split(df, train_size=prop, stratify=df['label'])
            val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'])
        
        train_df.to_csv(prop_dir / 'train.csv', index=False)
        val_df.to_csv(prop_dir / 'val.csv', index=False)
        test_df.to_csv(prop_dir / 'test.csv', index=False)
        
        logger.info(f'{prop_dir}: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}')
    
    logger.info("Done!")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv-path', required=True)
    parser.add_argument('--output-dir', default='./data/splits_proportions')
    args = parser.parse_args()
    create_splits(args.csv_path, args.output_dir)
