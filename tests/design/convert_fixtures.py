"""Convert R fixtures to numpy format for testing."""

import os
import numpy as np
import pandas as pd
from pathlib import Path


def convert_csv_to_npz(csv_path, npz_path):
    """Convert CSV file to NPZ format."""
    df = pd.read_csv(csv_path)
    
    # Save as dictionary of arrays
    data = {}
    for col in df.columns:
        data[col] = df[col].values
    
    # Also save as single array if numeric
    if df.select_dtypes(include=[np.number]).shape[1] == df.shape[1]:
        data['array'] = df.values
    
    np.savez(npz_path, **data)
    print(f"Converted {csv_path} to {npz_path}")


def convert_rds_metadata(fixture_dir):
    """Create metadata file for RDS fixtures."""
    metadata = {
        'simple_events': {
            'type': 'dataframe',
            'shape': (8, 4),
            'columns': ['onset', 'condition', 'duration', 'block']
        },
        'sampling_frame': {
            'type': 'sampling_frame',
            'blocklens': [100, 100, 100],
            'TR': 2.0
        },
        'design_matrix_simple': {
            'type': 'matrix',
            'shape': (300, 2),
            'columns': ['conditionA', 'conditionB']
        },
        'design_matrix_interaction': {
            'type': 'matrix',
            'shape': (300, 4),
            'columns': ['conditionA', 'conditionB', 'block1', 'block2']
        },
        'baseline_matrix_poly': {
            'type': 'matrix',
            'shape': (300, 9),  # 3 blocks * 3 poly terms
        },
        'contrast_weights_simple': {
            'type': 'vector',
            'length': 2,
            'values': [1, 0]  # A > 0
        }
    }
    
    np.savez(Path(fixture_dir) / 'metadata.npz', **metadata)
    print("Created metadata file")


def main():
    """Convert all fixtures."""
    fixture_dir = Path(__file__).parent / "fixtures"
    
    if not fixture_dir.exists():
        print(f"Fixture directory {fixture_dir} does not exist")
        print("Run generate_fixtures.R first")
        return
    
    # Convert CSV files to NPZ
    for csv_file in fixture_dir.glob("*.csv"):
        npz_file = csv_file.with_suffix(".npz")
        convert_csv_to_npz(csv_file, npz_file)
    
    # Create metadata
    convert_rds_metadata(fixture_dir)
    
    print("\nConversion complete!")


if __name__ == "__main__":
    main()