"""
Dataset Download and Preparation Script for RailFOD23

This script provides utilities to download and prepare the RailFOD23 dataset
for RAAG-DETR training.

Dataset: RailFOD23 (Railway Foreign Object Detection 2023)
Source: https://figshare.com/articles/figure/RailFOD23_zip/24180738?file=43616139
"""

import os
import sys
import argparse
from pathlib import Path


def setup_dataset_directory(data_dir='data'):
    """Create dataset directory structure."""
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)
    
    print(f"Dataset directory created at: {data_path.absolute()}")
    return data_path


def print_download_instructions(data_dir):
    """Print instructions for manual dataset download."""
    print("\n" + "="*70)
    print("RAILFOD23 DATASET DOWNLOAD INSTRUCTIONS")
    print("="*70)
    print("\nPlease follow these steps to download the RailFOD23 dataset:\n")
    print("1. Visit the dataset page:")
    print("   https://figshare.com/articles/figure/RailFOD23_zip/24180738?file=43616139")
    print("\n2. Click the 'Download' button to download RailFOD23.zip")
    print(f"\n3. Move the downloaded file to: {data_dir}/")
    print("\n4. Extract the dataset:")
    print(f"   cd {data_dir}")
    print("   unzip RailFOD23.zip")
    print("\n5. Verify the dataset structure:")
    print(f"   ls {data_dir}/RailFOD23/")
    print("\n" + "="*70)
    print("\nNote: The dataset requires rail corridor annotations for training")
    print("the segmentation head. If not provided, you may need to:")
    print("  - Generate rail masks from existing annotations")
    print("  - Use rail line detection algorithms for pseudo-labels")
    print("  - Manually annotate rail corridors for a subset of images")
    print("="*70 + "\n")


def check_dataset_exists(data_dir='data'):
    """Check if dataset has been downloaded and extracted."""
    data_path = Path(data_dir)
    railfod_path = data_path / 'RailFOD23'
    zip_path = data_path / 'RailFOD23.zip'
    
    if railfod_path.exists():
        print(f"✓ Dataset found at: {railfod_path.absolute()}")
        return True
    elif zip_path.exists():
        print(f"! Dataset zip file found at: {zip_path.absolute()}")
        print(f"  Please extract it: cd {data_dir} && unzip RailFOD23.zip")
        return False
    else:
        print(f"✗ Dataset not found in: {data_path.absolute()}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Download and setup RailFOD23 dataset for RAAG-DETR'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Directory to store dataset (default: data)'
    )
    parser.add_argument(
        '--check',
        action='store_true',
        help='Check if dataset exists without showing download instructions'
    )
    
    args = parser.parse_args()
    
    # Setup dataset directory
    data_path = setup_dataset_directory(args.data_dir)
    
    # Check if dataset exists
    dataset_exists = check_dataset_exists(args.data_dir)
    
    if not dataset_exists and not args.check:
        # Show download instructions
        print_download_instructions(args.data_dir)
    elif dataset_exists:
        print("\nDataset is ready for use!")
        print("\nNext steps:")
        print("  1. Review the dataset structure")
        print("  2. Prepare rail corridor annotations if needed")
        print("  3. Start training with: python train.py --data", args.data_dir)
    
    return 0 if dataset_exists else 1


if __name__ == '__main__':
    sys.exit(main())
