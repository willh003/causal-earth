import os
import random
import shutil
import argparse
from pathlib import Path

def create_validation_split(base_dir, val_percent):
    """
    Create a validation set from a training set by randomly selecting X percent of files
    from each region and moving them to a validation directory with matching structure.
    
    Args:
        base_dir (str): Path to the directory containing the train folder
        val_percent (float): Percentage of files to move to validation (0-100)
    """
    # Convert percentage to fraction
    val_fraction = val_percent / 100.0
    
    # Define paths
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'val')
    
    # Ensure train directory exists
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Train directory not found at {train_dir}")
    
    # Create validation directory if it doesn't exist
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)
        print(f"Created validation directory at {val_dir}")
    
    # Get all regions in the train directory
    regions = [d for d in os.listdir(train_dir) 
               if os.path.isdir(os.path.join(train_dir, d))]
    
    print(f"Found {len(regions)} regions in the train directory")
    
    # Track statistics
    total_files_moved = 0
    total_files = 0
    
    # Process each region
    for region in regions:
        region_train_dir = os.path.join(train_dir, region)
        region_val_dir = os.path.join(val_dir, region)
        
        # Create region directory in val if it doesn't exist
        if not os.path.exists(region_val_dir):
            os.makedirs(region_val_dir)
        
        # Get all .nc files in the region
        region_files = [f for f in os.listdir(region_train_dir) 
                        if f.endswith('.nc') and os.path.isfile(os.path.join(region_train_dir, f))]
        
        # Calculate number of files to move
        num_files_to_move = max(1, int(len(region_files) * val_fraction))
        
        # Don't move more files than available
        num_files_to_move = min(num_files_to_move, len(region_files))
        
        # Randomly select files to move
        files_to_move = random.sample(region_files, num_files_to_move)
        
        # Move selected files
        for file_name in files_to_move:
            src = os.path.join(region_train_dir, file_name)
            dst = os.path.join(region_val_dir, file_name)
            shutil.move(src, dst)
        
        # Update statistics
        total_files_moved += num_files_to_move
        total_files += len(region_files)
        
        print(f"Region {region}: Moved {num_files_to_move} of {len(region_files)} files to validation")
    
    # Print summary
    overall_percent = (total_files_moved / total_files) * 100 if total_files > 0 else 0
    print(f"\nSummary:")
    print(f"  - Total files processed: {total_files}")
    print(f"  - Total files moved to validation: {total_files_moved}")
    print(f"  - Overall validation percentage: {overall_percent:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create validation split from training data")
    parser.add_argument("--base_dir", type=str, default="/home/wph52/earthnet2021/earthnet2021x/",
                        help="Base directory containing the train folder")
    parser.add_argument("--val_percent", type=float, default=20.0,
                        help="Percentage of files to move to validation (default: 20.0)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    print(f"Creating validation split with {args.val_percent}% of files")
    create_validation_split(args.base_dir, args.val_percent)
    print("Done!")