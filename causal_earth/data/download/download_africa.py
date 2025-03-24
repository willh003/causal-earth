import os
import sys
import requests
import tarfile
from tqdm import tqdm

def download_file(url, output_path):
    """
    Download a file from a URL with a progress bar.
    
    Args:
        url (str): URL to download from
        output_path (str): Path where the file will be saved
    """
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    # Stream the download with progress tracking
    print(f"Downloading from {url}")
    response = requests.get(url, stream=True)
    
    # Check if the request was successful
    if response.status_code != 200:
        print(f"Failed to download: HTTP status code {response.status_code}")
        return False
    
    # Get the total file size if available
    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192  # 8KB chunks
    
    with open(output_path, 'wb') as file, tqdm(
            desc=output_path,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
        for data in response.iter_content(block_size):
            file.write(data)
            bar.update(len(data))
    
    print(f"Download completed: {output_path}")
    return True

def extract_tarball(tarball_path, extract_dir='.'):
    """
    Extract a tarball to the specified directory.
    
    Args:
        tarball_path (str): Path to the tarball
        extract_dir (str): Directory to extract the contents to
    """
    print(f"Extracting {tarball_path} to {extract_dir}")
    with tarfile.open(tarball_path) as tar:
        # Get the total number of members for progress tracking
        members = tar.getmembers()
        total = len(members)
        
        # Extract with progress bar
        for i, member in enumerate(members):
            tar.extract(member, path=extract_dir)
            # Print progress
            if i % 100 == 0 or i == total - 1:
                sys.stdout.write(f"\rProgress: {i+1}/{total} files extracted ({(i+1)/total*100:.1f}%)")
                sys.stdout.flush()
        
        print("\nExtraction completed!")

def main():
    # URL of the dataset
    url = "https://zenodo.org/records/5079843/files/africa_minicubes.tar.gz?download=1"
    
    # Local path to save the tarball
    tarball_path = "/home/wph52/africa_minicubes.tar.gz"
    
    # Directory to extract the contents to
    extract_dir = "/home/wph52/africa_minicubes"
    
    # Download the tarball
    if download_file(url, tarball_path):
        # Create extraction directory
        os.makedirs(extract_dir, exist_ok=True)
        
        # Extract the tarball
        extract_tarball(tarball_path, extract_dir)
        
        print(f"Dataset has been downloaded and extracted to {extract_dir}/")
    else:
        print("Download failed")

if __name__ == "__main__":
    main()