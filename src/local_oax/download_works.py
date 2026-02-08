"""
Download OpenAlex Works data from S3 bucket to local directory.
"""
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import os

# --- CONFIGURATION ---
BUCKET_NAME = "openalex"
PREFIX = "data/works/"
# Ensure this matches where you want the files
LOCAL_DIR = Path("/data/raw_openalex/works")
MAX_THREADS = 8 

# Setup S3 client (Anonymous/Public Access)
s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

def download_file(obj):
    key = obj['Key']
    size = obj['Size']
    
    # Calculate local path 
    # Key looks like "data/works/updated_date=2023.../part_000.gz"
    # We strip "data/works/" to match local structure
    rel_path = key.replace(PREFIX, "", 1).lstrip("/")
    if not rel_path: return 
    
    local_path = LOCAL_DIR / rel_path
    
    # Check if file exists and is valid
    if local_path.exists():
        if local_path.stat().st_size == size:
            print(f"[SKIP] {rel_path}")
            return
    
    # Create parent dirs
    local_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Download
    print(f"[DOWN] {rel_path} ({size / 1024 / 1024:.2f} MB)")
    try:
        s3.download_file(BUCKET_NAME, key, str(local_path))
    except Exception as e:
        print(f"[ERR ] {rel_path}: {e}")

def main():
    print(f"Starting sync from s3://{BUCKET_NAME}/{PREFIX}")
    print(f"Target Directory: {LOCAL_DIR}")
    
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=BUCKET_NAME, Prefix=PREFIX)
    
    files_to_download = []
    
    print("Listing files from S3 (this takes a moment)...")
    for page in pages:
        if 'Contents' in page:
            files_to_download.extend(page['Contents'])
            
    print(f"Found {len(files_to_download)} files. Starting download...")
    
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        executor.map(download_file, files_to_download)
        
    print("Download complete.")

if __name__ == "__main__":
    main()