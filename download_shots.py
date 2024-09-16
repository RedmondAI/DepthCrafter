import os
import boto3
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse

# S3 Configuration
AWS_REGION = ""
S3_BUCKET_NAME = "luma3d"
S3_ACCESS_KEY = "283b8c4c028487ff3e1bbc2a891189e7"
S3_SECRET_KEY = "bc3d3ee909929216aa16c764d85e9b743bbc91e2589295955bda9624d0afdd21"
DEFAULT_ENDPOINT = "47ec7d0d5b6a6c2bcda5211d2d412fd0.r2.cloudflarestorage.com"
S3_ROOT_URL = "https://pub-da6ae3cf12bc4de49e659943f4080da6.r2.dev/"
START_PREFIX = "reel01/"

# Initialize S3 client
s3_client = boto3.client(
    's3',
    region_name=AWS_REGION or 'us-east-1',
    endpoint_url=f"https://{DEFAULT_ENDPOINT}",
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY
)

def list_directories(prefix):
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=S3_BUCKET_NAME, Prefix=prefix, Delimiter='/')
    directories = []
    for page in pages:
        dirs = page.get('CommonPrefixes', [])
        for d in dirs:
            directories.append(d.get('Prefix'))
    return directories

def download_rgb_deflicker(root_prefix):
    print("root_prefix", root_prefix)
    rgb_prefix = f"{root_prefix}rgb_deflicker/"
    print("rgb_prefix", rgb_prefix)
    response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=rgb_prefix)
    print("response", response)
    if 'Contents' not in response:
        return
    root_folder = root_prefix.rstrip('/').split('/')[-1]
    os.makedirs(os.path.join("input2", root_folder), exist_ok=True)
    for obj in response['Contents']:
        key = obj['Key']
        filename = os.path.basename(key)
        local_path = os.path.join("input2", root_folder, filename)
        
        # Check if the file already exists locally
        if os.path.exists(local_path):
            print(f"File {local_path} already exists, skipping download.")
            continue
        
        s3_client.download_file(S3_BUCKET_NAME, key, local_path)
        print(f"Downloaded {key} to {local_path}")

def main():
    parser = argparse.ArgumentParser(description='Download RGB deflicker files from S3')
    parser.add_argument('--start_prefix', type=str, help='The prefix to start listing directories from')
    args = parser.parse_args()
    directories = list_directories(args.start_prefix)
    for dir in directories:
        print(dir)
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(download_rgb_deflicker, d) for d in directories]
        for future in as_completed(futures):
            future.result()

if __name__ == "__main__":
    main()
