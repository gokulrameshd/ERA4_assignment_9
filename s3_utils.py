"""
s3_utils.py
--------------
Helper functions for initializing Boto3 S3 client and uploading files.
"""
import boto3
import os

def init_s3_client(bucket_name, root_folder):
    """
    Initializes and returns the Boto3 S3 client using the instance's IAM Role.
    Returns None if S3 uploads are disabled or an error occurs.
    """
    try:
        # This automatically uses the IAM Role from your EC2 instance
        s3_client = boto3.client('s3')
        print(f"✅ S3 Uploads Enabled. Target: s3://{bucket_name}/{root_folder}")
        return s3_client
    except Exception as e:
        print(f"⚠️ Failed to initialize S3 client. S3 uploads disabled. Error: {e}")
        return None

def upload_file_to_s3(s3_client, local_file_path, bucket_name, s3_key):
    """
    Uploads a single local file to a specific S3 path (key).
    """
    if not s3_client:
        return  # S3 is disabled or failed to init

    # Check if the local file exists before trying to upload
    if not os.path.exists(local_file_path):
        print(f"⚠️ S3 Upload Warning: Local file not found at {local_file_path}. Skipping.")
        return

    try:
        # Use os.path.basename to get just the filename for the print log
        filename = os.path.basename(local_file_path)
        print(f"  Uploading {filename} to s3://{bucket_name}/{s3_key}...")
        
        s3_client.upload_file(
            Filename=local_file_path,
            Bucket=bucket_name,
            Key=s3_key
        )
    except Exception as e:
        print(f"  ⚠️ S3 upload FAILED for {local_file_path}: {e}")