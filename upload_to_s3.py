import argparse
import os
import logging
import sys
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

s3 = boto3.client('s3')


def upload_file(file_path: str, bucket: str, key: str):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        # Use upload_file for large files; it handles multipart automatically
        s3.upload_file(file_path, bucket, key)
        logger.info('Uploaded %s to s3://%s/%s', file_path, bucket, key)
        return True
    except ClientError as e:
        logger.exception('Failed to upload file to S3')
        raise


def main():
    parser = argparse.ArgumentParser(description='Upload a local file to S3')
    parser.add_argument('--file', '-f', required=True, help='Local file path to upload')
    parser.add_argument('--bucket', '-b', required=True, help='Destination S3 bucket')
    parser.add_argument('--key', '-k', help='S3 object key (defaults to basename of file)')

    args = parser.parse_args()
    key = args.key or os.path.basename(args.file)

    try:
        upload_file(args.file, args.bucket, key)
    except Exception as e:
        logger.error('Upload failed: %s', e)
        sys.exit(1)


if __name__ == '__main__':
    main()
