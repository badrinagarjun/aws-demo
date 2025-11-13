import argparse
import json
import logging
import sys
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

sqs = boto3.client('sqs')


def send_s3_message(queue_url: str, bucket: str, key: str, delay_seconds: int = 0):
    body = {
        'bucket': bucket,
        'key': key
    }

    try:
        resp = sqs.send_message(
            QueueUrl=queue_url,
            MessageBody=json.dumps(body),
            DelaySeconds=delay_seconds
        )
        logger.info('Sent message to SQS. MessageId=%s', resp.get('MessageId'))
        return resp
    except ClientError:
        logger.exception('Failed to send message to SQS')
        raise


def main():
    parser = argparse.ArgumentParser(description='Send S3 object reference to SQS')
    parser.add_argument('--queue-url', '-q', required=True, help='SQS queue URL')
    parser.add_argument('--bucket', '-b', required=True, help='S3 bucket name')
    parser.add_argument('--key', '-k', required=True, help='S3 object key')
    parser.add_argument('--delay', '-d', type=int, default=0, help='DelaySeconds for the message')

    args = parser.parse_args()

    try:
        send_s3_message(args.queue_url, args.bucket, args.key, args.delay)
    except Exception as e:
        logger.error('Failed to produce message: %s', e)
        sys.exit(1)


if __name__ == '__main__':
    main()
