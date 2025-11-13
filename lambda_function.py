import os
import json
import logging
from collections import Counter
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3 = boto3.client('s3')


def process_s3_object(bucket: str, key: str) -> dict:
    """
    Read an S3 object (text) and return simple analysis results.
    Returns a dict with line_count, word_count, top_words.
    """
    try:
        resp = s3.get_object(Bucket=bucket, Key=key)
        body = resp['Body'].read()
        text = body.decode('utf-8', errors='replace')
    except ClientError as e:
        logger.exception("Failed to read S3 object %s/%s", bucket, key)
        raise

    lines = text.splitlines()
    words = []
    for line in lines:
        # simple tokenization on whitespace and lowercasing
        words.extend([w.strip(".,;:\"'()[]{}") .lower() for w in line.split() if w.strip()])

    word_counts = Counter(w for w in words if w)
    top_words = word_counts.most_common(10)

    result = {
        'bucket': bucket,
        'key': key,
        'line_count': len(lines),
        'word_count': len(words),
        'top_words': top_words,
    }
    return result


def _get_s3_from_s3_event(record: dict) -> tuple:
    # Support S3 event format: records[0].s3.bucket.name and records[0].s3.object.key
    s3rec = record.get('s3', {})
    bucket = s3rec.get('bucket', {}).get('name')
    key = s3rec.get('object', {}).get('key')
    return bucket, key


def _get_s3_from_custom_event(record: dict) -> tuple:
    # Support custom messages (for example, sent by `sqs_producer.py`) where body contains JSON with bucket/key
    # If record is an SQS record, the body may be a stringified JSON in record['body']
    if 'body' in record:
        # SQS record
        try:
            body = json.loads(record['body'])
        except Exception:
            # body might already be JSON object
            body = record['body']
    else:
        body = record

    bucket = None
    key = None
    if isinstance(body, dict):
        bucket = body.get('bucket') or (body.get('s3', {}).get('bucket') if 's3' in body else None)
        key = body.get('key') or (body.get('s3', {}).get('key') if 's3' in body else None)
    return bucket, key


def lambda_handler(event, context):
    """
    AWS Lambda handler that supports two input shapes:
    - A native S3 event (when Lambda is triggered by S3 PUT)
    - A message-like event (e.g., via SQS) that includes JSON with `bucket` and `key` fields

    It reads the referenced S3 object and returns a small analysis JSON.
    """
    logger.info('Received event: %s', json.dumps(event))

    # Many AWS event types put records under event['Records']
    records = event.get('Records') if isinstance(event, dict) else None
    results = []

    try:
        if records:
            for rec in records:
                # Try native S3 event first
                bucket, key = _get_s3_from_s3_event(rec)
                if not bucket or not key:
                    # Try custom/SQS body
                    bucket, key = _get_s3_from_custom_event(rec)

                if not bucket or not key:
                    logger.warning('No bucket/key found in record: %s', rec)
                    continue

                results.append(process_s3_object(bucket, key))
        else:
            # event could be a single object with bucket/key
            bucket = event.get('bucket')
            key = event.get('key')
            if not bucket or not key:
                bucket, key = _get_s3_from_custom_event(event)

            if not bucket or not key:
                raise ValueError('Event did not contain S3 bucket/key')

            results.append(process_s3_object(bucket, key))

        # Lambda may return JSON-serializable data
        return {
            'statusCode': 200,
            'body': json.dumps({'results': results}, default=str)
        }
    except Exception as exc:
        logger.exception('Processing failed')
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(exc)})
        }
