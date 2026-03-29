import logging
import time
import json
import signal
import sys
import os
from typing import Optional
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

sqs = boto3.client('sqs')
s3 = boto3.client('s3')

RUNNING = True

# ML model - loaded lazily
_model = None


def get_model():
    """Load the sentiment model lazily."""
    global _model
    if _model is None:
        try:
            import joblib
            model_path = os.environ.get('MODEL_PATH', 'sentiment_model.pkl')
            if os.path.exists(model_path):
                _model = joblib.load(model_path)
                logger.info("Loaded sentiment model from %s", model_path)
            else:
                logger.warning("Model file not found at %s, sentiment analysis disabled", model_path)
        except Exception as e:
            logger.warning("Failed to load model: %s", e)
    return _model


def graceful_shutdown(signum, frame):
    global RUNNING
    logger.info('Received shutdown signal')
    RUNNING = False


signal.signal(signal.SIGINT, graceful_shutdown)
signal.signal(signal.SIGTERM, graceful_shutdown)


def process_s3_object(bucket: str, key: str) -> dict:
    try:
        resp = s3.get_object(Bucket=bucket, Key=key)
        text = resp['Body'].read().decode('utf-8', errors='replace')
    except ClientError:
        logger.exception('Failed to read S3 object %s/%s', bucket, key)
        raise

    lines = text.splitlines()
    words = []
    for line in lines:
        for w in line.split():
            if w.strip():
                cleaned = w.strip('.,;:\"\'()[]{}!?').lower()
                if cleaned:
                    words.append(cleaned)

    # simple analysis
    from collections import Counter
    wc = Counter(words)

    result = {
        'bucket': bucket,
        'key': key,
        'lines': len(lines),
        'words': len(words),
        'top_5': wc.most_common(5),
    }
    
    # Add sentiment analysis if model is available
    model = get_model()
    if model is not None:
        try:
            # Analyze overall text sentiment
            if text.strip():
                prediction = model.predict([text])[0]
                try:
                    proba = model.predict_proba([text])[0]
                    confidence = float(max(proba))
                except AttributeError:
                    confidence = 1.0
                
                result['sentiment'] = {
                    'overall': prediction,
                    'confidence': confidence
                }
        except Exception as e:
            logger.warning("Sentiment analysis failed: %s", e)
    
    return result


def handle_message(message: dict):
    # message is an SQS message dict
    body = message.get('Body')
    try:
        data = json.loads(body)
    except Exception:
        # Maybe it's already JSON-like
        data = body

    if isinstance(data, dict):
        bucket = data.get('bucket')
        key = data.get('key')
    else:
        logger.warning('Message body not JSON/dict: %s', data)
        return False

    if not bucket or not key:
        logger.warning('No bucket/key in message: %s', data)
        return False

    logger.info('Processing s3://%s/%s', bucket, key)
    result = process_s3_object(bucket, key)
    logger.info('Result: %s', result)
    return True


def poll_queue(queue_url: str, wait_time: int = 20, max_messages: int = 5, visibility_timeout: Optional[int] = None):
    global RUNNING
    while RUNNING:
        try:
            params = dict(QueueUrl=queue_url, MaxNumberOfMessages=max_messages, WaitTimeSeconds=wait_time)
            if visibility_timeout is not None:
                params['VisibilityTimeout'] = visibility_timeout

            resp = sqs.receive_message(**params)
            messages = resp.get('Messages', [])

            if not messages:
                logger.debug('No messages received, waiting...')
                continue

            for msg in messages:
                receipt = msg['ReceiptHandle']
                try:
                    ok = handle_message(msg)
                    if ok:
                        # delete message
                        sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt)
                        logger.info('Deleted message %s', msg.get('MessageId'))
                    else:
                        logger.warning('Message processing indicated not-ok; leaving message')
                except Exception:
                    logger.exception('Error processing message; leaving in queue for retry')

        except ClientError:
            logger.exception('SQS client error')
            time.sleep(5)


def main():
    if len(sys.argv) < 2:
        print('Usage: python sqs_consumer.py <queue_url>')
        sys.exit(2)

    queue_url = sys.argv[1]
    logger.info('Polling SQS queue: %s', queue_url)
    poll_queue(queue_url)


if __name__ == '__main__':
    main()
