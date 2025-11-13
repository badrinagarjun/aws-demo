# aws-demo-project

A compact, real-feeling Python demo showing an AWS workflow using S3, SQS, and Lambda.

**Architecture**
- A client uploads a text file to S3 (`upload_to_s3.py`).
- A producer sends an SQS message referencing the S3 object (`sqs_producer.py`).
- A consumer polls SQS, reads the referenced S3 object, processes it, then deletes the message (`sqs_consumer.py`).
- An AWS Lambda function (`lambda_function.py`) does the same processing when invoked with an event (supports native S3 events and SQS-like messages).

This project is intentionally small but uses real code patterns (argument parsing, error handling, logging, and boto3 usage).

**Services used**
- S3: object storage for text files.
- SQS: message queue to decouple producers and consumers.
- Lambda: serverless function that can process S3-based events.

**Files**
- `lambda_function.py` - Lambda handler and helper functions. Reads S3 objects and returns analysis.
- `upload_to_s3.py` - CLI script to upload a local file to S3.
- `sqs_producer.py` - CLI script that sends a message to SQS with `bucket` and `key`.
- `sqs_consumer.py` - Long-polling SQS consumer that processes messages and deletes them on success.
- `requirements.txt` - Python dependencies (`boto3`).


## IAM / Permissions

Minimum policies required for each role or principal:

- For the machine running `upload_to_s3.py` (user or role):
  - `s3:PutObject` on the target bucket.

- For the machine running `sqs_producer.py`:
  - `sqs:SendMessage` on the target queue.

- For the machine running `sqs_consumer.py`:
  - `sqs:ReceiveMessage`, `sqs:DeleteMessage`, `sqs:GetQueueAttributes` on the queue.
  - `s3:GetObject` on the bucket(s) referenced.

- For Lambda (`lambda_function.py` execution role):
  - `logs:CreateLogGroup`, `logs:CreateLogStream`, `logs:PutLogEvents` for CloudWatch.
  - `s3:GetObject` for the source bucket.

Example minimal inline policy for the Lambda role (JSON):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject"],
      "Resource": ["arn:aws:s3:::YOUR_BUCKET_NAME/*"]
    },
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "*"
    }
  ]
}
```


## How to run locally

Prerequisites:
- Python 3.8+
- `pip install -r requirements.txt`
- AWS credentials configured (environment variables, shared credentials file, or IAM role if run on EC2/ECS).

Examples (PowerShell / Windows):

1) Upload a file to S3

```powershell
python .\upload_to_s3.py --file .\sample.txt --bucket my-demo-bucket --key sample.txt
```

2) Send a message to SQS referencing the uploaded object

```powershell
python .\sqs_producer.py --queue-url https://sqs.us-east-1.amazonaws.com/123456789012/my-queue --bucket my-demo-bucket --key sample.txt
```

3) Run the SQS consumer (will long-poll and process messages)

```powershell
python .\sqs_consumer.py https://sqs.us-east-1.amazonaws.com/123456789012/my-queue
```

4) Test the Lambda handler locally (optional)

You can invoke the `lambda_handler` function locally by creating an event file `event.json` and running a small wrapper. Example event shapes supported:

S3 event (partial example):

```json
{
  "Records": [
    {
      "s3": {
        "bucket": {"name": "my-demo-bucket"},
        "object": {"key": "sample.txt"}
      }
    }
  ]
}
```

Or an SQS-like message:

```json
{
  "Records": [
    { "body": "{\"bucket\": \"my-demo-bucket\", \"key\": \"sample.txt\"}" }
  ]
}
```

If you want to run it locally to see output, use a short wrapper (not provided) or run Python REPL:

```powershell
python -c "import json, lambda_function; print(lambda_function.lambda_handler(json.load(open('event.json')), None))"
```


## Sample output

When the consumer or Lambda processes a text file, it returns a JSON object with counts. Example:

```
INFO: Processing s3://my-demo-bucket/sample.txt
INFO: Result: {'bucket': 'my-demo-bucket', 'key': 'sample.txt', 'lines': 12, 'words': 86, 'top_5': [('the', 8), ('and', 6), ('example', 5), ('to', 4), ('a', 4)]}
```


## Notes and next steps

- For production use, add retries, exponential backoff, observability (structured logs, metrics), and consider idempotence.
- For Lambda deployment, package `lambda_function.py` and any dependencies (or use Layers) and configure triggers (S3 event or SQS event source mapping).

If you'd like, I can:
- Add a small `deploy.sh`/PowerShell script to create the SQS queue and S3 bucket and attach policies (quick, for demo only).
- Provide a CloudFormation / CDK template to deploy this demo.
