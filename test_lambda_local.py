#!/usr/bin/env python3
"""
Test the Lambda function locally with a mock event.
"""

import json
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Mock the S3 client for local testing
from unittest.mock import MagicMock, patch

def test_lambda_with_mock_s3():
    """Test lambda function with mocked S3 to avoid AWS credentials requirement."""
    
    # Sample text to simulate S3 object
    sample_text = """This is an amazing and wonderful product. I absolutely love it!
The quality is excellent and the service was fantastic.
Highly recommend to everyone. Best purchase ever!"""
    
    # Mock S3 client
    mock_s3_response = {
        'Body': MagicMock(read=lambda: sample_text.encode('utf-8'))
    }
    
    with patch('lambda_function.s3') as mock_s3:
        mock_s3.get_object.return_value = mock_s3_response
        
        # Import lambda function after patching
        import lambda_function
        
        # Test event
        event = {
            'bucket': 'test-bucket',
            'key': 'test.txt'
        }
        
        # Call lambda handler
        result = lambda_function.lambda_handler(event, None)
        
        print("Lambda Response:")
        print(json.dumps(result, indent=2, default=str))
        
        # Parse and check result
        body = json.loads(result['body'])
        print("\nProcessed Results:")
        print(json.dumps(body, indent=2, default=str))
        
        # Verify sentiment analysis is included
        if body['results'] and 'sentiment' in body['results'][0]:
            print("\n✓ Sentiment analysis working!")
            sentiment_info = body['results'][0]['sentiment']
            print(f"  Sentiment: {sentiment_info['overall']}")
            print(f"  Confidence: {sentiment_info['confidence']:.2f}")
        else:
            print("\n✗ Sentiment analysis not found (model may not be loaded)")


if __name__ == '__main__':
    test_lambda_with_mock_s3()
