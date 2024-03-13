import boto3
import json
from io import StringIO
import pandas as pd

def ls_bucket(bucket_name, directory_prefix, s3_client, plotting=False):
    """
    List files in a directory (prefix) within an S3 bucket.
    """
    # List objects in the specified directory
    import pdb; pdb.set_trace()
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=directory_prefix)
    # Extract file keys from the response
    file_keys = []
    if 'Contents' in response:
        for obj in response['Contents']:
            if plotting:
                file_name = obj['Key']
                suffix = file_name.split('.')[-1]
                if suffix != 'csv':
                    file_name = None
            else:
                file_name = obj['Key'].split('/')[-1]
            file_keys.append(file_name)

    return file_keys

def read_credentials_from_json(file_path):
    with open(file_path, 'r') as f:
        credentials = json.load(f)
    return credentials.get('accessKey'), credentials.get('secretKey')

def upload_df_to_s3(object_name, dataframe, s3_client, bucket_name):
    # Convert DataFrame to CSV format in memory
    csv_buffer = StringIO()
    dataframe.to_csv(csv_buffer)
    
    s3_client.put_object(
        Bucket=bucket_name, 
        Key=object_name, 
        Body=csv_buffer.getvalue(),
    )

def download_df_from_s3(object_name, s3_client, bucket_name):
    # Get the CSV file object from S3
    response = s3_client.get_object(Bucket=bucket_name, Key=object_name)

    # Read the CSV data from the object into a Pandas DataFrame
    csv_data = response['Body'].read().decode('utf-8')
    df = pd.read_csv(StringIO(csv_data))

    return df