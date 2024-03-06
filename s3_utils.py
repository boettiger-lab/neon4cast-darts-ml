import boto3
import json
from io import StringIO

def ls_bucket(bucket_name, directory_prefix, s3_client):
    """
    List files in a directory (prefix) within an S3 bucket.
    """
    # List objects in the specified directory
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=directory_prefix)

    # Extract file keys from the response
    file_keys = []
    if 'Contents' in response:
        for obj in response['Contents']:
            file_keys.append(obj['Key'])

    return file_keys

def read_credentials_from_json(file_path):
    with open(file_path, 'r') as f:
        credentials = json.load(f)
    return credentials.get('accessKey'), credentials.get('secretKey')

def upload_df_to_s3(object_name, dataframe, s3_client, bucket_name):
    # Convert DataFrame to CSV format in memory
    csv_buffer = StringIO()
    dataframe.to_csv(csv_buffer, index=False)
    
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

def ls_bucket(directory_prefix, s3_client, bucket_name):
    """
    List files in a directory (prefix) within an S3 bucket.
    """
    # List objects in the specified directory
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=directory_prefix)

    # Extract file keys from the response
    file_keys = []
    if 'Contents' in response:
        for obj in response['Contents']:
            file_keys.append(obj['Key'])

    return file_keys