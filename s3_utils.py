import boto3
import json
from io import StringIO
import pandas as pd

def ls_bucket(directory_prefix, s3_dict={'client': None, 'bucket': None}, plotting=False):
    """
    List files in a directory (prefix) within an S3 bucket.
    """
    # List objects in the specified directory
    response = s3_dict['client'].list_objects_v2(Bucket=s3_dict['bucket'], Prefix=directory_prefix)
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

def upload_df_to_s3(object_name, dataframe, s3_dict={'client': None, 'bucket': None}):
    # Convert DataFrame to CSV format in memory
    csv_buffer = StringIO()
    dataframe.to_csv(csv_buffer, index=False)
    
    s3_dict['client'].put_object(
        Bucket=s3_dict['bucket'], 
        Key=object_name, 
        Body=csv_buffer.getvalue(),
    )

def download_df_from_s3(object_name, s3_dict={'client': None, 'bucket': None}):
    # Get the CSV file object from S3
    response = s3_dict['client'].get_object(Bucket=s3_dict['bucket'], Key=object_name)

    # Read the CSV data from the object into a Pandas DataFrame
    csv_data = response['Body'].read().decode('utf-8')
    df = pd.read_csv(StringIO(csv_data))

    return df