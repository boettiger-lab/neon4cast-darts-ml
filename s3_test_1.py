import boto3
from botocore.client import Config

# Other valid options here are 'auto' (default) and 'virtual'

s3 = boto3.client(
    's3', 
    endpoint_url='https://minio.carlboettiger.info',
    aws_access_key_id="AB2jBtalL1aDSXV4pbpy",
    aws_secret_access_key="skmDgFmHwMxbqB2BCKkR8ZnshOUGOhfqA2x0JpMI"
)
s3.list_objects(Bucket = "shared-neon4cast-darts")