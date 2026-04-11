import boto3
from botocore.credentials import RefreshableCredentials
from botocore.session import get_session
import json
import base64


def get_sagemaker_client():
    # Initialize STS client to assume role
    sts_client = boto3.client("sts")

    def refresh():
        # Replace with your own AWS account ID and cross-account role name.
        response = sts_client.assume_role(
            RoleArn="arn:aws:iam::YOUR_ACCOUNT_ID:role/YOUR_CROSS_ACCOUNT_ROLE",
            RoleSessionName="sagemaker-session",
        )
        credentials = {
            "access_key": response["Credentials"]["AccessKeyId"],
            "secret_key": response["Credentials"]["SecretAccessKey"],
            "token": response["Credentials"]["SessionToken"],
            "expiry_time": response["Credentials"]["Expiration"].isoformat(),
        }
        return credentials

    session_credentials = RefreshableCredentials.create_from_metadata(
        metadata=refresh(),
        refresh_using=refresh,
        method="sts-assume-role",
    )

    session = get_session()
    session._credentials = session_credentials
    session.set_config_variable("region", "us-west-2")
    autorefresh_session = boto3.Session(botocore_session=session)

    client = autorefresh_session.client("sagemaker-runtime")
    return client


