# download_data.py
import os
import boto3
from dotenv import load_dotenv
from config import Config


def load_aws_credentials():
    load_dotenv()

    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")

    if not aws_access_key or not aws_secret_key:
        raise ValueError("AWS credentials not found. Check your .env file.")

    return aws_access_key, aws_secret_key


def create_s3_client():
    aws_access_key, aws_secret_key = load_aws_credentials()

    s3_client = boto3.client(
        "s3",
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name="us-east-1",
    )

    return s3_client


def download_file_from_s3(s3_client, bucket_name, s3_key, local_path):
    try:
        local_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Downloading {s3_key} to {local_path}...")
        s3_client.download_file(bucket_name, s3_key, str(local_path))
        print(f"✅ Successfully downloaded {s3_key}")

    except Exception as e:
        print(f"❌ Error downloading {s3_key}: {str(e)}")
        raise


def download_all_data():
    BUCKET_NAME = "zrive-ds-data"

    files_to_download = [
        {
            "s3_key": "groceries/sampled-datasets/orders.parquet",
            "local_path": Config.DATA_DIR / "orders.parquet",
        },
        {
            "s3_key": "groceries/sampled-datasets/regulars.parquet",
            "local_path": Config.DATA_DIR / "regulars.parquet",
        },
        {
            "s3_key": "groceries/sampled-datasets/inventory.parquet",
            "local_path": Config.DATA_DIR / "inventory.parquet",
        },
        {
            "s3_key": "groceries/trained-models/model.joblib",
            "local_path": Config.MODEL_PATH,
        },
    ]

    Config.create_directories()

    s3_client = create_s3_client()

    for file_info in files_to_download:
        download_file_from_s3(
            s3_client, BUCKET_NAME, file_info["s3_key"], file_info["local_path"]
        )

    print("\n🎉 All files downloaded successfully!")
    print(f"📁 Data files: {Config.DATA_DIR}")
    print(f"🤖 Model file: {Config.MODEL_PATH}")


def verify_downloads():
    required_files = [
        Config.DATA_DIR / "orders.parquet",
        Config.DATA_DIR / "regulars.parquet",
        Config.DATA_DIR / "inventory.parquet",
        Config.MODEL_PATH,
    ]

    print("\nVerifying downloads...")
    all_good = True

    for file_path in required_files:
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"✅ {file_path.name}: {size_mb:.2f} MB")
        else:
            print(f"❌ Missing: {file_path.name}")
            all_good = False

    if all_good:
        print("\n🎉 All files verified successfully!")
    else:
        print("\n⚠️ Some files are missing!")

    return all_good


if __name__ == "__main__":
    try:
        print("Starting download from S3...")
        print("=" * 50)

        download_all_data()
        verify_downloads()

    except Exception as e:
        print(f"\n❌ Download failed: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Check your .env file has AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
        print("2. Verify your AWS credentials have S3 read permissions")
        print("3. Check your internet connection")
