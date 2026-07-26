import os
import sys
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError


def get_required_env(name: str) -> str:
    """
    Get a required environment variable.
    Fail immediately if it is missing or empty.
    """
    value = os.environ.get(name)

    if not value:
        raise EnvironmentError(
            f"Required environment variable '{name}' is not set."
        )

    return value


def download_artifacts():
    print("\n" + "=" * 75)
    print(" DOWNLOADING ARTIFACTS FROM CLOUDFLARE R2 ")
    print("=" * 75)

    # ---------------------------------------------------------
    # Read environment variables
    # ---------------------------------------------------------

    access_key = get_required_env("R2_ACCESS_KEY_ID")
    secret_key = get_required_env("R2_SECRET_ACCESS_KEY")
    account_id = get_required_env("R2_ACCOUNT_ID")
    bucket_name = get_required_env("R2_BUCKET_NAME")

    artifact_dir = os.environ.get(
        "ARTIFACT_DIR",
        "/app/artifacts"
    )

    # R2 bucket prefix where your artifacts were uploaded.
    #
    # Your current R2 structure is:
    #
    # bucket
    # └── artifacts
    #     ├── customer_segmentation
    #     ├── sentiment_classification
    #     └── llm
    #
    r2_prefix = os.environ.get(
        "R2_ARTIFACT_PREFIX",
        "artifacts"
    )

    # ---------------------------------------------------------
    # Validate configuration
    # ---------------------------------------------------------

    endpoint_url = (
        f"https://{account_id}.r2.cloudflarestorage.com"
    )

    print(f"[CONFIG] Bucket        : {bucket_name}")
    print(f"[CONFIG] R2 prefix     : {r2_prefix}")
    print(f"[CONFIG] Local path    : {artifact_dir}")
    print(f"[CONFIG] R2 endpoint   : {endpoint_url}")

    # ---------------------------------------------------------
    # Create local artifact directory
    # ---------------------------------------------------------

    local_path = Path(artifact_dir)

    local_path.mkdir(
        parents=True,
        exist_ok=True
    )

    # ---------------------------------------------------------
    # Create S3-compatible R2 client
    # ---------------------------------------------------------

    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
        config=Config(
            signature_version="s3v4",
            retries={
                "max_attempts": 5,
                "mode": "standard"
            }
        )
    )

    # ---------------------------------------------------------
    # Test access to bucket
    # ---------------------------------------------------------

    print("\n[CHECK] Testing Cloudflare R2 connection...")

    try:
        response = s3.list_objects_v2(
            Bucket=bucket_name,
            Prefix=r2_prefix + "/",
            MaxKeys=1
        )

        if "Contents" not in response:
            raise RuntimeError(
                f"No objects found under R2 prefix "
                f"'{r2_prefix}/'. "
                f"Check your bucket name and prefix."
            )

        print("[OK] Successfully connected to Cloudflare R2.")

    except (ClientError, BotoCoreError) as e:
        raise RuntimeError(
            f"Failed to connect to Cloudflare R2: {e}"
        ) from e

    # ---------------------------------------------------------
    # Download all objects
    # ---------------------------------------------------------

    print("\n[DOWNLOAD] Downloading artifacts...")
    print(
        "This may take several minutes because the artifact "
        "repository is approximately 9 GB."
    )

    paginator = s3.get_paginator(
        "list_objects_v2"
    )

    downloaded = 0
    skipped = 0
    failed = 0

    for page in paginator.paginate(
        Bucket=bucket_name,
        Prefix=r2_prefix + "/"
    ):
        contents = page.get(
            "Contents",
            []
        )

        for obj in contents:

            object_key = obj["Key"]

            # Skip folder marker objects
            if object_key.endswith("/"):
                continue

            # Remove "artifacts/" prefix.
            #
            # Example:
            #
            # R2:
            # artifacts/customer_segmentation/foo.pkl
            #
            # Local:
            # /app/artifacts/customer_segmentation/foo.pkl
            #
            relative_key = object_key[
                len(r2_prefix) + 1:
            ]

            destination = (
                local_path / relative_key
            )

            destination.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            # -------------------------------------------------
            # Skip download if local file already exists
            # and size matches R2 object.
            #
            # This is useful when the container restarts.
            # -------------------------------------------------

            remote_size = obj.get(
                "Size",
                0
            )

            if (
                destination.exists()
                and destination.stat().st_size
                == remote_size
            ):
                skipped += 1

                print(
                    f"[SKIP] {relative_key}"
                )

                continue

            try:
                print(
                    f"[DOWNLOAD] {relative_key} "
                    f"({remote_size / (1024 ** 2):.2f} MB)"
                )

                s3.download_file(
                    bucket_name,
                    object_key,
                    str(destination)
                )

                downloaded += 1

            except (
                ClientError,
                BotoCoreError,
                OSError
            ) as e:

                failed += 1

                print(
                    f"[ERROR] Failed to download "
                    f"{object_key}: {e}",
                    file=sys.stderr
                )

    # ---------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------

    print("\n" + "=" * 75)
    print(" ARTIFACT DOWNLOAD SUMMARY ")
    print("=" * 75)

    print(
        f"Downloaded : {downloaded}"
    )

    print(
        f"Skipped    : {skipped}"
    )

    print(
        f"Failed     : {failed}"
    )

    print(
        f"Location   : {local_path}"
    )

    # ---------------------------------------------------------
    # Fail startup if anything could not be downloaded
    # ---------------------------------------------------------

    if failed > 0:
        raise RuntimeError(
            f"{failed} artifact(s) failed to download "
            f"from Cloudflare R2."
        )

    print(
        "\n[OK] All artifacts are available locally."
    )

    print(
        "=" * 75
    )
    print(
        " ARTIFACT DOWNLOAD COMPLETE "
    )
    print(
        "=" * 75 + "\n"
    )


if __name__ == "__main__":
    try:
        download_artifacts()

    except Exception as e:
        print(
            "\n[FATAL] Artifact download failed:",
            file=sys.stderr
        )

        print(
            str(e),
            file=sys.stderr
        )

        sys.exit(1)