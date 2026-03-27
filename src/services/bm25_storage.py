import os
from pathlib import Path
from typing import Optional


def _ensure_boto3():
    try:
        import boto3  # noqa: F401
    except Exception as e:
        raise RuntimeError("bm25_storage_backend=s3 requires boto3 to be installed") from e


def get_bm25_local_dir(config: dict, collection_name: str) -> str:
    """
    Resolve local BM25 directory for a collection.
    Default: <persist_directory>/bm25_index/<collection_name>
    """
    base = config.get("bm25_local_base_dir")
    if not base:
        base = os.path.join(config["persist_directory"], "bm25_index")
    return os.path.join(base, collection_name)


def _s3_prefix(config: dict, collection_name: str) -> str:
    prefix = (config.get("bm25_s3_prefix") or "bm25_index").strip("/")
    return f"{prefix}/{collection_name}"


def upload_bm25_to_s3(config: dict, collection_name: str, local_dir: str) -> None:
    backend = (config.get("bm25_storage_backend") or "local").lower()
    if backend != "s3":
        return
    _ensure_boto3()
    import boto3

    bucket = config.get("bm25_s3_bucket")
    if not bucket:
        raise ValueError("bm25_s3_bucket is required when bm25_storage_backend=s3")
    client = boto3.client("s3", region_name=config.get("bm25_s3_region"))
    prefix = _s3_prefix(config, collection_name)
    root = Path(local_dir)
    if not root.exists():
        return
    for path in root.rglob("*"):
        if path.is_file():
            rel = path.relative_to(root).as_posix()
            key = f"{prefix}/{rel}"
            client.upload_file(str(path), bucket, key)


def download_bm25_from_s3_if_needed(config: dict, collection_name: str, local_dir: str) -> None:
    backend = (config.get("bm25_storage_backend") or "local").lower()
    if backend != "s3":
        return
    root = Path(local_dir)
    # If already present locally, use cache.
    if root.exists() and any(root.iterdir()):
        return

    _ensure_boto3()
    import boto3

    bucket = config.get("bm25_s3_bucket")
    if not bucket:
        raise ValueError("bm25_s3_bucket is required when bm25_storage_backend=s3")
    client = boto3.client("s3", region_name=config.get("bm25_s3_region"))
    prefix = _s3_prefix(config, collection_name)
    paginator = client.get_paginator("list_objects_v2")
    root.mkdir(parents=True, exist_ok=True)
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel = key[len(prefix) :].lstrip("/")
            if not rel:
                continue
            target = root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(bucket, key, str(target))
