"""Document storage helper for S3 pre-signed URLs and downloads."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from urllib.parse import quote

from apps.api.config import Settings


def _long_path(path: Path) -> str:
    """Return a Windows extended-length path for paths > 260 chars (no-op elsewhere)."""
    if os.name != "nt":
        return str(path)
    abs_str = str(path.resolve())
    if abs_str.startswith("\\\\?\\") or len(abs_str) < 240:
        return abs_str
    if abs_str.startswith("\\\\"):
        return "\\\\?\\UNC\\" + abs_str.lstrip("\\")
    return "\\\\?\\" + abs_str


class DocumentService:
    """Encapsulates object storage interactions used by document ingestion."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    # Max length of the filename segment in the object key. Keeps the full
    # local-storage path comfortably below Windows' 260-char MAX_PATH limit
    # even with the long workspace/document UUID prefix.
    _MAX_FILENAME_LEN = 80

    def build_s3_key(self, workspace_id: str, document_id: str, filename: str) -> str:
        """Build deterministic object key for workspace-scoped uploads.

        The filename segment is sanitised (no path separators) and truncated
        to keep the resulting on-disk path under the Windows MAX_PATH limit.
        Uniqueness comes from the document_id UUID, so truncation is safe.
        """
        safe_name = filename.replace("\\", "_").replace("/", "_")
        if len(safe_name) > self._MAX_FILENAME_LEN:
            stem = Path(safe_name).stem
            suffix = Path(safe_name).suffix
            # Reserve room for the suffix; trim the stem.
            keep = max(1, self._MAX_FILENAME_LEN - len(suffix))
            safe_name = stem[:keep] + suffix
        return f"workspaces/{workspace_id}/documents/{document_id}/{safe_name}"

    def generate_upload_url(self, s3_key: str) -> str:
        """Return a pre-signed URL when boto3 is available, otherwise a deterministic fallback."""

        client = self._build_boto_client()
        if client is None:
            return self._fallback_url("PUT", s3_key)

        try:
            return str(
                client.generate_presigned_url(
                    ClientMethod="put_object",
                    Params={"Bucket": self._settings.s3_bucket_name, "Key": s3_key},
                    ExpiresIn=self._settings.s3_presign_expire_seconds,
                )
            )
        except Exception:
            return self._fallback_url("PUT", s3_key)

    def generate_download_url(self, s3_key: str) -> str:
        """Return a pre-signed download URL."""

        client = self._build_boto_client()
        if client is None:
            return self._fallback_url("GET", s3_key)

        try:
            return str(
                client.generate_presigned_url(
                    ClientMethod="get_object",
                    Params={"Bucket": self._settings.s3_bucket_name, "Key": s3_key},
                    ExpiresIn=self._settings.s3_presign_expire_seconds,
                )
            )
        except Exception:
            return self._fallback_url("GET", s3_key)

    def download_to_tempfile(self, s3_key: str, filename: str) -> tuple[Path, bool]:
        """Download object into a temp file and return the local path."""

        suffix = Path(filename).suffix or ".tmp"
        fd, temp_path = tempfile.mkstemp(prefix="aris_ingest_", suffix=suffix)
        path = Path(temp_path)
        os.close(fd)

        try:
            client = self._build_boto_client()
            if client is None:
                local_object_path = self._local_object_path(s3_key)
                long_local = _long_path(local_object_path)
                if os.path.isfile(long_local):
                    shutil.copyfile(long_local, str(path))
                    return path, False

                # Last-resort fallback for environments that skipped upload PUT.
                with path.open("w", encoding="utf-8") as handle:
                    handle.write("ARIS fallback ingest content")
                return path, True

            try:
                client.download_file(self._settings.s3_bucket_name, s3_key, str(path))
                return path, False
            except Exception:
                local_object_path = self._local_object_path(s3_key)
                long_local = _long_path(local_object_path)
                if os.path.isfile(long_local):
                    shutil.copyfile(long_local, str(path))
                    return path, False

                with path.open("w", encoding="utf-8") as handle:
                    handle.write("ARIS fallback ingest content")
                return path, True
        except Exception:
            path.unlink(missing_ok=True)
            raise

    def _fallback_url(self, method: str, s3_key: str) -> str:
        encoded_bucket = quote(self._settings.s3_bucket_name, safe="")
        encoded_key = quote(s3_key, safe="/")
        base = self._settings.local_object_store_base_url.rstrip("/")
        return f"{base}/object-storage/{encoded_bucket}/{encoded_key}"

    def _local_object_path(self, s3_key: str) -> Path:
        root = Path(self._settings.local_object_store_dir)
        return root / self._settings.s3_bucket_name / s3_key

    def _build_boto_client(self):
        try:
            import boto3  # type: ignore
        except ImportError:
            return None

        kwargs: dict[str, object] = {
            "service_name": "s3",
            "region_name": self._settings.s3_region,
        }

        if self._settings.s3_endpoint_url:
            kwargs["endpoint_url"] = self._settings.s3_endpoint_url
        if self._settings.s3_access_key_id:
            kwargs["aws_access_key_id"] = self._settings.s3_access_key_id
        if self._settings.s3_secret_access_key:
            kwargs["aws_secret_access_key"] = self._settings.s3_secret_access_key

        return boto3.client(**kwargs)
