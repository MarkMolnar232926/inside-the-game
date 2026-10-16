"""Asset storage: Azure Blob Storage with a local-folder fallback."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


class AssetStore:
    def __init__(self, local_root: Path = Path("outputs/assets")):
        self.local_root = local_root
        self.conn = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        self.container = os.getenv("AZURE_STORAGE_CONTAINER", "pitchlens-assets")
        self._client = None
        if self.conn and os.getenv("PITCHLENS_MODE", "mock") == "azure":
            from azure.storage.blob import BlobServiceClient  # type: ignore[import-not-found]

            self._client = BlobServiceClient.from_connection_string(self.conn).get_container_client(self.container)

    def put(self, path: Path, key: str, content_type: str = "application/octet-stream") -> str:
        """Upload (or copy locally) and return a URL/path the UI can load."""
        if self._client is not None:
            from azure.storage.blob import ContentSettings  # type: ignore[import-not-found]

            with path.open("rb") as fh:
                self._client.upload_blob(key, fh, overwrite=True, content_settings=ContentSettings(content_type=content_type))
            return f"{self._client.url}/{key}"
        dst = self.local_root / key
        dst.parent.mkdir(parents=True, exist_ok=True)
        if path.resolve() != dst.resolve():
            shutil.copyfile(path, dst)
        return f"/assets/{key}"
