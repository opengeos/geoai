#!/usr/bin/env python3
"""Upload a packaged QGIS plugin zip to plugins.qgis.org via XML-RPC.

The official plugin repository exposes an XML-RPC endpoint at
``https://plugins.qgis.org/plugins/RPC2/`` with a ``plugin.upload`` method
that accepts the zipped plugin as a base64-encoded ``Binary`` payload and
returns the new plugin id and version id on success.

Credentials must belong to a user with upload rights for the plugin and are
read from the ``QGIS_PLUGIN_REPO_USERNAME`` and ``QGIS_PLUGIN_REPO_PASSWORD``
environment variables so this script can be used from CI without leaking
secrets onto the command line.
"""

from __future__ import annotations

import argparse
import base64
import http.client
import os
import sys
from xmlrpc.client import Binary, Fault, ProtocolError, SafeTransport, ServerProxy

REPO_URL = "https://plugins.qgis.org/plugins/RPC2/"
DEFAULT_TIMEOUT = 60


class _AuthenticatedSafeTransport(SafeTransport):
    """HTTPS XML-RPC transport sending Basic auth via header with a timeout."""

    def __init__(
        self,
        username: str,
        password: str,
        timeout: float = DEFAULT_TIMEOUT,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        token = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode(
            "ascii"
        )
        self._auth_header = f"Basic {token}"
        self._timeout = timeout

    def make_connection(self, host):
        if self._connection and host == self._connection[0]:
            return self._connection[1]
        chost, self._extra_headers, x509 = self.get_host_info(host)
        self._connection = host, http.client.HTTPSConnection(
            chost,
            None,
            timeout=self._timeout,
            context=self.context,
            **(x509 or {}),
        )
        return self._connection[1]

    def send_headers(self, connection, headers):
        super().send_headers(connection, headers)
        connection.putheader("Authorization", self._auth_header)


def upload(
    zip_path: str,
    username: str,
    password: str,
    timeout: float = DEFAULT_TIMEOUT,
) -> tuple[int, int]:
    """Upload the given zip to plugins.qgis.org and return ``(plugin_id, version_id)``."""
    with open(zip_path, "rb") as fh:
        payload = Binary(fh.read())

    transport = _AuthenticatedSafeTransport(username, password, timeout=timeout)
    server = ServerProxy(REPO_URL, transport=transport, verbose=False)
    plugin_id, version_id = server.plugin.upload(payload)
    return plugin_id, version_id


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip_path", help="Path to the packaged plugin zip")
    args = parser.parse_args()

    username = os.environ.get("QGIS_PLUGIN_REPO_USERNAME")
    password = os.environ.get("QGIS_PLUGIN_REPO_PASSWORD")
    if not username or not password:
        print(
            "Error: QGIS_PLUGIN_REPO_USERNAME and QGIS_PLUGIN_REPO_PASSWORD must be set.",
            file=sys.stderr,
        )
        return 1

    if not os.path.isfile(args.zip_path):
        print(f"Error: zip file not found: {args.zip_path}", file=sys.stderr)
        return 1

    try:
        plugin_id, version_id = upload(args.zip_path, username, password)
    except Fault as exc:
        print(
            f"Upload failed: {exc.faultString} (code {exc.faultCode})", file=sys.stderr
        )
        return 1
    except ProtocolError as exc:
        print(f"Upload failed: HTTP {exc.errcode} {exc.errmsg}", file=sys.stderr)
        return 1

    print(f"Uploaded plugin id={plugin_id}, version id={version_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
