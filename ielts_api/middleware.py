"""Bearer authentication and bounded input before BentoML deserializes requests."""

import asyncio
import hmac
import os

from starlette.datastructures import UploadFile
from starlette.requests import Request
from starlette.responses import JSONResponse


class APIGuard:
    def __init__(
        self,
        app,
        *,
        required=False,
        upload=False,
        max_bytes=1024 * 1024,
        max_concurrency=4,
    ):
        self.app = app
        self.key = os.environ.get("IELTS_API_KEY", "")
        if required and len(self.key) < 32:
            raise RuntimeError("Set IELTS_API_KEY to a random secret of at least 32 characters")
        self.upload = upload
        self.max_bytes = max_bytes
        self.max_concurrency = max_concurrency
        self.active = 0

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] in {"/livez", "/readyz"}:
            return await self.app(scope, receive, send)

        async def reject(status, message):
            await JSONResponse({"error": message}, status_code=status)(scope, receive, send)

        headers = dict(scope["headers"])
        if self.key and not hmac.compare_digest(
            headers.get(b"authorization", b""), f"Bearer {self.key}".encode()
        ):
            return await reject(401, "Valid bearer token required")
        if scope["method"] != "POST":
            return await self.app(scope, receive, send)
        # Reserve before reading/spooling a body; Bento's traffic guard runs further inside.
        if self.active >= self.max_concurrency:
            return await reject(429, "Server is busy; retry with backoff")
        self.active += 1
        try:
            return await self.handle_post(scope, receive, send, headers, reject)
        finally:
            self.active -= 1

    async def handle_post(self, scope, receive, send, headers, reject):
        media = headers.get(b"content-type", b"").split(b";", 1)[0].lower()
        expected = b"multipart/form-data" if self.upload else b"application/json"
        if media != expected:
            return await reject(415, f"Use {expected.decode()}")
        try:
            declared = int(headers.get(b"content-length", b"0"))
            if declared < 0 or declared > self.max_bytes:
                return await reject(413, "Request body is too large")
        except ValueError:
            return await reject(400, "Invalid Content-Length")
        # Enforce limits even with chunked transfer or a dishonest Content-Length.
        body = bytearray()
        try:
            async with asyncio.timeout(30):
                while True:
                    message = await receive()
                    if message["type"] == "http.disconnect":
                        return
                    body.extend(message.get("body", b""))
                    if len(body) > self.max_bytes:
                        return await reject(413, "Request body is too large")
                    if not message.get("more_body", False):
                        break
        except TimeoutError:
            return await reject(408, "Upload took longer than 30 seconds")

        def replay():
            sent = False

            async def read():
                nonlocal sent
                if not sent:
                    sent = True
                    return {"type": "http.request", "body": bytes(body), "more_body": False}
                return await receive()

            return read

        if self.upload:
            try:
                async with Request(scope, replay()).form(max_files=12, max_fields=1) as form:
                    audio = form.getlist("audio")
                    if (
                        set(form) != {"audio", "metadata"}
                        or not audio
                        or any(not isinstance(item, UploadFile) for item in audio)
                        or len(form.getlist("metadata")) != 1
                        or not isinstance(form["metadata"], str)
                        or len(form["metadata"]) > 64 * 1024
                    ):
                        return await reject(
                            400, "Send audio file parts and one metadata JSON field"
                        )
            except Exception:
                return await reject(400, "Invalid multipart upload")
        return await self.app(scope, replay(), send)
