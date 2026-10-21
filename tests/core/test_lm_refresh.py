"""Existing Copilot clients must refresh authentication at request time."""

import asyncio
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from evolution.core import lm


@pytest.mark.parametrize("method", ["__call__", "acall", "forward", "aforward"])
def test_existing_model_refreshes_key_on_next_request(monkeypatch, method):
    authorizations = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            authorizations.append(self.headers["Authorization"])
            body = json.dumps({"id": "test", "object": "chat.completion", "created": 1,
                               "model": "test-model", "choices": [{"index": 0,
                               "message": {"role": "assistant", "content": "ready"},
                               "finish_reason": "stop"}],
                               "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state = {"token": "synthetic-old"}
    monkeypatch.setattr(lm, "_copilot_credentials", lambda: (state["token"], f"http://127.0.0.1:{server.server_port}/v1"))
    client = lm.make_lm("copilot/test-model", cache=False, num_retries=0)
    try:
        for token in ["synthetic-old", "synthetic-new"]:
            state["token"] = token
            result = getattr(client, method)("Reply ready")
            if method in {"acall", "aforward"}:
                asyncio.run(result)
        assert authorizations == ["Bearer synthetic-old", "Bearer synthetic-new"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_token_exchange_cache_refreshes_before_expiry(monkeypatch):
    lm._copilot_cache.clear()
    monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "synthetic-gh")
    clock = [1000]
    monkeypatch.setattr(lm.time, "time", lambda: clock[0])
    requests = []

    def exchange(request, timeout):
        requests.append(request)
        return io.BytesIO(json.dumps({"token": f"synthetic-{len(requests)}",
                                      "endpoints": {"api": "https://api.example.invalid"},
                                      "expires_at": clock[0] + 120}).encode())

    monkeypatch.setattr(lm.urllib.request, "urlopen", exchange)
    try:
        assert lm._copilot_credentials()[0] == "synthetic-1"
        clock[0] = 1059
        assert lm._copilot_credentials()[0] == "synthetic-1"
        clock[0] = 1060
        assert lm._copilot_credentials()[0] == "synthetic-2"
        assert len(requests) == 2
        assert requests[0].get_header("Authorization") == "token synthetic-gh"
    finally:
        lm._copilot_cache.clear()
