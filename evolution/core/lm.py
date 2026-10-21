"""Single place that builds every dspy.LM, so the model/endpoint is configurable.

Model string forms:
  - "openai/gpt-4.1", "anthropic/...", ...   -> passed to litellm unchanged
  - "copilot/<model>"                        -> GitHub Copilot API, authenticated by exchanging
                                                $COPILOT_GITHUB_TOKEN for a short-lived API token
Optional env overrides for any non-copilot model (OpenAI-compatible proxy/gateway):
  EVOLUTION_API_BASE, EVOLUTION_API_KEY
"""

import json
import os
import time
import urllib.request
import dspy

_COPILOT_HEADERS = {
    "editor-version": "vscode/1.95.0",
    "Copilot-Integration-Id": "vscode-chat",
    "User-Agent": "GithubCopilot/1.0",
}
_copilot_cache: dict = {}  # {"token", "base", "exp"}; ponytail: process-local cache, fine for one run


def _copilot_credentials() -> tuple[str, str]:
    c = _copilot_cache
    if c and c["exp"] - 60 > time.time():
        return c["token"], c["base"]
    gh_token = os.environ.get("COPILOT_GITHUB_TOKEN")
    if not gh_token:
        raise RuntimeError("copilot/<model> needs the COPILOT_GITHUB_TOKEN environment variable")
    req = urllib.request.Request(
        "https://api.github.com/copilot_internal/v2/token",
        headers={"Authorization": f"token {gh_token}", **_COPILOT_HEADERS},
    )
    with urllib.request.urlopen(req, timeout=20) as response:
        d = json.load(response)
    c.update(token=d["token"], base=d["endpoints"]["api"], exp=d.get("expires_at", time.time() + 1500))
    return c["token"], c["base"]


class _CopilotLM(dspy.LM):
    """Refresh existing clients without mutating shared connection kwargs."""

    def _request_kwargs(self, kwargs):
        token, base = _copilot_credentials()
        return {"api_key": token, "api_base": base, **kwargs}

    def __call__(self, prompt=None, *, messages=None, **kwargs):
        return super().__call__(prompt, messages=messages, **self._request_kwargs(kwargs))

    async def acall(self, prompt=None, *, messages=None, **kwargs):
        return await super().acall(prompt, messages=messages, **self._request_kwargs(kwargs))

    def forward(self, prompt=None, messages=None, **kwargs):
        return super().forward(prompt, messages=messages, **self._request_kwargs(kwargs))

    async def aforward(self, prompt=None, messages=None, **kwargs):
        return await super().aforward(prompt, messages=messages, **self._request_kwargs(kwargs))


def make_lm(model: str, **kwargs) -> dspy.LM:
    if model.startswith("copilot/"):
        token, base = _copilot_credentials()
        kwargs.setdefault("api_key", token)
        kwargs.setdefault("api_base", base)
        kwargs.setdefault("extra_headers", _COPILOT_HEADERS)
        if model.split("/", 1)[1].startswith("gpt"):  # Copilot serves OpenAI models only via /responses
            kwargs.setdefault("model_type", "responses")
        return _CopilotLM("openai/" + model.split("/", 1)[1], **kwargs)
    if os.environ.get("EVOLUTION_API_BASE"):
        kwargs.setdefault("api_base", os.environ["EVOLUTION_API_BASE"])
    if os.environ.get("EVOLUTION_API_KEY"):
        kwargs.setdefault("api_key", os.environ["EVOLUTION_API_KEY"])
    return dspy.LM(model, **kwargs)
