import asyncio
import base64
from dataclasses import dataclass
from typing import Any, Iterator

import requests


@dataclass
class _Message:
    content: str


@dataclass
class _Delta:
    content: str


@dataclass
class _Choice:
    message: _Message | None = None
    delta: _Delta | None = None


@dataclass
class _Usage:
    total_tokens: int | None = None


@dataclass
class CompletionResponseCompat:
    choices: list[_Choice]
    usage: _Usage


class _CompatCompletions:
    def __init__(self, parent: "DocProcessingLLMClient"):
        self._parent = parent

    def create(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str | None = None,
        stream: bool = False,
        response_format: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
        **_: Any,
    ) -> CompletionResponseCompat | Iterator[CompletionResponseCompat]:
        if response_format is None and extra_body and extra_body.get("guided_json"):
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "guided_schema",
                    "schema": extra_body["guided_json"],
                    "strict": True,
                },
            }
        result = self._parent.complete(messages=messages, model=model, response_format=response_format)
        content = result.get("content", "")
        if not stream:
            return CompletionResponseCompat(
                choices=[_Choice(message=_Message(content=content))],
                usage=_Usage(total_tokens=None),
            )
        return self._parent.stream_text(content)


class _CompatChat:
    def __init__(self, parent: "DocProcessingLLMClient"):
        self.completions = _CompatCompletions(parent)


class OpenAICompatClient:
    """Drop-in subset for `client.chat.completions.create(...)`."""

    def __init__(self, parent: "DocProcessingLLMClient"):
        self.chat = _CompatChat(parent)


class _AsyncCompatCompletions:
    def __init__(self, parent: "DocProcessingLLMClient"):
        self._parent = parent

    async def create(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str | None = None,
        stream: bool = False,
        response_format: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> CompletionResponseCompat:
        if stream:
            raise ValueError("async streaming is not supported by llm-service /llm/complete")
        if response_format is None and extra_body and extra_body.get("guided_json"):
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "guided_schema",
                    "schema": extra_body["guided_json"],
                    "strict": True,
                },
            }
        result = await self._parent.acomplete(
            messages=messages,
            model=model,
            response_format=response_format,
        )
        return CompletionResponseCompat(
            choices=[_Choice(message=_Message(content=result.get("content", "")))],
            usage=_Usage(total_tokens=None),
        )


class _AsyncCompatChat:
    def __init__(self, parent: "DocProcessingLLMClient"):
        self.completions = _AsyncCompatCompletions(parent)


class AsyncOpenAICompatClient:
    """Drop-in subset for `await async_client.chat.completions.create(...)`."""

    def __init__(self, parent: "DocProcessingLLMClient"):
        self.chat = _AsyncCompatChat(parent)


class DocProcessingLLMClient:
    """
    Shared client for llm-service `/llm/complete`.
    All runtime LLM calls in src/ should route through this class.
    """

    def __init__(
        self,
        *,
        base_url: str,
        provider: str,
        timeout_seconds: int = 120,
        endpoint_path: str = "/llm/complete",
    ):
        self.base_url = base_url.rstrip("/")
        self.provider = provider
        self.timeout_seconds = timeout_seconds
        self.endpoint_path = endpoint_path

    @classmethod
    def from_config(cls, config: dict, provider_key: str = "llm_service_provider") -> "DocProcessingLLMClient":
        base_url = config.get("llm_service_base_url")
        provider = config.get(provider_key) or config.get("llm_service_provider") or "openai"
        timeout = int(config.get("llm_service_timeout_seconds", 120))
        endpoint = config.get("llm_service_llm_endpoint_path", "/llm/complete")
        if not base_url:
            raise ValueError("llm_service_base_url is required")
        return cls(base_url=base_url, provider=provider, timeout_seconds=timeout, endpoint_path=endpoint)

    def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str | None = None,
        response_format: dict[str, Any] | None = None,
        reasoning_effort: str | None = None,
        provider: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "provider": provider or self.provider,
            "messages": messages,
        }
        if model:
            payload["model"] = model
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort
        if response_format:
            payload["response_format"] = response_format

        resp = requests.post(
            f"{self.base_url}{self.endpoint_path}",
            json=payload,
            timeout=self.timeout_seconds,
        )
        try:
            resp.raise_for_status()
        except Exception as e:
            raise RuntimeError(f"LLM request failed: {resp.status_code} {resp.text}") from e
        data = resp.json()
        if "content" not in data:
            raise RuntimeError(f"Invalid llm response: {data}")
        return data

    async def acomplete(self, **kwargs: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self.complete, **kwargs)

    def stream_text(self, text: str, chunk_size: int = 64) -> Iterator[CompletionResponseCompat]:
        if not text:
            return iter(())

        def _gen() -> Iterator[CompletionResponseCompat]:
            for i in range(0, len(text), chunk_size):
                piece = text[i : i + chunk_size]
                yield CompletionResponseCompat(
                    choices=[_Choice(delta=_Delta(content=piece))],
                    usage=_Usage(total_tokens=None),
                )

        return _gen()

    def as_openai_compat(self) -> OpenAICompatClient:
        return OpenAICompatClient(self)

    def as_async_openai_compat(self) -> AsyncOpenAICompatClient:
        return AsyncOpenAICompatClient(self)

    def embeddings(
        self,
        *,
        input_text: str | list[str],
        model: str | None = None,
        provider: str | None = None,
        encoding_format: str = "float",
        dimensions: int | None = None,
        input_type: str | None = None,
        user: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "provider": provider or self.provider,
            "input": input_text,
            "encoding_format": encoding_format,
        }
        if model:
            payload["model"] = model
        if dimensions is not None:
            payload["dimensions"] = dimensions
        if input_type:
            payload["input_type"] = input_type
        if user:
            payload["user"] = user

        resp = requests.post(
            f"{self.base_url}/llm/embeddings",
            json=payload,
            timeout=self.timeout_seconds,
        )
        try:
            resp.raise_for_status()
        except Exception as e:
            raise RuntimeError(f"Embedding request failed: {resp.status_code} {resp.text}") from e
        data = resp.json()
        if "data" not in data or not isinstance(data["data"], list):
            raise RuntimeError(f"Invalid embedding response: {data}")
        return data


class DocProcessingEmbeddings:
    """
    Embedding adapter compatible with:
    - LangChain embedding interface (`embed_documents`, `embed_query`)
    - Chroma embedding_function callable (`__call__`)
    """

    def __init__(
        self,
        *,
        client: DocProcessingLLMClient,
        model: str | None = None,
        provider: str | None = None,
        batch_size: int = 64,
    ):
        self.client = client
        self.model = model
        self.provider = provider or client.provider
        self.batch_size = batch_size

    @classmethod
    def from_config(cls, config: dict) -> "DocProcessingEmbeddings":
        client = DocProcessingLLMClient.from_config(config)
        return cls(
            client=client,
            model=config.get("embeddings_model_name"),
            provider=config.get("llm_service_embeddings_provider") or config.get("llm_service_provider"),
            batch_size=int(config.get("llm_service_embeddings_batch_size", 64)),
        )

    def _normalize_embedding(self, emb: Any) -> list[float]:
        if isinstance(emb, list):
            return [float(x) for x in emb]
        if isinstance(emb, str):
            # Handle optional base64-encoded vectors if endpoint is configured that way.
            raw = base64.b64decode(emb)
            # Fallback to unsupported to avoid silently returning wrong vectors.
            raise RuntimeError(f"Unsupported base64 embedding payload of {len(raw)} bytes; use encoding_format=float")
        raise RuntimeError(f"Unsupported embedding type: {type(emb)!r}")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            data = self.client.embeddings(
                input_text=batch,
                model=self.model,
                provider=self.provider,
                encoding_format="float",
            )
            vectors.extend(self._normalize_embedding(item.get("embedding")) for item in data["data"])
        return vectors

    def embed_query(self, text: str) -> list[float]:
        if not text:
            return []
        data = self.client.embeddings(
            input_text=text,
            model=self.model,
            provider=self.provider,
            encoding_format="float",
            input_type="query",
        )
        first = data["data"][0].get("embedding")
        return self._normalize_embedding(first)

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self.embed_documents(input)
