"""LLM によるプロンプト支援（仕様: docs/spec.md「LLMプロンプト支援」）。

llama-server（OpenAI 互換 API）を GPU キュー上のサービスとして起動し、アダプターごとの
ガイドと rules に沿った英語プロンプト・尺・補足を JSON（json_schema で強制）で作らせる。
rules に違反したら、違反内容を伝えて同じジョブの中で再生成させる。
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import httpx
import yaml
from pydantic import BaseModel, Field

from .config import Settings
from .containers import ContainerError, ContainerManager, ServiceSpec
from .manifest import Manifest
from .prompt_rules import check, render_rules

log = logging.getLogger(__name__)

LLAMA_PORT = "8080/tcp"


class LlmModel(BaseModel):
    name: str
    gguf: str
    download: str = ""
    context: int = 16384
    thinking: bool = True
    reasoning_budget: int = 4096
    max_answer_tokens: int = 1024
    sampling: dict[str, float] = Field(default_factory=dict)
    vram_gb: float = 0
    load_time_s: float | None = None


class LlmConfig(BaseModel):
    image: str
    default: str
    models: dict[str, LlmModel]
    retry_limit: int = 3


def load_llm_config(path: Path) -> LlmConfig:
    return LlmConfig(**yaml.safe_load(path.read_text()))


def model_key(name: str) -> str:
    return f"llm:{name}"


def llm_spec(config: LlmConfig, name: str, settings: Settings) -> ServiceSpec:
    model = config.models[name]
    gguf = settings.models_dir / model.gguf
    if not gguf.exists():
        raise ContainerError("missing_asset", f"{gguf} がない。ダウンロード: {model.download}")
    rel = Path(model.gguf)
    command = [
        "-m", f"/models/{rel.as_posix()}",
        "--host", "0.0.0.0", "--port", LLAMA_PORT.split("/")[0],
        "-c", str(model.context),
        "-ngl", "999",
        "-np", "1",
        "--jinja",
        "--reasoning", "on" if model.thinking else "off",
        "--reasoning-format", "deepseek",
    ]  # fmt: skip
    if model.thinking:
        command += ["--reasoning-budget", str(model.reasoning_budget)]
    timeout = model.load_time_s * settings.timeouts.startup_factor if model.load_time_s else settings.timeouts.startup_default
    return ServiceSpec(
        key=model_key(name),
        image=config.image,
        role="llm",
        command=command,
        volumes={str(settings.models_dir / rel.parts[0]): {"bind": f"/models/{rel.parts[0]}", "mode": "ro"}},
        port=LLAMA_PORT,
        health="llama",
        startup_timeout=timeout,
    )


def output_schema(manifest: Manifest) -> dict[str, Any]:
    props: dict[str, Any] = {"prompt": {"type": "string"}, "notes": {"type": "string"}}
    required = ["prompt", "notes"]
    if manifest.duration:
        props["duration"] = {"type": "number"}
        required.insert(1, "duration")
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}


def system_prompt(manifest: Manifest) -> str:
    rules = render_rules(manifest.prompt.rules, manifest.duration)
    fields = [
        "- prompt: the prompt for the motion model",
        *(["- duration: the motion length in seconds that fits the request"] if manifest.duration else []),
        "- notes: one or two sentences in Japanese explaining how you interpreted the request "
        "(for example, what you simplified or left out to fit the model)",
    ]
    return "\n".join(
        [
            f'You convert a user\'s request (often written in Japanese) into a prompt for the text-to-motion model "{manifest.name}".',
            "Follow the model's prompt guide and the constraints exactly.",
            "",
            f"## Prompt guide for {manifest.name}",
            manifest.prompt.guide.strip(),
            "",
            "## Constraints (checked automatically)",
            *[f"- {r}" for r in rules],
            "",
            "## Output",
            "Return a JSON object with these fields:",
            *fields,
        ]
    )


class PromptResult(BaseModel):
    prompt: str
    duration: float | None
    notes: str
    violations: list[str]
    attempts: int
    reasoning: str
    seconds: float


def generate_prompt(
    base_url: str, config: LlmConfig, name: str, manifest: Manifest, request_text: str, timeout: float
) -> PromptResult:
    model = config.models[name]
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt(manifest)},
        {"role": "user", "content": request_text},
    ]
    schema = output_schema(manifest)
    started = time.monotonic()
    reasoning_parts: list[str] = []
    parsed: dict[str, Any] = {}
    violations: list[str] = []
    attempts = 0
    for attempts in range(1, config.retry_limit + 2):
        body = {
            "messages": messages,
            "response_format": {"type": "json_schema", "json_schema": {"name": "motion_prompt", "schema": schema, "strict": True}},
            "max_tokens": (model.reasoning_budget if model.thinking else 0) + model.max_answer_tokens,
            **model.sampling,
        }
        try:
            r = httpx.post(f"{base_url}/v1/chat/completions", json=body, timeout=timeout)
        except httpx.TimeoutException:
            raise ContainerError("internal", f"LLM request timed out after {timeout:.0f}s") from None
        if r.status_code != 200:
            raise ContainerError("internal", f"LLM error {r.status_code}: {r.text[:500]}")
        message = r.json()["choices"][0]["message"]
        content = message.get("content") or ""
        if message.get("reasoning_content"):
            reasoning_parts.append(message["reasoning_content"])
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            violations = [f"The output is not valid JSON: {content[:200]}"]
        else:
            duration = parsed.get("duration")
            violations = check(
                str(parsed.get("prompt", "")),
                float(duration) if isinstance(duration, (int, float)) else None,
                manifest.prompt.rules,
                manifest.duration,
            )
        if not violations:
            break
        log.info("prompt for %s violates rules (attempt %d): %s", manifest.id, attempts, violations)
        messages += [
            {"role": "assistant", "content": content},
            {
                "role": "user",
                "content": "The output violates these constraints:\n"
                + "\n".join(f"- {v}" for v in violations)
                + "\nFix them and return the JSON again.",
            },
        ]

    duration = parsed.get("duration")
    if isinstance(duration, (int, float)) and manifest.duration:
        step = manifest.duration.step or 0.1
        duration = round(round(float(duration) / step) * step, 3)
    return PromptResult(
        prompt=str(parsed.get("prompt", "")).strip(),
        duration=float(duration) if isinstance(duration, (int, float)) else None,
        notes=str(parsed.get("notes", "")).strip(),
        violations=violations,
        attempts=attempts,
        reasoning="\n\n---\n\n".join(reasoning_parts),
        seconds=time.monotonic() - started,
    )


class PromptService:
    """llm_prompt ジョブの処理と、プロンプト案の作成。"""

    def __init__(self, settings: Settings, config: LlmConfig, containers: ContainerManager) -> None:
        self.settings = settings
        self.config = config
        self.containers = containers

    def list_models(self) -> list[dict[str, Any]]:
        return [
            {
                "id": key,
                "name": m.name,
                "default": key == self.config.default,
                "available": (self.settings.models_dir / m.gguf).exists(),
                "download": m.download,
                "thinking": m.thinking,
                "vram_gb": m.vram_gb,
            }
            for key, m in self.config.models.items()
        ]
